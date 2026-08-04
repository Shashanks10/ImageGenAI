"""
train_lora.py

FLUX.1-schnell LoRA trainer with Flow Matching and Automatic Mixed Precision (AMP).
Fine-tunes LoRA weights on the FLUX Transformer for image style transfer.

Key differences from SD 1.5 training:
  - Uses Flow Matching instead of DDPM noise prediction
  - Operates on packed latent sequences (not spatial tensors)
  - Dual text encoders: CLIP (pooled) + T5 (sequence)
  - Pre-computes text embeddings to free ~10GB VRAM from T5

Requirements:
  - GPU with >=16GB VRAM (T4, A10G, A100, RTX 3090/4090)
  - ~32GB+ system RAM (for loading FLUX pipeline on CPU first)
  - HF_TOKEN environment variable (FLUX.1 is a gated model)
"""

import os
import sys
import re

# Set PyTorch memory allocator to avoid fragmentation on GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Fallback shim for Windows Application Control blocking regex._regex DLL
try:
    import regex
except Exception:
    sys.modules["regex"] = re

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ImageCaptionDataset, load_hf_dataset
from model import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "black-forest-labs/FLUX.1-schnell"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "output", "cool_posters_lora"))
HF_DATASET_NAME = os.environ.get("HF_DATASET_NAME", "AIGCDuckBoss/fluxlora_cool-posters")

BATCH_SIZE = 1       # Keep at 1 for memory efficiency on single GPU
EPOCHS = 70          # High epoch count to compensate for small dataset (4 images)
LR = 5e-5            # Lower LR for stable FLUX LoRA training
IMAGE_SIZE = 512
T5_MAX_LENGTH = 256  # FLUX.1-schnell default max sequence length for T5


# -------------------------------------------------------
# FLUX-specific Helper Functions
# -------------------------------------------------------

def pack_latents(latents):
    """Pack spatial latents into sequence format for FLUX Transformer.

    FLUX processes latents as a sequence of 2x2 patches, not spatial grids.
    This converts [B, C, H, W] -> [B, (H//2)*(W//2), C*4].

    Example for 512x512 image:
        VAE output:  [1, 16, 64, 64]
        Packed:      [1, 1024, 64]
    """
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
    return latents


def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    """Create 3D positional IDs for image latent patches.

    Each packed patch receives a (stream_id=0, row, col) position vector.
    These IDs are used by the FLUX Transformer's rotary position embeddings.
    """
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    return latent_image_ids.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)


def encode_prompt(tokenizer, tokenizer_2, text_encoder, text_encoder_2, captions, device, dtype):
    """Encode text prompts using both CLIP and T5 encoders.

    FLUX uses dual text encoders:
      - CLIP:  produces pooled embeddings [B, 768] for global conditioning
      - T5:    produces sequence embeddings [B, seq_len, 4096] for cross-attention

    Returns:
        prompt_embeds:       [B, seq_len, 4096] T5 sequence embeddings
        pooled_prompt_embeds: [B, 768] CLIP pooled embeddings
        text_ids:            [B, seq_len, 3] positional IDs (all zeros for text)
    """
    # CLIP encoding -> pooled embeddings (global style conditioning)
    clip_inputs = tokenizer(
        captions,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    clip_output = text_encoder(clip_inputs.input_ids, output_hidden_states=False)
    pooled_prompt_embeds = clip_output.pooler_output.to(dtype=dtype)

    # T5 encoding -> sequence embeddings (detailed text conditioning)
    t5_inputs = tokenizer_2(
        captions,
        padding="max_length",
        max_length=T5_MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    t5_output = text_encoder_2(t5_inputs.input_ids)
    prompt_embeds = t5_output[0].to(dtype=dtype)

    # Text positional IDs (all zeros — position info is in the T5 embeddings)
    text_ids = torch.zeros(
        prompt_embeds.shape[0], prompt_embeds.shape[1], 3,
        device=device, dtype=dtype,
    )

    return prompt_embeds, pooled_prompt_embeds, text_ids


# -------------------------------------------------------
# 1. Dataset & DataLoader
# -------------------------------------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_path = os.path.join(base_dir, "Dataset", "Images")
caption_path = os.path.join(base_dir, "Dataset", "Captions")

if HF_DATASET_NAME:
    print(f"Loading Hugging Face repository/dataset: '{HF_DATASET_NAME}'...")
    dataset = load_hf_dataset(
        HF_DATASET_NAME,
        image_size=IMAGE_SIZE,
        default_caption=(
            "cool_style, a stylish cool poster art, graphic vector illustration, "
            "high contrast black lineart, bold ink shadows, red graphic poster background"
        ),
        augment=True,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

elif os.path.exists(dataset_path) and os.path.exists(caption_path):
    print(f"Loading local dataset from '{dataset_path}'...")
    dataset = ImageCaptionDataset(
        image_dir=dataset_path,
        caption_dir=caption_path,
        image_size=IMAGE_SIZE,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

else:
    print(f"Warning: Neither HuggingFace dataset nor local folders ('{dataset_path}', '{caption_path}') found.")
    dataloader = []


# -------------------------------------------------------
# 2. Load FLUX Pipeline & LoRA Model
# -------------------------------------------------------
if len(dataloader) == 0:
    print("No training data available. Add images or set HF_DATASET_NAME. Exiting.")
    sys.exit(1)

# Pipeline loads to CPU first (see model.py) — avoids GPU OOM during download
pipe = load_model(MODEL_NAME, device=DEVICE)

transformer = pipe.transformer
vae = pipe.vae
text_encoder = pipe.text_encoder
text_encoder_2 = pipe.text_encoder_2
tokenizer = pipe.tokenizer
tokenizer_2 = pipe.tokenizer_2

# Determine weight dtype (matches what load_model sets)
if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
    weight_dtype = torch.bfloat16
elif DEVICE == "cuda":
    weight_dtype = torch.float16
else:
    weight_dtype = torch.float32

# VAE scaling factors (FLUX uses different values than SD 1.5)
vae_shift_factor = getattr(vae.config, "shift_factor", 0.0)
vae_scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)


# -------------------------------------------------------
# 3. Pre-compute Text Embeddings on CPU (saves ~10GB GPU VRAM)
# -------------------------------------------------------
# T5 alone is ~10GB — we CANNOT fit it on a 16GB T4 alongside the transformer.
# Strategy: run text encoders on CPU, cache embeddings, then only move
# VAE + Transformer to GPU for training.

print("Pre-computing text embeddings on CPU (T5 never touches GPU)...")

# Collect unique captions to avoid redundant encoding
unique_captions = list(set(dataset[i]["caption"] for i in range(len(dataset))))
print(f"Found {len(unique_captions)} unique caption(s) across {len(dataset)} images.")

cached_embeds = {}
with torch.no_grad():
    for caption in unique_captions:
        # Text encoders stay on CPU — encode on CPU
        prompt_embeds, pooled_embeds, text_ids = encode_prompt(
            tokenizer, tokenizer_2, text_encoder, text_encoder_2,
            [caption], "cpu", weight_dtype,
        )
        cached_embeds[caption] = {
            "prompt_embeds": prompt_embeds,
            "pooled_embeds": pooled_embeds,
            "text_ids": text_ids,
        }

print(f"Cached {len(cached_embeds)} text embedding(s).")

# Delete text encoders entirely to free CPU RAM (~10GB)
del text_encoder, text_encoder_2
pipe.text_encoder = None
pipe.text_encoder_2 = None
import gc; gc.collect()

# NOW move only VAE + Transformer to GPU (fits in 16GB T4)
if DEVICE == "cuda":
    print(f"Moving VAE + Transformer to {DEVICE}...")
    vae.to(DEVICE)
    transformer.to(DEVICE)
    torch.cuda.empty_cache()
    
    # Print GPU memory usage
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU VRAM: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

print("Ready for training.\n")


# -------------------------------------------------------
# 4. Optimizer & Scaler
# -------------------------------------------------------
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, transformer.parameters()),
    lr=LR,
    eps=1e-8,
    weight_decay=0.01,
)

# GradScaler is only needed for float16 (not bfloat16). With bf16, enabled=False
# makes the scaler a transparent pass-through.
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(DEVICE == "cuda" and weight_dtype == torch.float16),
)


# -------------------------------------------------------
# 5. Training Loop (Flow Matching)
# -------------------------------------------------------
total_steps = EPOCHS * len(dataloader)
print(f"Starting FLUX LoRA training for {EPOCHS} epochs ({len(dataset)} images, {len(dataloader)} steps/epoch)...")
print(f"Total training steps: {total_steps}\n")

for epoch in range(EPOCHS):
    transformer.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        images = batch["image"].to(DEVICE, dtype=weight_dtype)
        captions = batch["caption"]
        batch_size = images.shape[0]

        # --- Look up pre-computed text embeddings by caption ---
        prompt_embeds_list = []
        pooled_embeds_list = []
        text_ids_list = []

        for caption in captions:
            embeds = cached_embeds[caption]
            prompt_embeds_list.append(embeds["prompt_embeds"])
            pooled_embeds_list.append(embeds["pooled_embeds"])
            text_ids_list.append(embeds["text_ids"])

        prompt_embeds = torch.cat(prompt_embeds_list, dim=0).to(DEVICE, dtype=weight_dtype)
        pooled_embeds = torch.cat(pooled_embeds_list, dim=0).to(DEVICE, dtype=weight_dtype)
        text_ids = torch.cat(text_ids_list, dim=0).to(DEVICE, dtype=weight_dtype)

        # --- Encode images to latent space ---
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = (latents - vae_shift_factor) * vae_scaling_factor

        b, c, h, w = latents.shape

        # --- Pack latents into sequence format for Transformer ---
        packed_latents = pack_latents(latents)

        # --- Sample noise in packed space ---
        noise = torch.randn_like(packed_latents)

        # --- Sample timesteps (logit-normal distribution for flow matching) ---
        # Logit-normal provides better coverage across the full sigma range
        # compared to uniform sampling, improving training stability.
        u = torch.sigmoid(torch.randn(b, device=DEVICE, dtype=weight_dtype))
        sigmas = u.view(-1, 1, 1)  # [B, 1, 1] for sequence broadcasting

        # --- Flow matching interpolation ---
        # Creates a noisy version by interpolating between clean latents and noise:
        #   noisy = (1 - σ) · latents + σ · noise
        noisy_latents = (1.0 - sigmas) * packed_latents + sigmas * noise

        # Target velocity: the model learns to predict the "direction" from noise to data
        #   target = noise - latents
        target = noise - packed_latents

        # --- Prepare image positional IDs ---
        img_ids = prepare_latent_image_ids(b, h // 2, w // 2, DEVICE, weight_dtype)

        # --- Forward pass through FLUX Transformer ---
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda"), dtype=weight_dtype):
            model_pred = transformer(
                hidden_states=noisy_latents,
                timestep=u,  # Sigma values in [0, 1]
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]

            # Compute MSE loss in float32 for numerical stability
            loss = F.mse_loss(model_pred.float(), target.float())

        # --- Backward pass & optimizer step ---
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        print(
            f"Epoch: {epoch+1}/{EPOCHS} | "
            f"Step: {step+1}/{len(dataloader)} | "
            f"Loss: {loss.item():.6f} | "
            f"\u03c3: {u.mean().item():.3f}"
        )

    avg_epoch_loss = total_loss / max(len(dataloader), 1)
    print(f"--- Epoch {epoch+1} Complete | Avg Loss: {avg_epoch_loss:.6f} ---\n")


# -------------------------------------------------------
# 6. Save LoRA Weights
# -------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
transformer.save_pretrained(OUTPUT_DIR)
print(f"\n\U0001f389 Training Complete! Saved LoRA weights to '{OUTPUT_DIR}'")
print(f"   Load in inference with: pipe.load_lora_weights('{OUTPUT_DIR}')")
