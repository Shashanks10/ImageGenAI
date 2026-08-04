"""
train_lora.py

Stable Diffusion LoRA trainer with Automatic Mixed Precision (AMP) and 4GB VRAM optimization.
Fine-tunes LoRA weights for image style transfer.
"""

import os
import sys
import re

# Set PyTorch memory allocator to avoid fragmentation on 4GB GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Fallback shim for Windows Application Control blocking regex._regex DLL
try:
    import regex
except Exception:
    sys.modules["regex"] = re

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ImageCaptionDataset, HFImageCaptionDataset, load_hf_dataset
from model import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "output", "cool_posters_lora"))
HF_DATASET_NAME = os.environ.get("HF_DATASET_NAME", "AIGCDuckBoss/fluxlora_cool-posters")

BATCH_SIZE = 1  # Ideal for 4GB VRAM GPUs (Quadro T1000, RTX 3050)
EPOCHS = 5
LR = 1e-4

# -----------------------------
# 1. Dataset & DataLoader
# -----------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_path = os.path.join(base_dir, "Dataset", "Images")
caption_path = os.path.join(base_dir, "Dataset", "Captions")

if HF_DATASET_NAME:
    print(f"Loading Hugging Face repository/dataset: '{HF_DATASET_NAME}'...")
    dataset = load_hf_dataset(HF_DATASET_NAME, image_size=512, default_caption="a cool poster")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

elif os.path.exists(dataset_path) and os.path.exists(caption_path):
    print(f"Loading local dataset from '{dataset_path}'...")
    dataset = ImageCaptionDataset(
        image_dir=dataset_path,
        caption_dir=caption_path,
        image_size=512,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
else:
    print(f"Warning: Neither HuggingFace dataset nor local folders ('{dataset_path}', '{caption_path}') found.")
    dataloader = []


# -----------------------------
# 2. Load Pipeline & LoRA Model
# -----------------------------
pipe = load_model(MODEL_NAME, device=DEVICE)

tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
vae = pipe.vae
unet = pipe.unet
noise_scheduler = pipe.scheduler

# Use float16 on CUDA for base models
weight_dtype = torch.float16 if DEVICE == "cuda" else torch.float32

# Set up optimizer and AMP GradScaler to prevent NaN loss
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, unet.parameters()),
    lr=LR,
    eps=1e-8,
)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

# -----------------------------
# 3. Training Loop
# -----------------------------
if len(dataloader) > 0:
    print(f"Starting LoRA training for {EPOCHS} epochs on {DEVICE}...")
    for epoch in range(EPOCHS):
        unet.train()
        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            images = batch["image"].to(DEVICE, dtype=weight_dtype)
            captions = batch["caption"]

            tokenized = tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )

            with torch.no_grad():
                text_embeddings = text_encoder(
                    tokenized.input_ids.to(DEVICE)
                ).last_hidden_state.to(dtype=weight_dtype)

                latents = vae.encode(images).latent_dist.sample()
                latents = latents * 0.18215

            noise = torch.randn_like(latents)

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=DEVICE,
            ).long()

            noisy_latents = noise_scheduler.add_noise(
                latents,
                noise,
                timesteps,
            )

            # Mixed Precision Forward Pass (prevents NaN loss)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda"), dtype=torch.float16):
                noise_prediction = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                ).sample

                # Compute MSE loss in float32 for numerical stability
                loss = F.mse_loss(
                    noise_prediction.float(),
                    noise.float(),
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            print(
                f"Epoch: {epoch+1}/{EPOCHS} | "
                f"Step: {step+1}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f}"
            )

        avg_epoch_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Complete | Avg Loss: {avg_epoch_loss:.4f} ---")

    # -----------------------------
    # 4. Save LoRA Weights
    # -----------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unet.save_pretrained(OUTPUT_DIR)
    print(f"\n🎉 Training Complete! Saved LoRA weights to '{OUTPUT_DIR}'")
else:
    print("Add training images & captions to Dataset/ directory to start training.")
