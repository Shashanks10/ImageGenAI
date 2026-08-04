"""
model.py

FLUX.1-schnell LoRA model loader with 4-bit quantization (nf4).
Loads the FLUX pipeline, freezes base components (VAE, Text Encoders, Transformer),
quantizes the Transformer to 4-bit, and attaches PEFT LoRA layers.

Optimized for A10G (24GB VRAM) / T4 (16GB VRAM) GPUs.
Requires HF_TOKEN environment variable (FLUX.1 is a gated model).
"""

import sys
import re
import os

# Fallback shim for Windows environments where AppLocker/Application Control blocks regex._regex binary DLL
try:
    import regex
except Exception:
    sys.modules["regex"] = re

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

MODEL_NAME = "black-forest-labs/FLUX.1-schnell"


def load_model(model_name: str = MODEL_NAME, device: str = None):
    """
    Loads FLUX.1-schnell pipeline with 4-bit quantized Transformer + LoRA.

    Memory strategy for A10G (24GB VRAM):
      1. Load Transformer separately in 4-bit nf4 (~6GB instead of ~24GB)
      2. Load rest of pipeline (VAE + text encoders) in fp16/bf16
      3. Freeze everything, attach LoRA to quantized Transformer
      4. Enable gradient checkpointing

    The pipeline stays on CPU. train_lora.py moves components to GPU selectively.

    Args:
        model_name: HuggingFace model ID for the FLUX pipeline.
        device: Target device ('cuda' or 'cpu'). Auto-detects if None.

    Returns:
        FluxPipeline with quantized, LoRA-wrapped Transformer.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute dtype: bfloat16 preferred for FLUX, fallback to float16
    if device == "cuda" and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    elif device == "cuda":
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    hf_token = os.environ.get("HF_TOKEN", None)

    # -------------------------------------------------------
    # Step 1: Load Transformer in 4-bit quantization (nf4)
    # -------------------------------------------------------
    # The FLUX Transformer is ~12B params (~24GB in fp16).
    # 4-bit nf4 quantization compresses it to ~6GB — fits on A10G/T4.
    print(f"Loading FLUX Transformer in 4-bit nf4 quantization...")
    print(f"  (~6GB instead of ~24GB — fits on A10G/T4)")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,  # Nested quantization for extra savings
    )

    transformer = FluxTransformer2DModel.from_pretrained(
        model_name,
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        token=hf_token,
    )

    # -------------------------------------------------------
    # Step 2: Load rest of pipeline (VAE + text encoders) normally
    # -------------------------------------------------------
    print(f"Loading rest of FLUX pipeline (VAE + text encoders) in {compute_dtype}...")

    pipe = FluxPipeline.from_pretrained(
        model_name,
        transformer=transformer,  # Use our quantized transformer
        torch_dtype=compute_dtype,
        token=hf_token,
    )

    # Disable safety checker if present
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    # -------------------------------------------------------
    # Step 3: Freeze everything + attach LoRA
    # -------------------------------------------------------
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    # LoRA on top of quantized Transformer (QLoRA pattern)
    # Only the small LoRA adapter weights are trainable in full precision.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
        ],
        lora_dropout=0.05,
        bias="none",
    )

    pipe.transformer = get_peft_model(pipe.transformer, lora_config)

    # -------------------------------------------------------
    # Step 4: Gradient checkpointing (saves ~40% activation VRAM)
    # -------------------------------------------------------
    if hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        pipe.transformer.enable_gradient_checkpointing()
        print("Enabled gradient checkpointing (saves ~40% VRAM during training).")
    elif hasattr(pipe.transformer, "gradient_checkpointing_enable"):
        pipe.transformer.gradient_checkpointing_enable()
        print("Enabled gradient checkpointing (saves ~40% VRAM during training).")

    # Pipeline stays on CPU — train_lora.py moves VAE + Transformer to GPU
    print("\n--- Trainable Parameters (LoRA on 4-bit Transformer) ---")
    pipe.transformer.print_trainable_parameters()
    print("-------------------------------------------------------\n")

    return pipe