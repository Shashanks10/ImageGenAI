"""
model.py

FLUX.1-schnell LoRA model loader.
Loads the FLUX pipeline, freezes base components (VAE, Text Encoders, Transformer),
and attaches PEFT LoRA layers to the Transformer's attention modules.

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
from diffusers import FluxPipeline
from peft import LoraConfig, get_peft_model

MODEL_NAME = "black-forest-labs/FLUX.1-schnell"


def load_model(model_name: str = MODEL_NAME, device: str = None):
    """
    Loads FLUX.1-schnell pipeline, freezes base components (VAE, Text Encoders, Transformer),
    and attaches PEFT LoRA layers to Transformer cross-attention modules.

    Args:
        model_name: HuggingFace model ID for the FLUX pipeline.
        device: Target device ('cuda' or 'cpu'). Auto-detects if None.

    Returns:
        FluxPipeline with LoRA-wrapped Transformer.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # FLUX prefers bfloat16; fall back to float16 if bf16 not supported
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    hf_token = os.environ.get("HF_TOKEN", None)

    print(f"Loading FLUX.1 pipeline '{model_name}' on {device} ({dtype})...")
    pipe = FluxPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        token=hf_token,
    )

    # Disable safety checker if present
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    # Freeze all base pipeline components
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    # LoRA Configuration (attaches to transformer cross-attention layers)
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

    # Attach PEFT LoRA to FLUX Transformer (not UNet — FLUX uses a Transformer)
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)

    pipe.to(device)

    print("\n--- Trainable Parameters (LoRA) ---")
    pipe.transformer.print_trainable_parameters()
    print("-----------------------------------\n")

    return pipe