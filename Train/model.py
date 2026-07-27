# Load SD + attach LoRA
import sys
import re

# Fallback shim for Windows environments where AppLocker/Application Control blocks regex._regex binary DLL
try:
    import regex
except Exception:
    sys.modules["regex"] = re

import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

MODEL_NAME = "runwayml/stable-diffusion-v1-5"


def load_model(model_name: str = MODEL_NAME, device: str = None):
    """
    Loads Stable Diffusion pipeline, freezes base components (VAE, Text Encoder, UNet),
    and attaches PEFT LoRA layers to UNet cross-attention modules.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading base pipeline '{model_name}' on {device} ({dtype})...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
        safety_checker=None,
    )

    # Disable safety checker
    pipe.safety_checker = None

    # Freeze base pipeline components
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # LoRA Configuration (attaches to cross-attention layers)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
        ],
        lora_dropout=0.1,
        bias="none",
    )

    # Attach PEFT LoRA to U-Net
    pipe.unet = get_peft_model(pipe.unet, lora_config)

    pipe.to(device)

    print("\n--- Trainable Parameters ---")
    pipe.unet.print_trainable_parameters()
    print("----------------------------\n")

    return pipe