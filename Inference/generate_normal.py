"""
generate_normal.py

Inference script for standard image generation using Stable Diffusion 3.5 Medium.
Uses stabilityai/stable-diffusion-3.5-medium with CPU offloading and fallbacks.
"""

import os
import sys
import re
import gc

# Fallback shim for Windows Application Control blocking regex._regex DLL
try:
    import regex
except Exception:
    sys.modules["regex"] = re

import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline, FluxPipeline


class NormalGenerator:
    """
    Standard Text-to-Image generation pipeline using Stable Diffusion 3.5 Medium.
    """

    def __init__(
        self,
        base_model_name: str = "stabilityai/stable-diffusion-3.5-medium",
        device: str = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Initializing NormalGenerator (SD 3.5 Medium) on device: {self.device}")

        dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if self.device == "cuda" else torch.float32
        )

        hf_token = os.environ.get("HF_TOKEN", None)

        try:
            print(f"Loading {base_model_name} with StableDiffusion3Pipeline...")
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                token=hf_token,
            )
        except Exception as e:
            print(f"Warning: Loading {base_model_name} failed ({e}). Trying SD 3.5 Large fallback...")
            try:
                self.pipe = StableDiffusion3Pipeline.from_pretrained(
                    "stabilityai/stable-diffusion-3.5-large",
                    torch_dtype=dtype,
                    token=hf_token,
                )
            except Exception as e2:
                print(f"Fallback to FLUX.1-dev for base image generation ({e2})")
                self.pipe = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-dev",
                    torch_dtype=dtype,
                    token=hf_token,
                )

        if self.device == "cuda":
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                print("Enabling model CPU offloading for SD 3.5 VRAM optimization...")
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to(self.device)
            torch.cuda.empty_cache()
            gc.collect()
        else:
            self.pipe.to("cpu")

        print("NormalGenerator (SD 3.5 Medium) ready!\n")

    def generate(
        self,
        prompt: str,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 28,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        base_prompt = prompt.strip() if (prompt and prompt.strip()) else "a high quality detailed photo"

        with torch.no_grad():
            print(f"Running SD 3.5 Medium Text-to-Image generation with prompt: '{base_prompt[:80]}...'")
            print(f"  steps={num_inference_steps} | size={width}x{height}")

            output = self.pipe(
                prompt=base_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            ).images[0]

        return output
