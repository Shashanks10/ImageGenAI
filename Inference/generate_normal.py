"""
generate_normal.py

Inference script for FLUX.2-dev standard image generation without LoRA.
Uses black-forest-labs/FLUX.2-dev with CPU offloading and fallbacks.
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
from diffusers import FluxPipeline
try:
    from diffusers import Flux2Pipeline
except ImportError:
    Flux2Pipeline = None


class NormalGenerator:
    """
    Standard Text-to-Image generation pipeline using FLUX.2-dev.
    """

    def __init__(
        self,
        base_model_name: str = "black-forest-labs/FLUX.2-dev",
        device: str = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Initializing NormalGenerator (FLUX.2-dev) on device: {self.device}")

        dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if self.device == "cuda" else torch.float32
        )

        hf_token = os.environ.get("HF_TOKEN", None)
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not set.\n"
                "FLUX.2-dev is a gated model. Run: export HF_TOKEN='your_hf_token_here'"
            )

        pipeline_cls = Flux2Pipeline if Flux2Pipeline is not None else FluxPipeline

        try:
            print(f"Loading {base_model_name} with {pipeline_cls.__name__}...")
            self.pipe = pipeline_cls.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                token=hf_token,
            )
        except Exception as e:
            print(f"Warning: Loading {base_model_name} with {pipeline_cls.__name__} failed ({e}). Trying FluxPipeline fallback...")
            try:
                self.pipe = FluxPipeline.from_pretrained(
                    base_model_name,
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
                print("Enabling model CPU offloading for FLUX.2-dev VRAM optimization...")
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to(self.device)
            torch.cuda.empty_cache()
            gc.collect()
        else:
            self.pipe.to("cpu")

        print("NormalGenerator (FLUX.2-dev) ready!\n")

    def generate(
        self,
        prompt: str,
        guidance_scale: float = 4.0,
        num_inference_steps: int = 25,
        width: int = 512,
        height: int = 512,
    ) -> Image.Image:
        base_prompt = prompt.strip() if (prompt and prompt.strip()) else "a beautiful high quality photo"

        with torch.no_grad():
            print(f"Running FLUX.2-dev Text-to-Image generation with prompt: '{base_prompt[:80]}...'")
            print(f"  steps={num_inference_steps} | size={width}x{height}")

            output = self.pipe(
                prompt=base_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            ).images[0]

        return output
