"""
generate.py

Inference script for FLUX.1 Cool Poster Image-to-Image style conversion.
Uses the pre-trained FLUX Cool Poster LoRA from HuggingFace + 4-bit quantized Transformer.

Strategy:
  1. Load FLUX Transformer in 4-bit nf4 (~6GB VRAM instead of ~24GB)
  2. Load pipeline with quantized Transformer
  3. Load pre-trained cool poster LoRA via load_lora_weights (diffusers-native)
  4. Use enable_sequential_cpu_offload for VAE/TextEncoders, keep Transformer on GPU
  5. Run Img2Img with strength=0.85 and lora_scale=1.0

This approach uses the properly-trained HuggingFace LoRA which produces real poster style.
The locally-trained LoRA (4 images, 70 epochs) is too weak for visible transformation.
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
from diffusers import FluxImg2ImgPipeline, FluxTransformer2DModel, AutoPipelineForImage2Image
from transformers import BitsAndBytesConfig

# Cool Poster LoRA trigger word — MUST be in every prompt to activate the style
DEFAULT_POSTER_PROMPT = (
    "cool_style, a stylish cool poster art of a person, graphic vector illustration, "
    "high contrast black lineart, bold ink shadows, red graphic poster background"
)


class PosterGenerator:
    """
    Image-to-Image transformation pipeline using FLUX.1-schnell with
    the pre-trained FLUX Cool Poster LoRA.
    """

    def __init__(
        self,
        base_model_name: str = "black-forest-labs/FLUX.1-dev",
        lora_repo_id: str = "AIGCDuckBoss/fluxlora_cool-posters",
        weight_name: str = "flux_cool_poster.safetensors",
        device: str = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.is_flux = "flux" in base_model_name.lower()
        print(f"Initializing PosterGenerator on device: {self.device}")

        dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if self.device == "cuda" else torch.float32
        )

        hf_token = os.environ.get("HF_TOKEN", None)
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not set.\n"
                "FLUX.1-dev is a gated model. Run: export HF_TOKEN='your_hf_token_here'"
            )

        # -------------------------------------------------------
        # Step 1: Load FLUX Transformer in 4-bit nf4 (saves VRAM)
        # -------------------------------------------------------
        if self.device == "cuda" and torch.cuda.is_available():
            print("Loading FLUX Transformer in 4-bit nf4 quantization (~6GB VRAM)...")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            transformer = FluxTransformer2DModel.from_pretrained(
                base_model_name,
                subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=dtype,
                token=hf_token,
            )
        else:
            print(f"Loading FLUX Transformer in {dtype} (CPU mode)...")
            transformer = FluxTransformer2DModel.from_pretrained(
                base_model_name,
                subfolder="transformer",
                torch_dtype=dtype,
                token=hf_token,
            )

        # -------------------------------------------------------
        # Step 2: Build FluxImg2ImgPipeline with quantized Transformer
        # -------------------------------------------------------
        print(f"Building FluxImg2ImgPipeline...")
        self.pipe = FluxImg2ImgPipeline.from_pretrained(
            base_model_name,
            transformer=transformer,
            torch_dtype=dtype,
            token=hf_token,
        )

        # -------------------------------------------------------
        # Step 3: Load the pre-trained Cool Poster LoRA weights
        # -------------------------------------------------------
        # The remote HuggingFace LoRA is properly trained on hundreds of poster images.
        # The locally-trained LoRA (4 images) is NOT strong enough for visible transformation.
        print(f"Loading pre-trained Cool Poster LoRA from '{lora_repo_id}'...")
        try:
            self.pipe.load_lora_weights(
                lora_repo_id,
                weight_name=weight_name,
                token=hf_token,
            )
            print("Successfully loaded pre-trained Cool Poster LoRA!")
        except Exception as e:
            print(f"Warning: Could not load remote LoRA weights ({e}).")
            # Fallback: try loading local LoRA if available
            cool_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "Train", "output", "cool_posters_lora")
            )
            if os.path.exists(cool_path):
                print(f"Falling back to local LoRA weights from '{cool_path}'...")
                try:
                    self.pipe.load_lora_weights(cool_path)
                except Exception as local_err:
                    print(f"Warning: Could not load local LoRA either: {local_err}")

        # -------------------------------------------------------
        # Step 4: Move components to GPU
        # -------------------------------------------------------
        # IMPORTANT: 4-bit bitsandbytes models are INCOMPATIBLE with
        # enable_sequential_cpu_offload() / enable_model_cpu_offload().
        # Those methods install Accelerate hooks that try to move 4-bit
        # tensors through meta tensors — causing "Cannot copy out of meta tensor".
        #
        # A10G has 24GB VRAM. Components fit easily:
        #   4-bit Transformer  : ~6.1 GB  (already on CUDA from from_pretrained)
        #   T5 text encoder    : ~10.2 GB
        #   CLIP text encoder  : ~0.6 GB
        #   VAE                : ~0.3 GB
        #   ─────────────────────────────
        #   Total              : ~17.2 GB  ✓ (within 24 GB)
        if self.device == "cuda":
            print("Moving pipeline components to GPU (no offloading — incompatible with 4-bit)...")
            # Transformer is already on CUDA from from_pretrained with quantization_config.
            # Only move the remaining components explicitly.
            self.pipe.vae.to(self.device)
            self.pipe.text_encoder.to(self.device)
            self.pipe.text_encoder_2.to(self.device)
            torch.cuda.empty_cache()
            gc.collect()
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved  = torch.cuda.memory_reserved()  / 1024**3
            print(f"GPU VRAM: {allocated:.1f} GB allocated / {reserved:.1f} GB reserved (24 GB total)")
        else:
            self.pipe.to("cpu")

        print("PosterGenerator ready!\n")

    def convert(
        self,
        input_image: Image.Image | str,
        prompt: str = None,
        strength: float = 0.85,
        guidance_scale: float = 3.5,   # FLUX.1-dev supports CFG (unlike schnell which needs 0.0)
        num_inference_steps: int = 25, # dev works well at 20-30 steps
        lora_scale: float = 1.0,
    ) -> Image.Image:
        """
        Transforms an input photo into FLUX Cool Poster graphic art.

        Args:
            input_image: PIL Image or file path.
            prompt: Optional text prompt. 'cool_style' trigger word is always prepended.
            strength: Img2Img denoising strength. Higher = more stylized (0.75-0.90 recommended).
            guidance_scale: How strongly to follow the prompt (3.5 is good for FLUX).
            num_inference_steps: Total diffusion steps (more = sharper result).
            lora_scale: LoRA adapter influence (1.0 = full effect).

        Returns:
            PIL.Image: Graphic poster styled output image.
        """
        # Always ensure trigger word 'cool_style' is at the front to activate LoRA
        if prompt and prompt.strip():
            base_prompt = prompt.strip()
            if "cool_style" not in base_prompt:
                final_prompt = f"cool_style, {base_prompt}"
            else:
                final_prompt = base_prompt
        else:
            final_prompt = DEFAULT_POSTER_PROMPT

        if isinstance(input_image, str):
            if not os.path.exists(input_image):
                raise FileNotFoundError(f"Input image file '{input_image}' does not exist.")
            image = Image.open(input_image).convert("RGB")
        else:
            image = input_image.convert("RGB")

        image = image.resize((512, 512))

        print(f"Generating poster with prompt: '{final_prompt[:80]}...'")
        print(f"  strength={strength} | steps={num_inference_steps} | lora_scale={lora_scale}")

        with torch.no_grad():
            output = self.pipe(
                prompt=final_prompt,
                image=image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                joint_attention_kwargs={"scale": lora_scale},
            ).images[0]

        return output


if __name__ == "__main__":
    print("FLUX Cool Poster Img2Img Generator")
    print("Usage:")
    print("  generator = PosterGenerator()")
    print("  output = generator.convert('photo.jpg')")
    print("  output.save('poster.jpg')")
