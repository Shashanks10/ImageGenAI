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
from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxTransformer2DModel, AutoPipelineForImage2Image
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
        # Step 2: Build FluxPipeline (T2I) and FluxImg2ImgPipeline (Img2Img)
        # -------------------------------------------------------
        print(f"Building FluxPipeline (Text-to-Image)...")
        self.t2i_pipe = FluxPipeline.from_pretrained(
            base_model_name,
            transformer=transformer,
            torch_dtype=dtype,
            token=hf_token,
        )

        print(f"Building FluxImg2ImgPipeline (Image-to-Image)...")
        self.img2img_pipe = FluxImg2ImgPipeline(
            vae=self.t2i_pipe.vae,
            text_encoder=self.t2i_pipe.text_encoder,
            text_encoder_2=self.t2i_pipe.text_encoder_2,
            tokenizer=self.t2i_pipe.tokenizer,
            tokenizer_2=self.t2i_pipe.tokenizer_2,
            transformer=self.t2i_pipe.transformer,
            scheduler=self.t2i_pipe.scheduler,
        )

        # Backward compatibility handle
        self.pipe = self.t2i_pipe

        # -------------------------------------------------------
        # Step 3: Load the pre-trained Cool Poster LoRA weights
        # -------------------------------------------------------
        print(f"Loading pre-trained Cool Poster LoRA from '{lora_repo_id}'...")
        try:
            self.t2i_pipe.load_lora_weights(
                lora_repo_id,
                weight_name=weight_name,
                token=hf_token,
            )
            print("Successfully loaded pre-trained Cool Poster LoRA!")
        except Exception as e:
            print(f"Warning: Could not load remote LoRA weights ({e}).")
            cool_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "Train", "output", "cool_posters_lora")
            )
            if os.path.exists(cool_path):
                print(f"Falling back to local LoRA weights from '{cool_path}'...")
                try:
                    self.t2i_pipe.load_lora_weights(cool_path)
                except Exception as local_err:
                    print(f"Warning: Could not load local LoRA either: {local_err}")

        # -------------------------------------------------------
        # Step 4: Move components to GPU
        # -------------------------------------------------------
        if self.device == "cuda":
            print("Moving pipeline components to GPU...")
            self.t2i_pipe.vae.to(self.device)
            self.t2i_pipe.text_encoder.to(self.device)
            self.t2i_pipe.text_encoder_2.to(self.device)
            torch.cuda.empty_cache()
            gc.collect()
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved  = torch.cuda.memory_reserved()  / 1024**3
            print(f"GPU VRAM: {allocated:.1f} GB allocated / {reserved:.1f} GB reserved (24 GB total)")
        else:
            self.t2i_pipe.to("cpu")

        print("PosterGenerator ready (supports both Text-to-Image and Image-to-Image)!\n")

    def generate(
        self,
        prompt: str = None,
        input_image: Image.Image | str = None,
        strength: float = 0.85,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 25,
        lora_scale: float = 1.0,
        width: int = 512,
        height: int = 512,
        use_lora_trigger: bool = True,
        use_lora: bool = True,
    ) -> Image.Image:
        """
        Generates an image using FLUX.1.
        - If input_image is provided: Runs Image-to-Image style transfer.
        - If input_image is None: Runs Text-to-Image generation from scratch.

        Args:
            prompt: Text prompt describing the image to generate.
            input_image: Optional PIL Image or file path for Img2Img.
            strength: Img2Img denoising strength (0.75-0.90 recommended).
            guidance_scale: Guidance scale (3.5 default for FLUX.1-dev).
            num_inference_steps: Total diffusion steps.
            lora_scale: LoRA influence.
            width: Image width for T2I.
            height: Image height for T2I.
            use_lora_trigger: Whether to auto-prepend trigger word 'cool_style'.
            use_lora: Whether to apply LoRA / trained style data. Set False for normal images.

        Returns:
            PIL.Image: Generated image.
        """
        base_prompt = prompt.strip() if (prompt and prompt.strip()) else "a cat and a dog fighting in an epic dramatic poster style"

        if use_lora and use_lora_trigger and "cool_style" not in base_prompt:
            final_prompt = f"cool_style, {base_prompt}"
        else:
            final_prompt = base_prompt

        effective_lora_scale = lora_scale if use_lora else 0.0

        with torch.no_grad():
            if input_image is not None:
                # --- Image-to-Image Mode ---
                if isinstance(input_image, str):
                    if not os.path.exists(input_image):
                        raise FileNotFoundError(f"Input image file '{input_image}' does not exist.")
                    image = Image.open(input_image).convert("RGB")
                else:
                    image = input_image.convert("RGB")

                image = image.resize((width, height))
                print(f"Running Img2Img generation (use_lora={use_lora}) with prompt: '{final_prompt[:80]}...'")
                print(f"  strength={strength} | steps={num_inference_steps}")

                output = self.img2img_pipe(
                    prompt=final_prompt,
                    image=image,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    joint_attention_kwargs={"scale": effective_lora_scale},
                ).images[0]
            else:
                # --- Text-to-Image Mode (from scratch) ---
                print(f"Running Text-to-Image generation (use_lora={use_lora}) with prompt: '{final_prompt[:80]}...'")
                print(f"  steps={num_inference_steps} | size={width}x{height}")

                output = self.t2i_pipe(
                    prompt=final_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    joint_attention_kwargs={"scale": effective_lora_scale},
                ).images[0]

        return output

    def convert(self, *args, **kwargs) -> Image.Image:
        """Alias for generate() for backward compatibility."""
        return self.generate(*args, **kwargs)


if __name__ == "__main__":
    print("FLUX Generator ready (Text-to-Image and Image-to-Image)")
    print("Usage:")
    print("  generator = PosterGenerator()")
    print("  # Text-to-Image:")
    print("  img = generator.generate(prompt='a cat and a dog fighting')")
    print("  # Image-to-Image:")
    print("  img = generator.generate(prompt='cool poster', input_image='my_photo.jpg')")
