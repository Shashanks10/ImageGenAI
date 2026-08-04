"""
generate.py

Inference script for FLUX.1 Cool Poster Image-to-Image style conversion using pre-trained FLUX LoRA weights.
Supports image transformation WITH or WITHOUT custom text prompts.
"""

import os
import sys
import re

# Fallback shim for Windows Application Control blocking regex._regex DLL
try:
    import regex
except Exception:
    sys.modules["regex"] = re

import torch
from PIL import Image
from diffusers import FluxImg2ImgPipeline, AutoPipelineForImage2Image

# Default Cool Poster trigger prompt used when no prompt is provided
DEFAULT_POSTER_PROMPT = (
    "cool_style, a stylish cool poster art of a person, graphic vector illustration, "
    "high contrast black lineart, bold ink shadows, red graphic poster background"
)

DEFAULT_NEGATIVE_PROMPT = (
    "photograph, real life photo, 3d render, realistic skin, camera shot, "
    "deformed face, distorted eyes, bad anatomy, bad facial features, disfigured face, "
    "mutated eyes, ugly face, unnatural expressions, blurry, low quality"
)


class PosterGenerator:
    """
    Image-to-Image transformation pipeline using FLUX.1 with FLUX Cool Poster LoRA.
    """

    def __init__(
        self,
        base_model_name: str = "black-forest-labs/FLUX.1-schnell",
        lora_repo_id: str = "AIGCDuckBoss/fluxlora_cool-posters",
        weight_name: str = "flux_cool_poster.safetensors",
        device: str = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.is_flux = "flux" in base_model_name.lower()
        print(f"Loading Img2Img base model '{base_model_name}' on {self.device}...")

        dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if self.device == "cuda" else torch.float32
        )

        hf_token = os.environ.get("HF_TOKEN", None)

        if self.is_flux:
            try:
                self.pipe = FluxImg2ImgPipeline.from_pretrained(
                    base_model_name,
                    torch_dtype=dtype,
                    token=hf_token,
                )
            except Exception as e:
                if "401" in str(e) or "GatedRepoError" in type(e).__name__ or "gated" in str(e).lower():
                    print(
                        "\n⚠️ HUGGING FACE AUTHENTICATION REQUIRED FOR FLUX.1 ⚠️\n"
                        "FLUX.1-schnell / FLUX.1-dev are gated repositories on Hugging Face.\n"
                        "To fix this on your EC2 instance:\n"
                        "1. Go to https://huggingface.co/black-forest-labs/FLUX.1-schnell (click 'Accept' once)\n"
                        "2. Get your free token at https://huggingface.co/settings/tokens\n"
                        "3. On EC2 run: export HF_TOKEN='your_hf_token_here'\n"
                    )
                    raise e
                print(f"Notice: Could not load '{base_model_name}' ({e}). Trying 'black-forest-labs/FLUX.1-schnell'...")
                self.pipe = FluxImg2ImgPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-schnell",
                    torch_dtype=dtype,
                    token=hf_token,
                )
        else:
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                safety_checker=None,
                token=hf_token,
            )


        # Load FLUX LoRA adapter
        print(f"Loading FLUX Cool Posters LoRA weights from '{lora_repo_id}'...")
        try:
            self.pipe.load_lora_weights(lora_repo_id, weight_name=weight_name, token=hf_token)
        except Exception as e:

            print(f"Notice: Could not load LoRA directly ({e}). Checking local files...")
            cool_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "Train", "output", "cool_posters_lora")
            )
            if os.path.exists(cool_path):
                self.pipe.load_lora_weights(cool_path)

        if self.device == "cuda":
            try:
                self.pipe.enable_model_cpu_offload()
            except Exception:
                self.pipe.to(self.device)
        else:
            self.pipe.to("cpu")

    def convert(
        self,
        input_image: Image.Image | str,
        prompt: str = None,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        strength: float = 0.65,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 20,
    ) -> Image.Image:
        """
        Transforms input photo into FLUX cool poster graphic art.

        Args:
            input_image: PIL Image object or file path.
            prompt: Optional user prompt. If None or empty, default cool_style prompt is applied.
            negative_prompt: Elements to avoid in the output image.
            strength: Img2Img denoising strength (0.60-0.75 recommended for FLUX graphic style transfer).
            guidance_scale: Guidance scale.
            num_inference_steps: Denoising steps.

        Returns:
            PIL.Image: Graphic poster styled output image.
        """
        final_prompt = prompt.strip() if (prompt and prompt.strip()) else DEFAULT_POSTER_PROMPT

        if isinstance(input_image, str):
            if not os.path.exists(input_image):
                raise FileNotFoundError(f"Input image file '{input_image}' does not exist.")
            image = Image.open(input_image).convert("RGB")
        else:
            image = input_image.convert("RGB")

        image = image.resize((512, 512))

        print(f"Processing image with prompt: '{final_prompt[:60]}...'")
        print(f"Img2Img Strength: {strength}")

        kwargs = {
            "prompt": final_prompt,
            "image": image,
            "strength": strength,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }

        output = self.pipe(**kwargs).images[0]
        return output


if __name__ == "__main__":
    print("FLUX Cool Poster Img2Img Generator ready!")
    print("Example usage in your app/API:")
    print("  generator = PosterGenerator()")
    print("  styled_img = generator.convert('my_photo.jpg')  # Automatic (cool_style prompt)")
    print("  styled_img = generator.convert('my_photo.jpg', prompt='cool_style, poster art')")
