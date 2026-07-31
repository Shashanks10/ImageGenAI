"""
generate.py

Inference script for Peaky Blinders Image-to-Image style conversion.
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
from diffusers import AutoPipelineForImage2Image

# Default Peaky Blinders trigger prompt used when no prompt is provided
DEFAULT_PEAKY_PROMPT = (
    "photo in peaky blinders style, 1920s cinematic mood, dark desaturated color grade, "
    "high contrast, atmospheric lighting, detailed portrait, gritty vintage aesthetic"
)

DEFAULT_NEGATIVE_PROMPT = (
    "deformed face, distorted eyes, bad anatomy, bad facial features, disfigured face, "
    "mutated eyes, ugly face, unnatural expressions, oversaturated, cartoon, 3d render, "
    "low quality, blurry, distorted"
)



class PeakyBlindersGenerator:
    """
    Image-to-Image transformation pipeline with LoRA style adapter.
    """

    def __init__(
        self,
        base_model_name: str = "runwayml/stable-diffusion-v1-5",
        lora_weights_path: str = None,
        device: str = None,
    ):
        if lora_weights_path is None:
            cool_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "Train", "output", "cool_posters_lora")
            )
            peaky_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "Train", "output", "peaky_lora")
            )
            lora_weights_path = cool_path if os.path.exists(cool_path) else peaky_path

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading Img2Img base model '{base_model_name}' on {self.device}...")

        # Load Img2Img pipeline (handles taking an image input + prompt)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(self.device)

        # Load LoRA adapter if trained weights exist
        if os.path.exists(lora_weights_path):
            print(f"Loading trained Peaky Blinders LoRA weights from '{lora_weights_path}'...")
            self.pipe.load_lora_weights(lora_weights_path)
        else:
            print(
                f"Notice: LoRA weights at '{lora_weights_path}' not found. "
                "Pipeline will run with base model until LoRA is trained."
            )

    def convert(
        self,
        input_image: Image.Image | str,
        prompt: str = None,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        strength: float = 0.50,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
    ) -> Image.Image:
        """
        Transforms input photo into poster aesthetic.

        Args:
            input_image: PIL Image object or file path.
            prompt: Optional user prompt. If None or empty, default prompt is applied.
            negative_prompt: Elements to avoid in the output image.
            strength: Img2Img denoising strength (0.0 = identical to input, 1.0 = completely new image).
                      0.40-0.55 is optimal for preserving human face identity while applying style.
            guidance_scale: Classifier-free guidance scale.
            num_inference_steps: Number of denoising steps.


        Returns:
            PIL.Image: Styled output image.
        """
        # Auto-fill default prompt if none provided by user
        final_prompt = prompt.strip() if (prompt and prompt.strip()) else DEFAULT_PEAKY_PROMPT

        # Ensure image is PIL RGB
        if isinstance(input_image, str):
            if not os.path.exists(input_image):
                raise FileNotFoundError(f"Input image file '{input_image}' does not exist.")
            image = Image.open(input_image).convert("RGB")
        else:
            image = input_image.convert("RGB")

        # Resize for standard SD resolution while keeping ratio
        image = image.resize((512, 512))

        print(f"Processing image with prompt: '{final_prompt[:60]}...'")
        print(f"Img2Img Strength: {strength}")

        output = self.pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images[0]

        return output


if __name__ == "__main__":
    import sys

    print("Peaky Blinders Img2Img Generator ready!")
    print("Example usage in your app/API:")
    print("  generator = PeakyBlindersGenerator()")
    print("  styled_img = generator.convert('my_photo.jpg')  # Automatic (no prompt)")
    print("  styled_img = generator.convert('my_photo.jpg', prompt='peaky blinders flat cap suit')")
