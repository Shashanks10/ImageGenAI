"""
service.py

API service wrapper for PosterGenerator.
"""

import sys
import os
from io import BytesIO
from PIL import Image

# Ensure parent and Inference directory are on path for both IDE resolution and runtime
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INFERENCE_DIR = os.path.join(BASE_DIR, "Inference")

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
if INFERENCE_DIR not in sys.path:
    sys.path.append(INFERENCE_DIR)

try:
    from Inference.generate import PosterGenerator
except ImportError:
    try:
        from generate import PosterGenerator
    except ImportError:
        PosterGenerator = None

try:
    from Inference.generate_normal import NormalGenerator
except ImportError:
    try:
        from generate_normal import NormalGenerator
    except ImportError:
        NormalGenerator = None


class ImageGenService:
    def __init__(self):
        self.generator = None
        self.normal_generator = None

    def initialize(self):
        if self.generator is None and PosterGenerator is not None:
            self.generator = PosterGenerator()

    def initialize_normal(self):
        if self.normal_generator is None and NormalGenerator is not None:
            self.normal_generator = NormalGenerator()

    def process_image(self, image_bytes: bytes = None, prompt: str = None, strength: float = 0.85) -> bytes:
        if self.generator is None:
            self.initialize()

        if image_bytes:
            input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        else:
            input_image = None

        output_image = self.generator.generate(prompt=prompt, input_image=input_image, strength=strength)

        buffer = BytesIO()
        output_image.save(buffer, format="JPEG")
        return buffer.getvalue()

    def process_text_prompt(self, prompt: str, width: int = 512, height: int = 512) -> bytes:
        if self.generator is None:
            self.initialize()

        output_image = self.generator.generate(prompt=prompt, input_image=None, width=width, height=height)

        buffer = BytesIO()
        output_image.save(buffer, format="JPEG")
        return buffer.getvalue()

    def process_normal_text_prompt(self, prompt: str, width: int = 512, height: int = 512) -> bytes:
        if self.normal_generator is None:
            self.initialize_normal()

        if self.normal_generator is not None:
            output_image = self.normal_generator.generate(
                prompt=prompt,
                width=width,
                height=height,
            )
        else:
            if self.generator is None:
                self.initialize()
            output_image = self.generator.generate(
                prompt=prompt,
                input_image=None,
                width=width,
                height=height,
                use_lora=False,
                use_lora_trigger=False,
            )

        buffer = BytesIO()
        output_image.save(buffer, format="JPEG")
        return buffer.getvalue()


service_instance = ImageGenService()
