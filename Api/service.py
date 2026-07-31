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


class ImageGenService:
    def __init__(self):
        self.generator = None

    def initialize(self):
        if self.generator is None and PosterGenerator is not None:
            self.generator = PosterGenerator()

    def process_image(self, image_bytes: bytes, prompt: str = None, strength: float = 0.65) -> bytes:

        if self.generator is None:
            self.initialize()

        input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        output_image = self.generator.convert(input_image=input_image, prompt=prompt, strength=strength)

        buffer = BytesIO()
        output_image.save(buffer, format="JPEG")
        return buffer.getvalue()


service_instance = ImageGenService()
