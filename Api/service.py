"""
service.py

API service wrapper for PeakyBlindersGenerator.
"""

import sys
import os
from io import BytesIO
from PIL import Image

# Ensure Inference directory is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Inference")))

try:
    from generate import PeakyBlindersGenerator
except ImportError:
    PeakyBlindersGenerator = None


class ImageGenService:
    def __init__(self):
        self.generator = None

    def initialize(self):
        if self.generator is None and PeakyBlindersGenerator is not None:
            self.generator = PeakyBlindersGenerator()

    def process_image(self, image_bytes: bytes, prompt: str = None, strength: float = 0.6) -> bytes:
        if self.generator is None:
            self.initialize()
            
        input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        output_image = self.generator.convert(input_image=input_image, prompt=prompt, strength=strength)
        
        buffer = BytesIO()
        output_image.save(buffer, format="JPEG")
        return buffer.getvalue()


service_instance = ImageGenService()
