# Load SD + attach LoRA
import torch

from diffusers import StableDiffusionPipeline
from peft import LoraConfig
from peft import get_peft_model


MODEL_NAME = "stabilityai/stable-diffusion-2-1"

    
def load_model():

    # Load Stable Diffusion Pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16
    )

    pipe = pipe.to("cuda")

    # Disable safety checker (optional for training)
    pipe.safety_checker = None

    # Freeze entire pipeline
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # LoRA Configuration
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

    # Attach LoRA to U-Net
    pipe.unet = get_peft_model(
        pipe.unet,
        lora_config
    )

    # Print trainable parameters
    pipe.unet.print_trainable_parameters()

    return pipe