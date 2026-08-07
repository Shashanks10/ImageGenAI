"""
router.py

FastAPI router exposing the image-to-image poster generator endpoint.
"""

from fastapi import APIRouter, UploadFile, File, Form, Response, HTTPException
from service import service_instance

router = APIRouter()


@router.post("/generate-poster")
async def generate_poster_image(
    file: UploadFile = File(None),
    prompt: str = Form(None),
    strength: float = Form(0.85),
):
    """
    Generates a cool poster image.
    - Upload an image to run Image-to-Image transformation.
    - Omit the image file to run Text-to-Image generation directly from text prompt!
    """
    image_bytes = None
    if file is not None and file.filename:
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        image_bytes = await file.read()

    output_bytes = service_instance.process_image(
        image_bytes=image_bytes,
        prompt=prompt,
        strength=strength,
    )

    return Response(content=output_bytes, media_type="image/jpeg")


@router.post("/generate-text")
async def generate_text_image(
    prompt: str = Form("a cat and a dog fighting in an epic art style"),
    width: int = Form(512),
    height: int = Form(512),
):
    """
    Text-to-Image generation endpoint.
    Generate brand new images from any text prompt (no input image required)!
    Example: 'a cat and a dog fighting'
    """
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt parameter is required.")

    output_bytes = service_instance.process_text_prompt(
        prompt=prompt.strip(),
        width=width,
        height=height,
    )

    return Response(content=output_bytes, media_type="image/jpeg")

