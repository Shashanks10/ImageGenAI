"""
router.py

FastAPI router exposing the image-to-image Peaky Blinders generator endpoint.
"""

from fastapi import APIRouter, UploadFile, File, Form, Response, HTTPException
from service import service_instance

router = APIRouter()


@router.post("/generate-peaky")
async def generate_peaky_blinders_image(
    file: UploadFile = File(...),
    prompt: str = Form(None),
    strength: float = Form(0.60),
):
    """
    Takes an input image and transforms it into Peaky Blinders style.
    Optional text prompt can be supplied.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    output_bytes = service_instance.process_image(
        image_bytes=image_bytes,
        prompt=prompt,
        strength=strength,
    )

    return Response(content=output_bytes, media_type="image/jpeg")
