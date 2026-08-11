"""
main.py

FastAPI entrypoint for FutureGenImage API.
"""

from fastapi import FastAPI
from router import router

app = FastAPI(
    title="FutureGenImage - Cool Poster Generator API",
    description="Img2Img AI transformation endpoint converting photos into Cool Poster aesthetic.",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "FutureGenImage API is running.",
        "endpoints": {
            "POST /generate-poster": "Convert photos to poster style or generate poster from prompt",
            "POST /generate-text": "Generate poster art from text prompt",
            "POST /generate-normal": "Generate standard image from prompt without trained LoRA data",
        },
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
