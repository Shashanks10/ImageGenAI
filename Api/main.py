"""
main.py

FastAPI entrypoint for FutureGenImage API.
"""

from fastapi import FastAPI
from router import router

app = FastAPI(
    title="FutureGenImage - Peaky Blinders Generator API",
    description="Img2Img AI transformation endpoint converting photos into Peaky Blinders aesthetic.",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "FutureGenImage API is running. POST to /generate-peaky to convert photos."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
