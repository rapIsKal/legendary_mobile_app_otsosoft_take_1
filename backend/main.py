"""
main.py — ClockScan FastAPI backend
Ultra-lightweight: OpenCV only, no AI model, runs on 1 vCPU / 512MB RAM
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from clock_reader import read_clock

app = FastAPI(title="ClockScan API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "service": "ClockScan", "engine": "OpenCV"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/read-clock")
async def read_clock_endpoint(image: UploadFile = File(...)):
    # Validate file type
    if image.content_type not in ("image/jpeg", "image/png", "image/webp", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG/WebP images accepted")

    # Limit file size to 10MB
    image_bytes = await image.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Max 10MB.")

    try:
        result = read_clock(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
