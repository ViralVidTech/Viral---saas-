import os
import io
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import Optional

RUNPOD_BASE_URL = os.getenv("RUNPOD_API_URL", "https://849ams2zdun0ya-8000.proxy.runpod.net")
RUNPOD_TIMEOUT = float(os.getenv("RUNPOD_TIMEOUT", "300"))

app = FastAPI(title="ViralVidTech API", version="2.0.0")


def _runpod_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=RUNPOD_BASE_URL, timeout=RUNPOD_TIMEOUT)


class WanT2VRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality"
    num_frames: int = 81
    width: int = 832
    height: int = 480
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    fps: int = 16


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    async with _runpod_client() as client:
        try:
            r = await client.get("/health")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"RunPod unreachable: {exc}")


# ---------------------------------------------------------------------------
# Voxtral — transcription audio → texte
# ---------------------------------------------------------------------------

@app.post("/voxtral/transcribe")
async def voxtral_transcribe(file: UploadFile = File(...)):
    """Transcrit un fichier audio en texte via Voxtral sur RunPod."""
    audio_bytes = await file.read()
    async with _runpod_client() as client:
        try:
            r = await client.post(
                "/voxtral/transcribe",
                files={"file": (file.filename, audio_bytes, file.content_type or "audio/wav")},
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc))


# ---------------------------------------------------------------------------
# Qwen — analyse d'image
# ---------------------------------------------------------------------------

@app.post("/qwen/analyze")
async def qwen_analyze(
    file: UploadFile = File(...),
    prompt: str = Query(default="Describe this image."),
):
    """Analyse une image et génère une description via Qwen sur RunPod."""
    image_bytes = await file.read()
    async with _runpod_client() as client:
        try:
            r = await client.post(
                "/qwen/analyze",
                params={"prompt": prompt},
                files={"file": (file.filename, image_bytes, file.content_type or "image/jpeg")},
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc))


# ---------------------------------------------------------------------------
# Wan — texte → vidéo (T2V)
# ---------------------------------------------------------------------------

@app.post("/wan/generate")
async def wan_generate(request: WanT2VRequest):
    """Génère une vidéo MP4 à partir d'un prompt texte via Wan 2.2 T2V sur RunPod."""
    async with _runpod_client() as client:
        try:
            r = await client.post("/wan/generate", json=request.model_dump())
            r.raise_for_status()
            return Response(
                content=r.content,
                media_type="video/mp4",
                headers={"Content-Disposition": "attachment; filename=output.mp4"},
            )
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc))


# ---------------------------------------------------------------------------
# Wan — image → vidéo (I2V / Animate)
# ---------------------------------------------------------------------------

@app.post("/wan/image2video")
async def wan_image2video(
    file: UploadFile = File(...),
    prompt: str = Query(default="Animate this image"),
    negative_prompt: str = Query(default="blurry, low quality"),
    num_frames: int = Query(default=81),
    num_inference_steps: int = Query(default=50),
    guidance_scale: float = Query(default=7.5),
    fps: int = Query(default=16),
):
    """Anime une image statique en vidéo MP4 via Wan 2.2 Animate sur RunPod."""
    image_bytes = await file.read()
    async with _runpod_client() as client:
        try:
            r = await client.post(
                "/wan/image2video",
                params={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_frames": num_frames,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "fps": fps,
                },
                files={"file": (file.filename, image_bytes, file.content_type or "image/jpeg")},
            )
            r.raise_for_status()
            return Response(
                content=r.content,
                media_type="video/mp4",
                headers={"Content-Disposition": "attachment; filename=animated.mp4"},
            )
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
