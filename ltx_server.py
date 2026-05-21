#!/usr/bin/env python3
"""
ltx_server.py — FastAPI inference server for LTX-Video 2.3 on Vast.ai

Endpoints:
    POST /ltx/generate         → {"job_id": "..."} (immediate)
    GET  /ltx/status/{job_id}  → {"status": "...", "video_url": "...", "error": "..."}
    GET  /outputs/{job_id}.mp4 → video file
    GET  /health               → diagnostics
"""

import gc
import os
import sys
import uuid
import asyncio
import logging
from io import BytesIO
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

# Must be set before any CUDA / torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
LTX_CODE = "/workspace/LTX-2"      # git checkout of the LTX-Video 2 repo
LTX_CKPT = "/workspace/ltx-2.3"   # model weights directory
GEMMA_DIR = "/workspace/gemma"     # Gemma text encoder weights
OUTPUTS   = Path("/workspace/outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, LTX_CODE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ltx_server")

# ── Global state ──────────────────────────────────────────────────────────────
JOBS: dict[str, dict] = {}
_pipe      = None
_pipe_lock = asyncio.Lock()   # serialise GPU access — one job at a time


# ── Pipeline loader (blocking, called once at startup) ────────────────────────
def _load_pipeline():
    log.info("Importing LTX-Video package from %s …", LTX_CODE)
    from ltx_video.pipelines.pipeline_ltx_video import DistilledPipeline

    log.info("Loading DistilledPipeline weights from %s …", LTX_CKPT)
    pipe = DistilledPipeline.from_pretrained(
        LTX_CKPT,
        text_encoder_path=GEMMA_DIR,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to("cuda")

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    log.info(
        "LTX-Video 2.3 DistilledPipeline ready — device: %s",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    )
    return pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipe
    _pipe = await asyncio.to_thread(_load_pipeline)
    yield
    del _pipe
    gc.collect()
    torch.cuda.empty_cache()


app = FastAPI(title="LTX-Video 2.3", lifespan=lifespan)


# ── Request schema ────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    image_url: Optional[str] = None
    num_frames: int = 97
    width: int = 704
    height: int = 1280


# ── Synchronous generation (executed inside asyncio.to_thread) ───────────────
def _generate_sync(
    job_id: str,
    req: GenerateRequest,
    image: Optional[Image.Image],
) -> None:
    import numpy as np
    import imageio

    kwargs: dict = dict(
        prompt=req.prompt,
        num_frames=req.num_frames,
        width=req.width,
        height=req.height,
        num_inference_steps=8,   # DistilledPipeline fast path
        guidance_scale=1.0,
        output_type="pil",
    )
    if image is not None:
        kwargs["image"] = image

    log.info(
        "[%s] Inference start — %dx%d %d frames prompt=%r",
        job_id, req.width, req.height, req.num_frames, req.prompt[:80],
    )
    result = _pipe(**kwargs)

    # Normalise output — pipelines return result.frames or result.videos
    if hasattr(result, "frames"):
        frames = result.frames[0]
    elif hasattr(result, "videos"):
        frames = result.videos[0]
    else:
        frames = result[0]

    out_path = str(OUTPUTS / f"{job_id}.mp4")
    log.info("[%s] Writing %d frames → %s", job_id, len(frames), out_path)
    imageio.mimwrite(out_path, [np.array(f) for f in frames], fps=24, quality=8)
    log.info("[%s] Saved %d KB", job_id, Path(out_path).stat().st_size // 1024)


# ── Async job runner ──────────────────────────────────────────────────────────
async def _run_job(job_id: str, req: GenerateRequest) -> None:
    JOBS[job_id] = {"status": "processing"}
    try:
        # Download conditioning image if requested
        image: Optional[Image.Image] = None
        if req.image_url:
            log.info("[%s] Downloading image: %s", job_id, req.image_url)
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(req.image_url)
                resp.raise_for_status()
            image = Image.open(BytesIO(resp.content)).convert("RGB")
            log.info("[%s] Image loaded: %s", job_id, image.size)

        async with _pipe_lock:
            await asyncio.to_thread(_generate_sync, job_id, req, image)

        JOBS[job_id] = {
            "status": "completed",
            "video_url": f"/outputs/{job_id}.mp4",
        }
        log.info("[%s] Job completed", job_id)

    except Exception as exc:
        log.exception("[%s] Job failed", job_id)
        JOBS[job_id] = {"status": "failed", "error": str(exc)}

    finally:
        gc.collect()
        torch.cuda.empty_cache()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/ltx/generate")
async def generate(req: GenerateRequest):
    """Submit a generation job; returns job_id immediately."""
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "pending"}
    asyncio.create_task(_run_job(job_id, req))
    return {"job_id": job_id}


@app.get("/ltx/status/{job_id}")
async def get_status(job_id: str):
    """Poll generation status."""
    job = JOBS.get(job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content={"status": "not_found", "error": "Unknown job_id"},
        )
    return job


@app.get("/outputs/{job_id}.mp4")
async def get_video(job_id: str):
    """Stream the finished MP4."""
    path = OUTPUTS / f"{job_id}.mp4"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "Video not found"})
    return FileResponse(str(path), media_type="video/mp4", filename=f"{job_id}.mp4")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "pipeline_loaded": _pipe is not None,
        "jobs_total": len(JOBS),
        "jobs_pending": sum(1 for j in JOBS.values() if j.get("status") == "pending"),
        "jobs_processing": sum(1 for j in JOBS.values() if j.get("status") == "processing"),
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("ltx_server:app", host="0.0.0.0", port=8000, workers=1)
