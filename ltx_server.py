#!/usr/bin/env python3
"""
ltx_server.py — FastAPI inference server for LTX-Video 2.3 on Vast.ai

Calls `python3 -m ltx_pipelines.distilled` via subprocess per job —
the same subprocess isolation pattern used by runpod_server.py.
This avoids in-process import issues (e.g. ltx_pipelines.multigpu).

Endpoints:
    POST /ltx/generate         → {"job_id": "..."} (immediate)
    GET  /ltx/status/{job_id}  → {"status": "...", "video_url": "...", "error": "..."}
    GET  /outputs/{job_id}.mp4 → stream finished video
    GET  /health               → diagnostics
"""

import gc
import os
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────
LTX_PKG_SRC  = "/workspace/LTX-2/LTX-2/LTX-2/packages/ltx-pipelines/src"
LTX_CORE_SRC = "/workspace/LTX-2/LTX-2/LTX-2/packages/ltx-core/src"
LTX_CKPT     = "/workspace/ltx-2.3/ltx-2.3-22b-distilled.safetensors"
LTX_UPSCALER = "/workspace/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_DIR    = "/workspace/gemma"
OUTPUTS      = Path("/workspace/outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Env forwarded to every subprocess — adds the ltx_pipelines src to PYTHONPATH
LTX_ENV = {
    **os.environ,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_LAUNCH_BLOCKING": "0",
    "PYTHONPATH": os.pathsep.join(filter(None, [
        LTX_PKG_SRC,
        LTX_CORE_SRC,
        os.environ.get("PYTHONPATH", ""),
    ])),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ltx_server")

# ── State ─────────────────────────────────────────────────────────────────────
JOBS: dict[str, dict] = {}
_job_sem = asyncio.Semaphore(1)   # one GPU job at a time


# ── Request schema ────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    image_url: Optional[str] = None
    num_frames: int = 97
    width: int = 704
    height: int = 1280


# ── Job runner ────────────────────────────────────────────────────────────────
async def _run_job(job_id: str, req: GenerateRequest) -> None:
    JOBS[job_id] = {"status": "processing"}
    out_path = str(OUTPUTS / f"{job_id}.mp4")
    img_path = None

    try:
        # Download conditioning image when image_url is provided (image-to-video)
        if req.image_url:
            log.info("[%s] Downloading conditioning image: %s", job_id, req.image_url)
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(req.image_url)
                resp.raise_for_status()
            img_path = str(OUTPUTS / f"{job_id}_cond.jpg")
            with open(img_path, "wb") as f:
                f.write(resp.content)
            log.info("[%s] Conditioning image saved (%d bytes)", job_id, len(resp.content))

        # Build CLI command for python -m ltx_pipelines.distilled
        cmd = [
            "python3", "-m", "ltx_pipelines.distilled",
            "--distilled-checkpoint-path", LTX_CKPT,
            "--spatial-upsampler-path",    LTX_UPSCALER,
            "--gemma-root",                GEMMA_DIR,
            "--prompt",                    req.prompt,
            "--output-path",               out_path,
            "--num-frames",                str(req.num_frames),
            "--width",                     str(req.width),
            "--height",                    str(req.height),
            "--frame-rate",                "24",
            "--quantization",              "fp8-cast",   # cast bf16→fp8 on the fly
        ]
        # Image-to-video: condition on first frame (frame index 0, strength 1.0)
        if img_path:
            cmd += ["--image", img_path, "0", "1.0"]

        log.info("[%s] CMD: %s", job_id, " ".join(cmd))

        async with _job_sem:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=LTX_ENV,
            )
            stdout, stderr = await proc.communicate()

        out_text = stdout.decode()
        err_text = stderr.decode()
        if out_text.strip():
            log.info("[%s] STDOUT: %s", job_id, out_text[-1000:])
        if err_text.strip():
            log.info("[%s] STDERR: %s", job_id, err_text[-2000:])

        if proc.returncode != 0:
            JOBS[job_id] = {
                "status": "failed",
                "error": err_text[-2000:] or f"subprocess exit code {proc.returncode}",
            }
            return

        if not os.path.exists(out_path):
            JOBS[job_id] = {"status": "failed", "error": "Video file not created"}
            return

        kb = os.path.getsize(out_path) // 1024
        log.info("[%s] Completed — %d KB → %s", job_id, kb, out_path)
        JOBS[job_id] = {
            "status": "completed",
            "video_url": f"/outputs/{job_id}.mp4",
        }

    except Exception as exc:
        log.exception("[%s] Job failed", job_id)
        JOBS[job_id] = {"status": "failed", "error": str(exc)}

    finally:
        if img_path and os.path.exists(img_path):
            os.remove(img_path)
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


# ── App & routes ──────────────────────────────────────────────────────────────
app = FastAPI(title="LTX-Video 2.3")


@app.post("/ltx/generate")
async def generate(req: GenerateRequest):
    """Queue a generation job; returns job_id immediately."""
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "pending"}
    asyncio.create_task(_run_job(job_id, req))
    return {"job_id": job_id}


@app.get("/ltx/status/{job_id}")
async def get_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return JSONResponse(
            status_code=404,
            content={"status": "not_found", "error": "Unknown job_id"},
        )
    return job


@app.get("/outputs/{job_id}.mp4")
async def get_video(job_id: str):
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
        "checkpoint": LTX_CKPT,
        "upscaler": LTX_UPSCALER,
        "jobs_total": len(JOBS),
        "jobs_pending": sum(1 for j in JOBS.values() if j.get("status") == "pending"),
        "jobs_processing": sum(1 for j in JOBS.values() if j.get("status") == "processing"),
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("ltx_server:app", host="0.0.0.0", port=8001, workers=1)
