#!/usr/bin/env python3
"""
ltx_server.py — FastAPI inference server for LTX-Video 2.3 on Vast.ai

Calls distilled.py directly via subprocess per job —
the same subprocess isolation pattern used by runpod_server.py.

Performs startup patches before any subprocess runs:
  1. encoder patch   — fixes encoder_configurator.py for transformers >= 4.47
                       where Gemma3's vision_tower is SiglipVisionModel directly
                       instead of SiglipModel (which had a .vision_model attribute)
  2. rope_config     — fixes rope_local_base_freq missing in transformers >= 4.51
  3. gemma config    — copies rope_scaling['type'] → 'rope_type' for transformers >= 4.46

NOTE: the ltx_pipelines.multigpu stub (blocks.py issue #216) is now applied
once at infra level by startup.sh — not here.

GEMMA_DIR must point to the root of a downloaded google/gemma-3-12b-it-qat-q4_0-unquantized
model (i.e. the directory that directly contains tokenizer.model,
preprocessor_config.json, and model-*.safetensors files).

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
LTX_ROOT     = "/workspace/LTX-2"
LTX_PKG_SRC  = f"{LTX_ROOT}/packages/ltx-pipelines/src"
LTX_CORE_SRC = f"{LTX_ROOT}/packages/ltx-core/src"
VENV_PYTHON  = f"{LTX_ROOT}/.venv/bin/python3"
LTX_CKPT     = "/workspace/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors"
LTX_UPSCALER = "/workspace/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
# GEMMA_DIR must be the directory that directly contains tokenizer.model,
# preprocessor_config.json, and model-*.safetensors — i.e. the root of a
# downloaded google/gemma-3-12b-it-qat-q4_0-unquantized HuggingFace repo.
# Example download: huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
#                     --local-dir /workspace/gemma
GEMMA_DIR    = "/workspace/gemma"
OUTPUTS      = Path("/workspace/outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)

# The venv created by `uv sync --frozen` carries all LTX deps;
# no PYTHONPATH manipulation needed — VENV_PYTHON resolves packages itself.
LTX_ENV = {
    **os.environ,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_LAUNCH_BLOCKING": "0",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ltx_server")


# ── encoder_configurator patch ────────────────────────────────────────────────
# Patch — SiglipVisionModel compat (transformers >= 4.47)
#   Old: v_model = model.model.vision_tower.vision_model
#   Pre-4.47: vision_tower = SiglipModel  →  .vision_model = SiglipVisionModel ✓
#   Post-4.47: vision_tower IS SiglipVisionModel directly  →  .vision_model fails
#   Fix: _tower = model.model.vision_tower
#        v_model = getattr(_tower, 'vision_model', _tower)
#   Both branches expose .embeddings.position_ids so the rest is unchanged.
def _patch_encoder_configurator() -> None:
    cfg_file = (
        Path(LTX_CORE_SRC)
        / "ltx_core" / "text_encoders" / "gemma" / "encoders" / "encoder_configurator.py"
    )
    if not cfg_file.exists():
        log.warning("encoder_configurator.py not found at %s — skipping", cfg_file)
        return

    content = cfg_file.read_text()
    old = "    v_model = model.model.vision_tower.vision_model"
    if old not in content:
        log.info("encoder_configurator.py: SiglipVisionModel patch already applied")
        return

    cfg_file.write_text(content.replace(old, (
        "    # Compat patch (ltx_server.py): transformers >= 4.47 exposes\n"
        "    # SiglipVisionModel as vision_tower directly; older versions\n"
        "    # wrapped it in SiglipModel(.vision_model).\n"
        "    _tower = model.model.vision_tower\n"
        "    v_model = getattr(_tower, 'vision_model', _tower)"
    )))
    log.info("encoder_configurator.py: SiglipVisionModel patch applied")


_patch_encoder_configurator()


# ── rope_config patch ─────────────────────────────────────────────────────────
# Patch — rope_local_base_freq missing (transformers >= 4.51+)
#   encoder_configurator.py line 166:
#       base = config.rope_local_base_freq
#   In newer transformers this attribute was moved into rope_parameters and is
#   no longer directly on Gemma3TextConfig.
#   Fix: base = getattr(config, 'rope_local_base_freq',
#                        getattr(config, 'rope_theta', 10000.0))
#   Falls back to rope_theta (the transformers-standard name) and finally to
#   10000.0 — the value defined in GEMMA3_CONFIG_FOR_LTX.text_config.
def _patch_rope_config() -> None:
    cfg_file = (
        Path(LTX_CORE_SRC)
        / "ltx_core" / "text_encoders" / "gemma" / "encoders" / "encoder_configurator.py"
    )
    if not cfg_file.exists():
        log.warning("encoder_configurator.py not found at %s — skipping rope patch", cfg_file)
        return

    content = cfg_file.read_text()
    old = "    base = config.rope_local_base_freq"
    if old not in content:
        log.info("encoder_configurator.py: rope_local_base_freq patch already applied")
        return

    cfg_file.write_text(content.replace(old, (
        "    # Compat patch (ltx_server.py): rope_local_base_freq removed from\n"
        "    # Gemma3TextConfig in newer transformers; fall back to rope_theta\n"
        "    # then to 10000.0 (GEMMA3_CONFIG_FOR_LTX default).\n"
        "    base = getattr(config, 'rope_local_base_freq', getattr(config, 'rope_theta', 10000.0))"
    )))
    log.info("encoder_configurator.py: rope_local_base_freq patch applied")


_patch_rope_config()


# ── gemma/config.json patch ───────────────────────────────────────────────────
# Patch — KeyError: 'rope_type' (transformers >= 4.46)
#   Newer transformers reads rope_scaling['rope_type'] directly.
#   Older Gemma config files (google/gemma-3-12b-it-qat-q4_0-unquantized) store
#   the same value under the key 'type' instead of 'rope_type'.
#   Fix: copy 'type' → 'rope_type' in every rope_scaling dict found in config.json
#   (top-level and nested under text_config).
def _patch_gemma_config() -> None:
    import json
    cfg_file = Path(GEMMA_DIR) / "config.json"
    if not cfg_file.exists():
        log.warning("gemma/config.json not found at %s — skipping rope_type patch", cfg_file)
        return

    data = json.loads(cfg_file.read_text())
    patched = False

    def _fix_rope(obj: dict, label: str) -> None:
        nonlocal patched
        rs = obj.get("rope_scaling")
        if not isinstance(rs, dict):
            return
        if "rope_type" in rs:
            log.info("gemma/config.json [%s]: rope_type already present", label)
            return
        if "type" not in rs:
            log.info("gemma/config.json [%s]: no 'type' key in rope_scaling — skipping", label)
            return
        rs["rope_type"] = rs["type"]
        patched = True
        log.info("gemma/config.json [%s]: added rope_type='%s'", label, rs["rope_type"])

    _fix_rope(data, "root")
    if "text_config" in data:
        _fix_rope(data["text_config"], "text_config")

    if patched:
        cfg_file.write_text(json.dumps(data, indent=2))
        log.info("gemma/config.json: rope_type patch written")


_patch_gemma_config()

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

        # Call distilled.py directly via subprocess isolation
        distilled_script = f"{LTX_PKG_SRC}/ltx_pipelines/distilled.py"
        cmd = [
            VENV_PYTHON, distilled_script,
            "--distilled-checkpoint-path", LTX_CKPT,
            "--spatial-upsampler-path", LTX_UPSCALER,
            "--gemma-root",            GEMMA_DIR,
            "--prompt",                req.prompt,
            "--output-path",           out_path,
            "--num-frames",            str(req.num_frames),
            "--width",                 str(req.width),
            "--height",                str(req.height),
            "--frame-rate",            "24",
            "--quantization",          "fp8-cast",
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
