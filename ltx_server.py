#!/usr/bin/env python3
"""
ltx_server.py — FastAPI inference server for LTX-Video 2.3 on Vast.ai

Calls distilled.py directly via subprocess per job —
the same subprocess isolation pattern used by runpod_server.py.

Performs two startup patches before any subprocess runs:
  1. multigpu stub   — creates ltx_pipelines/multigpu/delegating_builder.py
  2. encoder patch   — fixes encoder_configurator.py for transformers >= 4.47
                       where Gemma3's vision_tower is SiglipVisionModel directly
                       instead of SiglipModel (which had a .vision_model attribute)

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
LTX_PKG_SRC  = "/workspace/LTX-2/LTX-2/LTX-2/packages/ltx-pipelines/src"
LTX_CORE_SRC = "/workspace/LTX-2/LTX-2/LTX-2/packages/ltx-core/src"
LTX_CKPT     = "/workspace/ltx-2.3/ltx-2.3-22b-distilled.safetensors"
LTX_UPSCALER = "/workspace/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
# GEMMA_DIR must be the directory that directly contains tokenizer.model,
# preprocessor_config.json, and model-*.safetensors — i.e. the root of a
# downloaded google/gemma-3-12b-it-qat-q4_0-unquantized HuggingFace repo.
# Example download: huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
#                     --local-dir /workspace/gemma
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


# ── multigpu stub ─────────────────────────────────────────────────────────────
# ltx_pipelines/utils/blocks.py line 69 does:
#   from ltx_pipelines.multigpu.delegating_builder import DelegatingBuilder
# That submodule doesn't ship with the package.  We need TWO files:
#   multigpu/__init__.py          — makes it a package
#   multigpu/delegating_builder.py — where DelegatingBuilder actually lives
#
# DelegatingBuilder[LTXModel] is used only as a type annotation in
# DiffusionStage.__init__; it is NEVER instantiated in single-GPU inference.
# The class therefore only needs to be importable and support Generic[T] syntax.
def _ensure_multigpu_stub() -> None:
    stub_dir = Path(LTX_PKG_SRC) / "ltx_pipelines" / "multigpu"
    stub_dir.mkdir(parents=True, exist_ok=True)

    # ── multigpu/__init__.py ──────────────────────────────────────────────────
    init_file = stub_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text(
            "# Auto-generated stub — single-GPU mode\n"
            "from ltx_pipelines.multigpu.delegating_builder import DelegatingBuilder\n"
            "__all__ = ['DelegatingBuilder']\n"
        )
        log.info("Created multigpu __init__ stub at %s", init_file)

    # ── multigpu/delegating_builder.py — the actual import target ────────────
    db_file = stub_dir / "delegating_builder.py"
    if not db_file.exists():
        db_file.write_text(
            "# Auto-generated stub — single-GPU mode.\n"
            "#\n"
            "# ltx_pipelines/utils/blocks.py imports:\n"
            "#   from ltx_pipelines.multigpu.delegating_builder import DelegatingBuilder\n"
            "#\n"
            "# DelegatingBuilder[T] appears only as a type annotation in\n"
            "# DiffusionStage.__init__; it is never instantiated during single-GPU\n"
            "# inference (transformer_builder defaults to None, so the real Builder\n"
            "# is constructed instead).  This stub makes the import succeed and\n"
            "# supports DelegatingBuilder[LTXModel] generic syntax at runtime.\n"
            "from __future__ import annotations\n"
            "\n"
            "from typing import Generic, TypeVar\n"
            "\n"
            "_T = TypeVar('_T')\n"
            "\n"
            "\n"
            "class DelegatingBuilder(Generic[_T]):\n"
            "    \"\"\"Stub: real multi-GPU DelegatingBuilder not available in single-GPU mode.\"\"\"\n"
            "\n"
            "    def __getattr__(self, name: str) -> object:\n"
            "        raise RuntimeError(\n"
            "            f'DelegatingBuilder.{name} was called — this is a single-GPU stub; '\n"
            "            'multi-GPU inference is not configured.'\n"
            "        )\n"
        )
        log.info("Created multigpu delegating_builder stub at %s", db_file)


_ensure_multigpu_stub()


# ── encoder_configurator patches ─────────────────────────────────────────────
# LTX-2's encoder_configurator.py was written against an older transformers.
# Each patch is independent and idempotent — applied only if the old line is
# still present; safe to re-run on every server start.
#
# Patch 1 — SiglipVisionModel compat (transformers >= 4.47)
#   Old: v_model = model.model.vision_tower.vision_model
#   Pre-4.47: vision_tower = SiglipModel  →  .vision_model = SiglipVisionModel ✓
#   Post-4.47: vision_tower IS SiglipVisionModel directly  →  .vision_model fails
#   Fix: _tower = model.model.vision_tower
#        v_model = getattr(_tower, 'vision_model', _tower)
#   Both branches expose .embeddings.position_ids so the rest is unchanged.
#
# Patch 2 — rope_local_base_freq removed (transformers >= 4.51+)
#   Old: base = config.rope_local_base_freq
#   New transformers moved this into rope_parameters dict; the attribute no
#   longer exists on Gemma3TextConfig directly.
#   Fix: base = getattr(config, 'rope_local_base_freq', 10000)
#   10000 is the value from GEMMA3_CONFIG_FOR_LTX (Gemma3TextConfig default).
def _patch_encoder_configurator() -> None:
    cfg_file = (
        Path(LTX_CORE_SRC)
        / "ltx_core" / "text_encoders" / "gemma" / "encoders" / "encoder_configurator.py"
    )
    if not cfg_file.exists():
        log.warning("encoder_configurator.py not found at %s — skipping patches", cfg_file)
        return

    content = cfg_file.read_text()
    patched = False

    # ── Patch 1: vision_tower.vision_model ────────────────────────────────────
    old1 = "    v_model = model.model.vision_tower.vision_model"
    if old1 in content:
        content = content.replace(old1, (
            "    # Compat patch 1 (ltx_server.py): transformers >= 4.47 exposes\n"
            "    # SiglipVisionModel as vision_tower directly; older versions\n"
            "    # wrapped it in SiglipModel(.vision_model).\n"
            "    _tower = model.model.vision_tower\n"
            "    v_model = getattr(_tower, 'vision_model', _tower)"
        ))
        patched = True
        log.info("encoder_configurator patch 1 applied (SiglipVisionModel compat)")

    # ── Patch 2: rope_local_base_freq ─────────────────────────────────────────
    old2 = "    base = config.rope_local_base_freq"
    if old2 in content:
        content = content.replace(old2, (
            "    # Compat patch 2 (ltx_server.py): rope_local_base_freq removed from\n"
            "    # Gemma3TextConfig in newer transformers (moved into rope_parameters).\n"
            "    # 10000 is the GEMMA3_CONFIG_FOR_LTX default.\n"
            "    base = getattr(config, 'rope_local_base_freq', 10000)"
        ))
        patched = True
        log.info("encoder_configurator patch 2 applied (rope_local_base_freq compat)")

    if patched:
        cfg_file.write_text(content)
    else:
        log.info("encoder_configurator.py: all patches already applied")


_patch_encoder_configurator()

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

        # Call distilled.py directly (avoids __init__.py importing multigpu)
        distilled_script = (
            "/workspace/LTX-2/LTX-2/LTX-2/packages/ltx-pipelines/src"
            "/ltx_pipelines/distilled.py"
        )
        cmd = [
            "python3", distilled_script,
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
