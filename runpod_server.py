import os
import io
import tempfile
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import Response
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Chemins des modèles sur RunPod
# ---------------------------------------------------------------------------

VOXTRAL_PATH = os.getenv("VOXTRAL_PATH", "/workspace/voxtral")
QWEN_PATH    = os.getenv("QWEN_PATH",    "/workspace/qwen-image")
WAN_T2V_PATH = os.getenv("WAN_T2V_PATH", "/workspace/wan2.2-t2v")
WAN_I2V_PATH = os.getenv("WAN_I2V_PATH", "/workspace/wan2.2-animate")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

models: dict = {}


# ---------------------------------------------------------------------------
# Chargement des modèles
# ---------------------------------------------------------------------------

def load_voxtral():
    if "voxtral" in models:
        return
    print(f"[voxtral] chargement depuis {VOXTRAL_PATH} ...")
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    processor = AutoProcessor.from_pretrained(VOXTRAL_PATH, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        VOXTRAL_PATH,
        torch_dtype=DTYPE,
        device_map="auto",
        local_files_only=True,
    )
    models["voxtral"] = {"model": model, "processor": processor}
    print("[voxtral] prêt")


def load_qwen():
    if "qwen" in models:
        return
    print(f"[qwen] chargement depuis {QWEN_PATH} ...")
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(
        QWEN_PATH, trust_remote_code=True, local_files_only=True
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_PATH,
        torch_dtype=DTYPE,
        device_map="auto",
        local_files_only=True,
    )
    models["qwen"] = {"model": model, "processor": processor}
    print("[qwen] prêt")


def load_wan_t2v():
    if "wan_t2v" in models:
        return
    print(f"[wan-t2v] chargement depuis {WAN_T2V_PATH} ...")
    try:
        from wan.pipelines import WanT2VPipeline
        pipe = WanT2VPipeline.from_pretrained(WAN_T2V_PATH, torch_dtype=DTYPE)
    except (ImportError, Exception):
        from diffusers import WanPipeline
        pipe = WanPipeline.from_pretrained(
            WAN_T2V_PATH, torch_dtype=DTYPE, local_files_only=True
        )
    pipe.to(DEVICE)
    models["wan_t2v"] = pipe
    print("[wan-t2v] prêt")


def load_wan_i2v():
    if "wan_i2v" in models:
        return
    print(f"[wan-i2v] chargement depuis {WAN_I2V_PATH} ...")
    try:
        from wan.pipelines import WanI2VPipeline
        pipe = WanI2VPipeline.from_pretrained(WAN_I2V_PATH, torch_dtype=DTYPE)
    except (ImportError, Exception):
        from diffusers import WanImageToVideoPipeline
        pipe = WanImageToVideoPipeline.from_pretrained(
            WAN_I2V_PATH, torch_dtype=DTYPE, local_files_only=True
        )
    pipe.to(DEVICE)
    models["wan_i2v"] = pipe
    print("[wan-i2v] prêt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    # Voxtral et Qwen en parallèle au démarrage
    await asyncio.gather(
        loop.run_in_executor(None, load_voxtral),
        loop.run_in_executor(None, load_qwen),
        loop.run_in_executor(None, load_wan_t2v),
        return_exceptions=True,
    )
    # Wan I2V chargé à la demande (VRAM limitée)
    yield
    models.clear()


app = FastAPI(title="ViralVidTech RunPod API", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schémas
# ---------------------------------------------------------------------------

class WanT2VRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted"
    num_frames: int = 81
    width: int = 832
    height: int = 480
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    fps: int = 16


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _save_video(frames, fps: int) -> bytes:
    import imageio
    import numpy as np
    from PIL import Image as PILImage

    np_frames = []
    for f in frames:
        if hasattr(f, "cpu"):
            arr = (f.cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
        elif isinstance(f, PILImage.Image):
            arr = np.array(f)
        else:
            arr = np.array(f)
        np_frames.append(arr)

    buf = io.BytesIO()
    with imageio.get_writer(buf, fps=fps, format="ffmpeg", codec="libx264", output_params=["-f", "mp4"]) as w:
        for frame in np_frames:
            w.append_data(frame)
    return buf.getvalue()


def _read_audio(audio_bytes: bytes, filename: str):
    import soundfile as sf
    suffix = os.path.splitext(filename or ".wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        audio_array, sample_rate = sf.read(tmp_path)
    finally:
        os.unlink(tmp_path)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    return audio_array, sample_rate


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": list(models.keys()),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if torch.cuda.is_available() else None,
    }


# ---------------------------------------------------------------------------
# POST /voxtral/transcribe
# ---------------------------------------------------------------------------

@app.post("/voxtral/transcribe")
async def voxtral_transcribe(file: UploadFile = File(...)):
    """Transcrit un fichier audio en texte avec Voxtral-Mini-4B-Realtime."""
    load_voxtral()
    audio_bytes = await file.read()

    loop = asyncio.get_event_loop()

    def _run():
        audio_array, sample_rate = _read_audio(audio_bytes, file.filename or "audio.wav")
        proc  = models["voxtral"]["processor"]
        model = models["voxtral"]["model"]
        inputs = proc(
            audio=audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            predicted_ids = model.generate(**inputs, max_new_tokens=448)
        return proc.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    try:
        transcription = await loop.run_in_executor(None, _run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"transcription": transcription}


# ---------------------------------------------------------------------------
# POST /qwen/analyze
# ---------------------------------------------------------------------------

@app.post("/qwen/analyze")
async def qwen_analyze(
    file: UploadFile = File(...),
    prompt: str = Query(default="Décris cette image en détail."),
):
    """Analyse une image et retourne une description détaillée avec Qwen-Image."""
    load_qwen()
    from PIL import Image

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    loop = asyncio.get_event_loop()

    def _run():
        proc  = models["qwen"]["processor"]
        model = models["qwen"]["model"]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }]

        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = proc(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(DEVICE)
        except ImportError:
            inputs = proc(text=[text], padding=True, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return proc.batch_decode(trimmed, skip_special_tokens=True)[0]

    try:
        response = await loop.run_in_executor(None, _run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"response": response}


# ---------------------------------------------------------------------------
# POST /wan/generate  (texte → vidéo)
# ---------------------------------------------------------------------------

@app.post("/wan/generate")
async def wan_generate(request: WanT2VRequest):
    """Génère une vidéo MP4 à partir d'un prompt texte avec Wan 2.2 T2V."""
    load_wan_t2v()
    pipe = models["wan_t2v"]

    loop = asyncio.get_event_loop()

    def _run():
        with torch.no_grad():
            result = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_frames=request.num_frames,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
            )
        frames = result.frames[0] if hasattr(result, "frames") else result.videos[0]
        return _save_video(frames, fps=request.fps)

    try:
        video_bytes = await loop.run_in_executor(None, _run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={"Content-Disposition": "attachment; filename=output.mp4"},
    )


# ---------------------------------------------------------------------------
# POST /wan/image2video  (image → vidéo animée)
# ---------------------------------------------------------------------------

@app.post("/wan/image2video")
async def wan_image2video(
    file: UploadFile = File(...),
    prompt: str = Query(default="Anime cette image de façon réaliste"),
    negative_prompt: str = Query(default="blurry, low quality, distorted, static"),
    num_frames: int = Query(default=81),
    num_inference_steps: int = Query(default=50),
    guidance_scale: float = Query(default=7.5),
    fps: int = Query(default=16),
):
    """Anime une image statique en vidéo MP4 avec Wan 2.2 Animate."""
    load_wan_i2v()
    from PIL import Image

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pipe = models["wan_i2v"]

    loop = asyncio.get_event_loop()

    def _run():
        with torch.no_grad():
            result = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
        frames = result.frames[0] if hasattr(result, "frames") else result.videos[0]
        return _save_video(frames, fps=fps)

    try:
        video_bytes = await loop.run_in_executor(None, _run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={"Content-Disposition": "attachment; filename=animated.mp4"},
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
