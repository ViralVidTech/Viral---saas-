import os
import io
import json
import tempfile
import asyncio
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Chemins des modèles sur RunPod
# ---------------------------------------------------------------------------

VOXTRAL_PATH = os.getenv("VOXTRAL_PATH", "/workspace/voxtral")
QWEN_PATH    = os.getenv("QWEN_PATH",    "/workspace/qwen-image")
WAN_T2V_PATH = os.getenv("WAN_T2V_PATH", "/workspace/wan2.2-t2v")
WAN_I2V_PATH = os.getenv("WAN_I2V_PATH", "/workspace/wan2.2-animate")

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID  = 0 if torch.cuda.is_available() else -1
DTYPE      = torch.bfloat16 if torch.cuda.is_available() else torch.float32

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


def _load_wan_config(path: str):
    """Lit configuration.json et retourne un EasyDict."""
    from easydict import EasyDict
    config_path = os.path.join(path, "configuration.json")
    with open(config_path, "r") as f:
        return EasyDict(json.load(f))


def load_wan_t2v():
    if "wan_t2v" in models:
        return
    print(f"[wan-t2v] chargement depuis {WAN_T2V_PATH} ...")
    from wan import WanT2V
    config = _load_wan_config(WAN_T2V_PATH)
    pipe = WanT2V(
        config=config,
        checkpoint_dir=WAN_T2V_PATH,
        device_id=DEVICE_ID,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,       # T5 sur CPU pour économiser la VRAM GPU
        init_on_cpu=True,
    )
    models["wan_t2v"] = pipe
    print("[wan-t2v] prêt")


def load_wan_i2v():
    if "wan_i2v" in models:
        return
    print(f"[wan-i2v] chargement depuis {WAN_I2V_PATH} ...")
    from wan import WanI2V
    config = _load_wan_config(WAN_I2V_PATH)
    pipe = WanI2V(
        config=config,
        checkpoint_dir=WAN_I2V_PATH,
        device_id=DEVICE_ID,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        init_on_cpu=True,
    )
    models["wan_i2v"] = pipe
    print("[wan-i2v] prêt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await asyncio.gather(
        loop.run_in_executor(None, load_voxtral),
        loop.run_in_executor(None, load_qwen),
        loop.run_in_executor(None, load_wan_t2v),
        return_exceptions=True,
    )
    # Wan I2V chargé à la demande pour économiser la VRAM
    yield
    models.clear()


app = FastAPI(title="ViralVidTech RunPod API", version="2.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schémas
# ---------------------------------------------------------------------------

class WanT2VRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted"
    # frame_num doit être de la forme 4n+1 : 49, 81, 97, 113...
    num_frames: int = 81
    width: int = 832
    height: int = 480
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    fps: int = 16
    seed: int = -1
    # shift recommandé : 1.0 pour 720p T2V, 3.0 pour I2V/animate
    shift: float = 1.0
    sample_solver: str = "unipc"


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _wan_tensor_to_video_bytes(tensor: torch.Tensor, fps: int) -> bytes:
    """
    Convertit le tenseur Wan (C, N, H, W) en bytes MP4.
    Les valeurs sont dans [-1, 1] en sortie du VAE.
    """
    import imageio
    import numpy as np

    # (C, N, H, W) → (N, H, W, C), [-1,1] → [0,255]
    tensor = tensor.float().clamp(-1.0, 1.0)
    frames_np = ((tensor.permute(1, 2, 3, 0).cpu().numpy() + 1.0) / 2.0 * 255.0).astype(np.uint8)

    buf = io.BytesIO()
    with imageio.get_writer(
        buf, fps=fps, format="ffmpeg", codec="libx264",
        output_params=["-f", "mp4", "-pix_fmt", "yuv420p"]
    ) as writer:
        for frame in frames_np:
            writer.append_data(frame)
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
    info = {
        "status": "ok",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": list(models.keys()),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["vram_total_gb"] = round(props.total_memory / 1e9, 1)
        info["vram_free_gb"]  = round(
            (props.total_memory - torch.cuda.memory_allocated(0)) / 1e9, 1
        )
    return info


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
            audio=audio_array, sampling_rate=sample_rate, return_tensors="pt"
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

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    loop  = asyncio.get_event_loop()

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
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
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
    """Génère une vidéo MP4 à partir d'un prompt texte avec Wan 2.2 T2V natif."""
    load_wan_t2v()
    pipe = models["wan_t2v"]
    loop = asyncio.get_event_loop()

    def _run():
        seed = request.seed if request.seed >= 0 else None
        video_tensor = pipe.generate(
            input_prompt=request.prompt,
            size=(request.width, request.height),
            frame_num=request.num_frames,
            sampling_steps=request.num_inference_steps,
            guide_scale=request.guidance_scale,
            n_prompt=request.negative_prompt,
            seed=seed,
            offload_model=True,
            shift=request.shift,
            sample_solver=request.sample_solver,
        )
        return _wan_tensor_to_video_bytes(video_tensor, fps=request.fps)

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
    num_inference_steps: int = Query(default=40),
    guidance_scale: float = Query(default=5.0),
    fps: int = Query(default=16),
    seed: int = Query(default=-1),
    # shift=3.0 recommandé pour l'animation I2V
    shift: float = Query(default=3.0),
    sample_solver: str = Query(default="unipc"),
):
    """Anime une image statique en vidéo MP4 avec Wan 2.2 Animate natif."""
    load_wan_i2v()
    from PIL import Image

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    pipe  = models["wan_i2v"]
    loop  = asyncio.get_event_loop()

    def _run():
        seed_val = seed if seed >= 0 else None
        video_tensor = pipe.generate(
            input_prompt=prompt,
            img=image,
            frame_num=num_frames,
            sampling_steps=num_inference_steps,
            guide_scale=guidance_scale,
            n_prompt=negative_prompt,
            seed=seed_val,
            offload_model=True,
            shift=shift,
            sample_solver=sample_solver,
        )
        return _wan_tensor_to_video_bytes(video_tensor, fps=fps)

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
