import os, asyncio, uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

app = FastAPI(title="ViralVidTech RunPod API", version="4.0.0")

WORK_DIR = "/workspace/outputs"
os.makedirs(WORK_DIR, exist_ok=True)

WAN_CODE = "/workspace/wan2.2-code"
WAN_T2V_CKPT = "/workspace/wan2.2-t2v"
WAN_ANIMATE_CKPT = "/workspace/wan2.2-animate"

WAN_JOBS = {}

@app.get("/health")
async def health():
    return {"status": "ok", "gpu": "H100", "version": "4.0.0"}

async def _run_wan_job(job_id: str, cmd: list):
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=WAN_CODE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            WAN_JOBS[job_id] = {"status": "error", "detail": stderr.decode()[-2000:]}
            return
        video_path = f"{WORK_DIR}/{job_id}.mp4"
        if not os.path.exists(video_path):
            WAN_JOBS[job_id] = {"status": "error", "detail": "Video file not created"}
            return
        WAN_JOBS[job_id] = {"status": "done", "video_path": video_path}
    except Exception as e:
        WAN_JOBS[job_id] = {"status": "error", "detail": str(e)}
    finally:
        try:
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

@app.post("/wan/generate")
async def wan_generate(
    prompt: str = Form(...),
    size: str = Form("832*480"),
    sample_steps: int = Form(20),
    frame_num: int = Form(97),
    image_url: str = Form(""),
):
    job_id = uuid.uuid4().hex
    output_path = f"{WORK_DIR}/{job_id}.mp4"
    if image_url:
        cmd = [
            "python3", f"{WAN_CODE}/generate.py",
            "--task", "i2v-A14B",
            "--size", size,
            "--ckpt_dir", WAN_ANIMATE_CKPT,
            "--image", image_url,
            "--prompt", prompt,
            "--sample_steps", str(sample_steps),
            "--frame_num", str(frame_num),
            "--save_file", output_path
        ]
    else:
        cmd = [
            "python3", f"{WAN_CODE}/generate.py",
            "--task", "t2v-A14B",
            "--size", size,
            "--ckpt_dir", WAN_T2V_CKPT,
            "--prompt", prompt,
            "--sample_steps", str(sample_steps),
            "--frame_num", str(frame_num),
            "--save_file", output_path
        ]
    WAN_JOBS[job_id] = {"status": "processing"}
    asyncio.create_task(_run_wan_job(job_id, cmd))
    return JSONResponse({"job_id": job_id, "status": "processing"})

@app.get("/wan/status/{job_id}")
async def wan_status(job_id: str):
    job = WAN_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job)

@app.get("/wan/video/{job_id}")
async def wan_video(job_id: str):
    job = WAN_JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Video not ready")
    video_path = job["video_path"]
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file missing")
    def iter_file():
        with open(video_path, "rb") as f:
            yield from iter(lambda: f.read(65536), b"")
    return StreamingResponse(iter_file(), media_type="video/mp4")

@app.post("/wan/animate")
async def wan_animate(
    character_image: UploadFile = File(...),
    reference_video: UploadFile = File(...),
    mode: str = Form("animation"),
    sample_steps: int = Form(20),
):
    job_id = uuid.uuid4().hex
    img_path = f"{WORK_DIR}/{job_id}_img.jpg"
    vid_path = f"{WORK_DIR}/{job_id}_ref.mp4"
    out_path = f"{WORK_DIR}/{job_id}.mp4"
    with open(img_path, "wb") as f:
        f.write(await character_image.read())
    with open(vid_path, "wb") as f:
        f.write(await reference_video.read())
    cmd = [
        "python3", f"{WAN_CODE}/generate.py",
        "--task", "i2v-A14B",
        "--ckpt_dir", WAN_ANIMATE_CKPT,
        "--image", img_path,
        "--pose_video", vid_path,
        "--save_file", out_path,
        "--sample_steps", str(sample_steps),
        "--size", "832*480",
        "--offload_model", "True",
    ]
    WAN_JOBS[job_id] = {"status": "processing"}
    asyncio.create_task(_run_wan_job(job_id, cmd))
    return JSONResponse({"job_id": job_id, "status": "processing"})

@app.post("/qwen/generate-image")
async def qwen_generate():
    return JSONResponse({"error": "coming soon"}, status_code=503)

@app.post("/voxtral/transcribe")
async def voxtral_transcribe(
    audio_file: UploadFile = File(...),
    language: str = Form("fr"),
    output_format: str = Form("text"),
):
    job_id = uuid.uuid4().hex
    ext = os.path.splitext(audio_file.filename)[1].lower() if audio_file.filename else ".mp3"
    audio_path = f"{WORK_DIR}/{job_id}{ext}"
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read())

    script = f"""
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor
processor = AutoProcessor.from_pretrained('/workspace/voxtral', local_files_only=True)
model = VoxtralForConditionalGeneration.from_pretrained('/workspace/voxtral', local_files_only=True, torch_dtype=torch.bfloat16, device_map='cuda')
inputs = processor.apply_transcription_request(language='{language}', audio='{audio_path}', model_id='mistralai/Voxtral-Mini-3B-2507')
inputs = inputs.to('cuda', dtype=torch.bfloat16)
outputs = model.generate(**inputs, max_new_tokens=1000)
decoded = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(decoded[0])
"""

    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return JSONResponse(status_code=500, content={"error": stderr.decode()[-2000:]})
        transcription = stdout.decode().strip()
        return JSONResponse({"transcription": transcription, "language": language})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
