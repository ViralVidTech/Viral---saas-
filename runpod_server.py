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

@app.post("/wan/generate")
async def wan_generate(
    prompt: str = Form(...),
    size: str = Form("832*480"),
    sample_steps: int = Form(20),
):
    job_id = uuid.uuid4().hex
    output_path = f"{WORK_DIR}/{job_id}.mp4"
    cmd = [
        "python3", f"{WAN_CODE}/generate.py",
        "--task", "t2v-A14B",
        "--size", size,
        "--ckpt_dir", WAN_T2V_CKPT,
        "--prompt", prompt,
        "--sample_steps", str(sample_steps),
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
        os.remove(video_path)
        del WAN_JOBS[job_id]
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
        "--task", "animate-14B",
        "--ckpt_dir", WAN_ANIMATE_CKPT,
        "--image", img_path,
        "--pose_video", vid_path,
        "--save_file", out_path,
        "--sample_steps", str(sample_steps),
    ]
    WAN_JOBS[job_id] = {"status": "processing"}
    asyncio.create_task(_run_wan_job(job_id, cmd))
    return JSONResponse({"job_id": job_id, "status": "processing"})

@app.post("/qwen/generate-image")
async def qwen_generate():
    return JSONResponse({"error": "coming soon"}, status_code=503)

@app.post("/voxtral/transcribe")
async def voxtral_transcribe():
    return JSONResponse({"error": "coming soon"}, status_code=503)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
