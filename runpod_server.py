import os, asyncio, uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

app = FastAPI(title="ViralVidTech RunPod API", version="3.0.0")

WORK_DIR = "/workspace/outputs"
os.makedirs(WORK_DIR, exist_ok=True)

WAN_CODE = "/workspace/wan2.2-code"
WAN_T2V_CKPT = "/workspace/wan2.2-t2v"
WAN_ANIMATE_CKPT = "/workspace/wan2.2-animate"

@app.get("/health")
async def health():
    return {"status": "ok", "gpu": "H100", "version": "3.0.0"}

@app.post("/wan/generate")
async def wan_generate(
    prompt: str = Form(...),
    size: str = Form("832*480"),
    sample_steps: int = Form(20),
):
    output_path = f"{WORK_DIR}/{uuid.uuid4().hex}.mp4"
    cmd = [
        "python3", f"{WAN_CODE}/generate.py",
        "--task", "t2v-A14B",
        "--size", size,
        "--ckpt_dir", WAN_T2V_CKPT,
        "--prompt", prompt,
        "--sample_steps", str(sample_steps),
        "--save_file", output_path
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=WAN_CODE
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=stderr.decode()[-2000:])
    if not os.path.exists(output_path):
        raise HTTPException(status_code=500, detail="Video file not created")
    def iter_file():
        with open(output_path, "rb") as f:
            yield from iter(lambda: f.read(65536), b"")
        os.remove(output_path)
    return StreamingResponse(iter_file(), media_type="video/mp4")

@app.post("/wan/animate")
async def wan_animate(
    character_image: UploadFile = File(...),
    reference_video: UploadFile = File(...),
    mode: str = Form("animation"),
    sample_steps: int = Form(20),
):
    img_path = f"{WORK_DIR}/{uuid.uuid4().hex}.jpg"
    vid_path = f"{WORK_DIR}/{uuid.uuid4().hex}.mp4"
    out_path = f"{WORK_DIR}/{uuid.uuid4().hex}.mp4"
    try:
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
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=WAN_CODE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=stderr.decode()[-2000:])
        if not os.path.exists(out_path):
            raise HTTPException(status_code=500, detail="Animated video not created")
        def iter_file():
            with open(out_path, "rb") as f:
                yield from iter(lambda: f.read(65536), b"")
            os.remove(out_path)
        return StreamingResponse(iter_file(), media_type="video/mp4")
    finally:
        for p in [img_path, vid_path]:
            if os.path.exists(p):
                os.remove(p)

@app.post("/qwen/generate-image")
async def qwen_generate():
    return JSONResponse({"error": "coming soon"}, status_code=503)

@app.post("/voxtral/transcribe")
async def voxtral_transcribe():
    return JSONResponse({"error": "coming soon"}, status_code=503)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
