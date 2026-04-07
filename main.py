from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    niche: str
    langue: str = "en"

class VideoRequest(BaseModel):
    text1: str
    text2: str
    text3: str
    text4: str
    video_url: str = ""
    video_url2: str = ""
    video_url3: str = ""
    video_url4: str = ""

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/generate")
async def generate(req: GenerateRequest):
    niche = req.niche.strip() or "general"
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.shotstack.io/stage/render",
            headers={
                "x-api-key": anthropic_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Create viral content about '{niche}' in language '{req.langue}'.
Return exactly this format:

TITLES:
1. [title 1]
2. [title 2]
3. [title 3]

HOOK: [one punchy sentence]

PROBLEM: [one sentence about the problem]

SOLUTION: [one sentence about the solution]

CTA: [one call to action sentence]"""
                    }
                ]
            }
        )

    data = response.json()
    if "content" not in data:
        return {"titles": [], "script": str(data)}

    text = data["content"][0]["text"]
    lines = text.strip().split("\n")
    titles = []
    hook = problem = solution = cta = ""

    for line in lines:
        line = line.strip()
        if line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
            titles.append(line[2:].strip())
        elif line.startswith("HOOK:"):
            hook = line.replace("HOOK:", "").strip()
        elif line.startswith("PROBLEM:"):
            problem = line.replace("PROBLEM:", "").strip()
        elif line.startswith("SOLUTION:"):
            solution = line.replace("SOLUTION:", "").strip()
        elif line.startswith("CTA:"):
            cta = line.replace("CTA:", "").strip()

    script = f"{hook}\n\n{problem}\n\n{solution}\n\n{cta}"

    pexels_key = os.getenv("PEXELS_API_KEY")
    video_urls = ["", "", "", ""]

    async with httpx.AsyncClient(timeout=30) as client:
        pexels_response = await client.get(
            f"https://api.pexels.com/videos/search?query={niche}&per_page=4",
            headers={"Authorization": pexels_key}
        )
        pexels_data = pexels_response.json()
        videos = pexels_data.get("videos", [])
        for i, video in enumerate(videos[:4]):
            files = video.get("video_files", [])
            hd_files = [f for f in files if f.get("quality") == "hd"]
            if hd_files:
                video_urls[i] = hd_files[0]["link"]
            elif files:
                video_urls[i] = files[0]["link"]

    return {
        "titles": titles,
        "script": script,
        "video_url": video_urls[0],
        "video_url2": video_urls[1],
        "video_url3": video_urls[2],
        "video_url4": video_urls[3],
    }

@app.post("/create-video")
async def create_video(req: VideoRequest):
    shotstack_key = os.getenv("SHOTSTACK_API_KEY")
    if not shotstack_key:
        return {"error": "Clé Shotstack manquante"}

    clips = []
    texts = [req.text1, req.text2, req.text3, req.text4]
    videos = [req.video_url, req.video_url2, req.video_url3, req.video_url4]
    start_time = 0

    for i in range(4):
        duration = 5
        if videos[i]:
            clips.append({
                "asset": {"type": "video", "src": videos[i]},
                "start": start_time,
                "length": duration,
                "fit": "cover"
            })
        clips.append({
            "asset": {
                "type": "title",
                "text": texts[i],
                "style": "bold",
                "color": "#ffffff",
                "size": "medium"
            },
            "start": start_time,
            "length": duration
        })
        start_time += duration

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.shotstack.io/v1/render",
            headers={
                "x-api-key": shotstack_key,
                "Content-Type": "application/json"
            },
            json={
                "timeline": {
                    "tracks": [
                        {"clips": [c for c in clips if c["asset"]["type"] == "video"]},
                        {"clips": [c for c in clips if c["asset"]["type"] == "title"]}
                    ]
                },
                "output": {
                    "format": "mp4",
                    "resolution": "hd",
                    "aspectRatio": "9:16"
                }
            }
        )
        data = response.json()
        return {"debug": data}
