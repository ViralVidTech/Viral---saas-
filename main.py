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
            "https://api.anthropic.com/v1/messages",
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

    # Chercher 4 vidéos Pexels selon la niche
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

    # Chercher une musique sur Pixabay
    pixabay_key = os.getenv("PIXABAY_API_KEY")
    music_url = ""
    async with httpx.AsyncClient(timeout=30) as client:
        pixabay_response = await client.get(
            f"https://pixabay.com/api/videos/music/?key={pixabay_key}&q={niche}&per_page=3"
        )
        pixabay_data = pixabay_response.json()
        hits = pixabay_data.get("hits", [])
        if hits:
            music_url = hits[0].get("audio", "")

    return {"titles": titles, "script": script, "video_url": video_urls[0], "video_url2": video_urls[1], "video_url3": video_urls[2], "video_url4": video_urls[3], "music_url": music_url}
@app.post("/create-video")
async def create_video(req: VideoRequest):
    api_key = os.getenv("CREATOMATE_API_KEY")
    template_id = os.getenv("CREATOMATE_TEMPLATE_ID")
    if not api_key or not template_id:
        return {"error": "Missing API key or template ID"}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.creatomate.com/v1/renders",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "template_id": template_id,
                "modifications": {
                    "Text-1.text": req.text1,
                    "Text-2.text": req.text2,
                    "Text-3.text": req.text3,
                    "Text-4.text": req.text4,
                    "Background-1.source": req.video_url,
                    "Background-2.source": req.video_url2,
                    "Background-3.source": req.video_url3,
                    "Background-4.source": req.video_url4,
                }
            }
        )
        return response.json()
