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

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/generate")
async def generate(req: GenerateRequest):
    niche = req.niche.strip() or "general"
    titles = [
        f"How to make money with {niche} in 2025",
        f"The {niche} secret nobody talks about",
        f"I tried {niche} for 30 days — here's what happened",
    ]
    script = (
        f"Hook: Did you know that {niche} is changing everything right now?\n\n"
        f"Problem: Most people ignore {niche} and lose out on huge opportunities.\n\n"
        f"Solution: Here's exactly how you can leverage {niche} starting today.\n\n"
        f"CTA: Follow for more {niche} content every week!"
    )
    return {"titles": titles, "script": script}

@app.post("/create-video")
async def create_video(req: VideoRequest):
    api_key = os.getenv("CREATOMATE_API_KEY")
    template_id = os.getenv("CREATOMATE_TEMPLATE_ID")
    async with httpx.AsyncClient() as client:
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
                }
            }
        )
    return response.json()
