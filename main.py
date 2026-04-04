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

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/generate")
async def generate(req: GenerateRequest):
    niche = req.niche.strip() or "general"
    openai_key = os.getenv("OPENAI_API_KEY")

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a viral content creator. Respond only in the requested language."
                    },
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
    text = data["choices"][0]["message"]["content"]

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

    return {"titles": titles, "script": script}

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
                }
            }
        )
        return response.json()
