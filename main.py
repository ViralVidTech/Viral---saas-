from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import base64
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
    music_url: str = ""
    voice_url: str = ""

class VoiceRequest(BaseModel):
    text: str
    voice: str

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

    mixkit_music = {
        "motivation": "https://assets.mixkit.co/music/preview/mixkit-life-is-a-dream-837.mp3",
        "business": "https://assets.mixkit.co/music/preview/mixkit-business-motivation-169.mp3",
        "money": "https://assets.mixkit.co/music/preview/mixkit-achieving-878.mp3",
        "education": "https://assets.mixkit.co/music/preview/mixkit-a-very-happy-christmas-897.mp3",
        "ai": "https://assets.mixkit.co/music/preview/mixkit-tech-house-vibes-130.mp3",
    }
    default_music = "https://assets.mixkit.co/music/preview/mixkit-life-is-a-dream-837.mp3"
    music_url = mixkit_music.get(niche.lower(), default_music)

    return {
        "titles": titles,
        "script": script,
        "video_url": video_urls[0],
        "video_url2": video_urls[1],
        "video_url3": video_urls[2],
        "video_url4": video_urls[3],
        "music_url": music_url
    }

@app.post("/create-video")
async def create_video(req: VideoRequest):
    api_key = os.getenv("CREATOMATE_API_KEY")
    template_id = os.getenv("CREATOMATE_TEMPLATE_ID")
    if not api_key or not template_id:
        return {"error": "Missing API key or template ID"}

    modifications = {
        "Text-1.text": req.text1,
        "Text-2.text": req.text2,
        "Text-3.text": req.text3,
        "Text-4.text": req.text4,
        "Background-1.source": req.video_url,
        "Background-2.source": req.video_url2,
        "Background-3.source": req.video_url3,
        "Background-4.source": req.video_url4,
        "Music.source": req.music_url,
    }

    if req.voice_url:
        modifications["Voiceover-WFH.source"] = req.voice_url

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.creatomate.com/v1/renders",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "template_id": template_id,
                "modifications": modifications
            }
        )
        return response.json()

@app.post("/generate-voice")
async def generate_voice(req: VoiceRequest):
    google_key = os.getenv("GOOGLE_TTS_API_KEY")
    if not google_key:
        return {"error": "Clé Google manquante"}

    language_code = req.voice[:5]

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"https://texttospeech.googleapis.com/v1/text:synthesize?key={google_key}",
            json={
                "input": {"text": req.text},
                "voice": {"languageCode": language_code, "name": req.voice},
                "audioConfig": {"audioEncoding": "MP3"}
            }
        )
    data = response.json()
    if "audioContent" not in data:
        return {"error": str(data)}

    audio_bytes = base64.b64decode(data["audioContent"])
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {"audio_b64": audio_b64}

@app.post("/upload-voice")
async def upload_voice(req: VoiceRequest):
    google_key = os.getenv("GOOGLE_TTS_API_KEY")
    api_key = os.getenv("CREATOMATE_API_KEY")
    if not google_key or not api_key:
        return {"error": "Clé manquante"}

    language_code = req.voice[:5]

    async with httpx.AsyncClient(timeout=30) as client:
        tts_response = await client.post(
            f"https://texttospeech.googleapis.com/v1/text:synthesize?key={google_key}",
            json={
                "input": {"text": req.text},
                "voice": {"languageCode": language_code, "name": req.voice},
                "audioConfig": {"audioEncoding": "MP3"}
            }
        )

    tts_data = tts_response.json()
    if "audioContent" not in tts_data:
        return {"error": str(tts_data)}

    audio_bytes = base64.b64decode(tts_data["audioContent"])

    async with httpx.AsyncClient(timeout=60) as client:
        upload_response = await client.post(
            "https://api.creatomate.com/v1/assets",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("voice.mp3", audio_bytes, "audio/mpeg")}
        )

    upload_data = upload_response.json()
    voice_url = upload_data.get("url", "")

    return {"voice_url": voice_url}
