from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import base64
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# CONFIG
SHOTSTACK_API_KEY = os.getenv("SHOTSTACK_API_KEY", "")
GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY", "")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


# MODELS
class GenerateRequest(BaseModel):
    niche: str
    langue: str = "en"


class TTSRequest(BaseModel):
    text: str
    languageCode: str = "en-US"
    voiceName: str = "en-US-Chirp3-HD-Achernar"
    speakingRate: float = 1.0


class VideoRequest(BaseModel):
    text1: str = ""
    text2: str = ""
    text3: str = ""
    text4: str = ""
    video_url: str = ""
    video_url2: str = ""
    video_url3: str = ""
    video_url4: str = ""
    audio_url: str = ""


# HOME
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>API is running</h1>"


# GENERATE SCRIPT + FETCH PEXELS VIDEOS
@app.post("/generate")
async def generate(req: GenerateRequest):
    niche = req.niche.strip() or "general topic"
    lang = (req.langue or "en").lower()

    if lang == "fr":
        titles = [
            f"3 vérités choquantes sur {niche}",
            f"Pourquoi la plupart des gens échouent dans {niche}",
            f"La manière la plus intelligente de réussir dans {niche}",
        ]
        hook = f"La plupart des gens comprennent mal {niche}, et cela leur coûte plus cher qu’ils ne l’imaginent."
        problem = f"Le problème, c’est que beaucoup de personnes abordent {niche} à l’aveugle et répètent toujours les mêmes erreurs."
        solution = f"La solution consiste à utiliser une stratégie simple, rester constant et se concentrer uniquement sur ce qui fonctionne."
        cta = f"Abonne-toi pour plus de vidéos sur {niche}."

    elif lang == "es":
        titles = [
            f"3 verdades impactantes sobre {niche}",
            f"Por qué la mayoría fracasa en {niche}",
            f"La forma más inteligente de triunfar en {niche}",
        ]
        hook = f"La mayoría de la gente entiende mal {niche}, y eso les cuesta más de lo que creen."
        problem = f"El problema es que muchas personas abordan {niche} a ciegas y repiten los mismos errores."
        solution = f"La solución es usar una estrategia simple, ser constante y enfocarse solo en lo que funciona."
        cta = f"Sigue la cuenta para más videos sobre {niche}."

    elif lang == "pt":
        titles = [
            f"3 verdades chocantes sobre {niche}",
            f"Por que a maioria falha em {niche}",
            f"A forma mais inteligente de vencer em {niche}",
        ]
        hook = f"A maioria das pessoas entende {niche} da forma errada, e isso custa mais do que imaginam."
        problem = f"O problema é que muitas pessoas entram em {niche} no escuro e repetem os mesmos erros."
        solution = f"A solução é usar uma estratégia simples, manter consistência e focar apenas no que funciona."
        cta = f"Siga para mais vídeos sobre {niche}."

    else:
        titles = [
            f"3 shocking truths about {niche}",
            f"Why most people fail in {niche}",
            f"The smartest way to win in {niche}",
        ]
        hook = f"Most people misunderstand {niche}, and it costs them more than they think."
        problem = f"The problem is that people approach {niche} blindly and repeat the same mistakes."
        solution = f"The solution is to use a simple strategy, stay consistent, and focus only on what works."
        cta = f"Follow for more videos about {niche}."

    script = f"{hook}\n\n{problem}\n\n{solution}\n\n{cta}"

    video_urls = ["", "", "", ""]

    if PEXELS_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                pexels_response = await client.get(
                    "https://api.pexels.com/videos/search",
                    params={"query": niche, "per_page": 4},
                    headers={"Authorization": PEXELS_API_KEY},
                )

            pexels_data = pexels_response.json()
            videos = pexels_data.get("videos", [])

            for i, video in enumerate(videos[:4]):
                files = video.get("video_files", [])
                hd_files = [f for f in files if f.get("quality") == "hd"]
                if hd_files:
                    video_urls[i] = hd_files[0].get("link", "")
                elif files:
                    video_urls[i] = files[0].get("link", "")
        except Exception:
            pass

    return {
        "titles": titles,
        "script": script,
        "video_url": video_urls[0],
        "video_url2": video_urls[1],
        "video_url3": video_urls[2],
        "video_url4": video_urls[3],
    }


# SERVE AUDIO FILES
@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    file_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        return {"error": "Audio file not found"}
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)


# GOOGLE TTS
@app.post("/generate-audio")
async def generate_audio(req: TTSRequest):
    if not GOOGLE_TTS_API_KEY:
        return {"error": "GOOGLE_TTS_API_KEY manquante"}

    if not PUBLIC_BASE_URL:
        return {"error": "PUBLIC_BASE_URL manquante"}

    if not req.text.strip():
        return {"error": "Le texte est vide"}

    google_url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_TTS_API_KEY}"

    payload = {
        "input": {"text": req.text},
        "voice": {
            "languageCode": req.languageCode,
            "name": req.voiceName,
        },
        "audioConfig": {
            "audioEncoding": "MP3",
            "speakingRate": req.speakingRate,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(google_url, json=payload)

        data = response.json()

        if response.status_code != 200:
            return {
                "error": "Google TTS a échoué",
                "details": data,
            }

        audio_content = data.get("audioContent")
        if not audio_content:
            return {"error": "Aucun audio retourné par Google"}

        filename = f"{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(base64.b64decode(audio_content))

        audio_url = f"{PUBLIC_BASE_URL}/audio/{filename}"

        return {
            "success": True,
            "audio_url": audio_url,
            "filename": filename,
        }

    except Exception as e:
        return {"error": f"Erreur TTS: {str(e)}"}


# SHOTSTACK: CREATE VIDEO
@app.post("/create-video")
async def create_video(req: VideoRequest):
    if not SHOTSTACK_API_KEY:
        return {"error": "SHOTSTACK_API_KEY manquante"}

    video_urls = [req.video_url, req.video_url2, req.video_url3, req.video_url4]
    texts = [req.text1, req.text2, req.text3, req.text4]

    clips_video = []
    clips_text = []
    start_time = 0

    for i in range(4):
        duration = 5

        if video_urls[i]:
            clips_video.append({
                "asset": {
                    "type": "video",
                    "src": video_urls[i]
                },
                "start": start_time,
                "length": duration,
                "fit": "cover"
            })

        if texts[i].strip():
            clips_text.append({
                "asset": {
                    "type": "title",
                    "text": texts[i],
                    "style": "bold",
                    "color": "#ffffff",
                    "size": "medium"
                },
                "start": start_time,
                "length": duration,
                "position": "center"
            })

        start_time += duration

    if not clips_video:
        return {"error": "Aucune vidéo n'a été fournie"}

    timeline = {
        "tracks": [
            {"clips": clips_video},
            {"clips": clips_text},
        ]
    }

    if req.audio_url.strip():
        timeline["soundtrack"] = {
            "src": req.audio_url
        }

    payload = {
        "timeline": timeline,
        "output": {
            "format": "mp4",
            "aspectRatio": "9:16",
            "resolution": "sd"
        }
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.shotstack.io/edit/stage/render",
                headers={
                    "x-api-key": SHOTSTACK_API_KEY,
                    "Content-Type": "application/json"
                },
                json=payload
            )

        data = response.json()

        if response.status_code not in [200, 201]:
            return {
                "error": "Shotstack a refusé le rendu",
                "details": data
            }

        render_id = data.get("response", {}).get("id")
        if not render_id:
            return {
                "error": "Shotstack n'a pas renvoyé d'id",
                "details": data
            }

        return {
            "success": True,
            "render_id": render_id,
            "message": "Vidéo envoyée à Shotstack. Vérifie ensuite le statut."
        }

    except Exception as e:
        return {"error": f"Erreur Shotstack: {str(e)}"}


# SHOTSTACK: CHECK STATUS
@app.get("/render-status/{render_id}")
async def render_status(render_id: str):
    if not SHOTSTACK_API_KEY:
        return {"error": "SHOTSTACK_API_KEY manquante"}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(
                f"https://api.shotstack.io/edit/stage/render/{render_id}",
                headers={
                    "x-api-key": SHOTSTACK_API_KEY,
                    "Content-Type": "application/json"
                }
            )

        data = response.json()

        if response.status_code != 200:
            return {
                "error": "Impossible de lire le statut Shotstack",
                "details": data
            }

        info = data.get("response", {})

        return {
            "success": True,
            "status": info.get("status"),
            "url": info.get("url"),
            "id": info.get("id"),
        }

    except Exception as e:
        return {"error": f"Erreur statut Shotstack: {str(e)}"}
