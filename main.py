from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import base64
import json
import os
import httpx
import subprocess
import shutil
import asyncio

app = FastAPI()

VIDEO_JOBS = {}
WAN_JOBS = {}  # job_id -> {"status": "processing/done/error", "video_path": str, "detail": str}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# CONFIG
GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY", "")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
FAL_API_KEY = os.getenv("FAL_API_KEY", "")
FISH_AUDIO_API_KEY = os.getenv("FISH_AUDIO_API_KEY", "")
WAN_API_URL = os.getenv("WAN_API_URL", "")
RUNPOD_API_URL = os.getenv("RUNPOD_API_URL", "")
RUNPOD_BASE_URL = os.getenv("RUNPOD_API_URL", "")
RUNPOD_TIMEOUT = float(os.getenv("RUNPOD_TIMEOUT", "300"))
QWEN_API_URL = os.getenv("QWEN_API_URL", "")
WAN_ANIMATE_API_URL = os.getenv("WAN_ANIMATE_API_URL", "")
VOXTRAL_API_URL = os.getenv("VOXTRAL_API_URL", "")

AUDIO_DIR = "audio"
VIDEO_DIR = "videos"
WORK_DIR = "work"
MUSIC_DIR = "music"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(MUSIC_DIR, exist_ok=True)

def _runpod_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=RUNPOD_BASE_URL, timeout=RUNPOD_TIMEOUT)


class WanT2VRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality"
    num_frames: int = 81
    width: int = 832
    height: int = 480
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    fps: int = 16


VIDEO_TYPES = [
    "Vidéo stock",
    "Paper cut-out",
    "Personnage animé",
    "Talking avatar",
    "Vidéo explicative",
    "Citation visuelle",
    "Témoignage",
    "Tutoriel",
    "Révélation & Choc",
    "Challenge",
]

NARRATIVE_STYLES = [
    "Storytelling",
    "Éducatif",
    "Humoristique",
    "Choc & Révélation",
    "Inspirant",
    "Informatif",
    "Conversationnel",
    "Dramatique",
]

VOICE_STYLES = {
    "Lent et posé": ["narration", "méditation", "religion", "développement personnel"],
    "Dynamique et rapide": ["motivation", "sport", "gaming", "challenge"],
    "Doux et chaleureux": ["amour", "parentalité", "bien-être", "cuisine"],
    "Autoritaire et confiant": ["finance", "business", "technologie"],
    "Humoristique et léger": ["blagues", "entertainment"],
    "Éducatif et clair": ["IA", "langues", "tutoriel"],
}

MUSIC_STYLES = {
    "Épique et motivant": ["motivation", "sport", "finance"],
    "Doux et apaisant": ["santé", "méditation", "religion"],
    "Rythmé et énergique": ["gaming", "challenge", "influenceur"],
    "Mystérieux et intrigant": ["révélation", "technologie"],
    "Neutre et professionnel": ["éducation", "langues", "IA"],
    "Joyeux et léger": ["blagues", "cuisine", "voyage"],
}

NICHES_BY_PLAN = {
    "free": [
        "Motivation",
        "Blagues",
    ],
    "starter": [
        "Motivation",
        "Blagues",
        "Santé",
        "Religion",
    ],
    "pro": [
        "Motivation",
        "Blagues",
        "Santé",
        "Religion",
        "IA",
        "Technologie",
        "Enseignement de langues",
        "Développement personnel",
        "Sport & Fitness",
        "Mindfulness",
    ],
    "premium": [
        "Motivation",
        "Blagues",
        "Santé",
        "Religion",
        "IA",
        "Technologie",
        "Enseignement de langues",
        "Développement personnel",
        "Sport & Fitness",
        "Mindfulness",
        "Finance",
        "Cryptomonnaie",
        "Influenceur",
        "Relation amoureuse",
        "Mode & Beauté",
        "Gaming",
        "Voyage",
        "Cuisine",
        "Parentalité",
        "Développement business",
    ],
}

PUBLISH_PLANS = {
    "free": {
        "manual_publications_per_day": 0,
        "auto_publications_per_day": 0,
        "platforms": [],
        "auto_publish": False,
        "price_monthly": 0
    },
    "starter": {
        "manual_publications_per_day": 1,
        "auto_publications_per_day": 1,
        "platforms": ["tiktok", "youtube", "instagram", "facebook"],
        "auto_publish": True,
        "price_monthly_manual": 19,
        "price_monthly_auto": 29
    },
    "pro": {
        "manual_publications_per_day": 2,
        "auto_publications_per_day": 2,
        "platforms": ["tiktok", "youtube", "instagram", "facebook"],
        "auto_publish": True,
        "price_monthly_manual": 49,
        "price_monthly_auto": 69
    },
    "premium": {
        "manual_publications_per_day": 4,
        "auto_publications_per_day": 4,
        "platforms": ["tiktok", "youtube", "instagram", "facebook"],
        "auto_publish": True,
        "price_monthly_manual": 99,
        "price_monthly_auto": 129
    }
}

PLATFORM_CONFIGS = {
    "tiktok": {
        "name": "TikTok",
        "env_key": "TIKTOK_ACCESS_TOKEN",
        "max_duration_seconds": 60,
        "supported_formats": ["mp4"],
        "optimal_post_times": ["18:00", "19:00", "20:00"]
    },
    "youtube": {
        "name": "YouTube Shorts",
        "env_key": "YOUTUBE_ACCESS_TOKEN",
        "max_duration_seconds": 60,
        "supported_formats": ["mp4"],
        "optimal_post_times": ["15:00", "17:00", "20:00"]
    },
    "instagram": {
        "name": "Instagram Reels",
        "env_key": "INSTAGRAM_ACCESS_TOKEN",
        "max_duration_seconds": 90,
        "supported_formats": ["mp4"],
        "optimal_post_times": ["11:00", "13:00", "19:00"]
    },
    "facebook": {
        "name": "Facebook",
        "env_key": "FACEBOOK_ACCESS_TOKEN",
        "max_duration_seconds": 120,
        "supported_formats": ["mp4"],
        "optimal_post_times": ["12:00", "15:00", "18:00"]
    }
}

PUBLISH_JOBS = {}


# ── FONCTION WAN 2.2 ────────────────────────────────────────────────────────
async def generate_wan_video(prompt: str) -> str:
    """
    Appelle api_wan.py sur RunPod.
    - Accepte GET et POST
    - Logs clairs si erreur
    - Retourne URL complète de la vidéo ou chaine vide
    """
    if not WAN_API_URL:
        print("WAN SKIP: WAN_API_URL non configuree dans Render")
        return ""

    base = WAN_API_URL.rstrip("/")

    # Tentative 1 : POST
    try:
        async with httpx.AsyncClient(timeout=900) as client:
            response = await client.post(
                f"{base}/generate",
                params={"prompt": prompt},
                headers={"Content-Type": "application/json"}
            )

        print(f"WAN POST status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            video_url = data.get("video_url", "")
            if video_url:
                # Construire URL absolue si relative
                if video_url.startswith("http"):
                    return video_url
                return f"{base}{video_url}"
            else:
                print(f"WAN POST: pas de video_url dans reponse: {data}")

        else:
            print(f"WAN POST erreur {response.status_code}: {response.text[:300]}")

    except Exception as e:
        print(f"WAN POST exception: {e}")

    # Tentative 2 : GET (fallback)
    try:
        async with httpx.AsyncClient(timeout=900) as client:
            response = await client.get(
                f"{base}/generate",
                params={"prompt": prompt}
            )

        print(f"WAN GET status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            video_url = data.get("video_url", "")
            if video_url:
                if video_url.startswith("http"):
                    return video_url
                return f"{base}{video_url}"
            else:
                print(f"WAN GET: pas de video_url dans reponse: {data}")
        else:
            print(f"WAN GET erreur {response.status_code}: {response.text[:300]}")

    except Exception as e:
        print(f"WAN GET exception: {e}")

    print("WAN FINAL: echec total, retour chaine vide")
    return ""


# MODELS
class GenerateRequest(BaseModel):
    niche: str
    langue: str = "en"
    duration: int = 30


class TTSRequest(BaseModel):
    text: str
    languageCode: str = "en-US"
    voiceName: str = "en-US-Chirp3-HD-Achernar"
    speakingRate: float = 1.0


class FishTTSRequest(BaseModel):
    text: str
    voice_id: str = "a5474df3-4f8e-4e4c-b5e3-d70a7c1c7dc1"
    language: str = "en"
    format: str = "mp3"
    latency: str = "normal"


class FluxImageRequest(BaseModel):
    prompt: str
    image_size: str = "portrait_4_3"
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    num_images: int = 1
    enable_safety_checker: bool = True


class VideoRequest(BaseModel):
    text1: str = ""
    text2: str = ""
    text3: str = ""
    text4: str = ""
    text5: str = ""
    text6: str = ""
    text7: str = ""
    text8: str = ""
    video_url: str = ""
    video_url2: str = ""
    video_url3: str = ""
    video_url4: str = ""
    video_url5: str = ""
    video_url6: str = ""
    video_url7: str = ""
    video_url8: str = ""
    video_url9: str = ""
    video_url10: str = ""
    video_url11: str = ""
    video_url12: str = ""
    video_url13: str = ""
    video_url14: str = ""
    video_url15: str = ""
    video_url16: str = ""
    video_url17: str = ""
    video_url18: str = ""
    video_url19: str = ""
    video_url20: str = ""
    video_url21: str = ""
    video_url22: str = ""
    video_url23: str = ""
    video_url24: str = ""
    video_url25: str = ""
    video_url26: str = ""
    video_url27: str = ""
    video_url28: str = ""
    video_url29: str = ""
    video_url30: str = ""
    video_url31: str = ""
    video_url32: str = ""
    video_url33: str = ""
    video_url34: str = ""
    video_url35: str = ""
    video_url36: str = ""
    video_url37: str = ""
    video_url38: str = ""
    video_url39: str = ""
    video_url40: str = ""
    audio_url: str = ""
    sync_url: str = ""
    music_url: str = ""
    wan_video: str = ""
    duration: int = 30


# ── UTILITAIRES FFMPEG ──────────────────────────────────────────────────────

def ffmpeg_exists():
    return shutil.which("ffmpeg") is not None


def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Commande échouée:\n{' '.join(cmd)}"
            f"\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return result


async def async_run_cmd(cmd):
    return await asyncio.to_thread(run_cmd, cmd)


async def download_file(url: str, dest_path: str, retries: int = 3, delay: float = 2.0):
    last_error = None
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                follow_redirects=True
            ) as client:
                async with client.stream("GET", url) as response:
                    if response.status_code in (502, 503, 504):
                        await asyncio.sleep(delay * (attempt + 1))
                        continue
                    response.raise_for_status()
                    with open(dest_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            f.write(chunk)
                return
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                await asyncio.sleep(delay * (attempt + 1))
    raise last_error or RuntimeError(f"Échec téléchargement après {retries} tentatives: {url}")


async def download_audio_file(url: str, dest_path: str, retries: int = 4, delay: float = 3.0):
    last_error = None
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                follow_redirects=True
            ) as client:
                response = await client.get(url)
                if response.status_code in (502, 503, 504):
                    await asyncio.sleep(delay * (attempt + 1))
                    continue
                response.raise_for_status()
                with open(dest_path, "wb") as f:
                    f.write(response.content)
                return
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                await asyncio.sleep(delay * (attempt + 1))
    raise last_error or RuntimeError(f"Échec téléchargement audio: {url}")


def srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(subtitle_texts: list, segment_duration: float, out_path: str):
    WORDS_PER_BLOCK = 5
    all_texts = []
    for text in subtitle_texts:
        clean = " ".join((text or "").strip().split())
        if clean:
            all_texts.append(clean)

    if not all_texts:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("")
        return

    total_words = sum(len(t.split()) for t in all_texts)
    total_duration = len(all_texts) * segment_duration

    if total_words > 0:
        seconds_per_word = total_duration / total_words
    else:
        seconds_per_word = 0.35

    entries = []
    idx = 1
    current_time = 0.0
    ADVANCE = 0.7

    for text in all_texts:
        words = text.split()
        for j in range(0, len(words), WORDS_PER_BLOCK):
            block = words[j:j + WORDS_PER_BLOCK]
            block_text = " ".join(block)
            block_word_count = len(block)
            duration = block_word_count * seconds_per_word
            start = max(0.0, current_time - ADVANCE)
            end = max(start + 0.1, start + duration - 0.05)
            entries.append(
                f"{idx}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{block_text}\n"
            )
            idx += 1
            current_time += duration

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries))


def escape_srt_path(path_str: str) -> str:
    return path_str.replace("\\", "\\\\").replace(":", "\\:")


def get_audio_duration(audio_path: str) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# ── ROUTES ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>API is running</h1>"


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    file_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        return {"error": "Audio file not found"}
    if filename.endswith(".json"):
        return FileResponse(file_path, media_type="application/json", filename=filename)
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)


@app.get("/video/{filename}")
async def serve_video(filename: str):
    file_path = os.path.join(VIDEO_DIR, filename)
    if not os.path.exists(file_path):
        return {"error": "Video file not found"}
    return FileResponse(file_path, media_type="video/mp4", filename=filename)


# ── FISH AUDIO TTS ──────────────────────────────────────────────────────────
@app.post("/generate-audio-fish")
async def generate_audio_fish(req: FishTTSRequest):
    if not FISH_AUDIO_API_KEY:
        return {"error": "FISH_AUDIO_API_KEY manquante dans Render"}
    if not PUBLIC_BASE_URL:
        return {"error": "PUBLIC_BASE_URL manquante"}
    if not req.text.strip():
        return {"error": "Le texte est vide"}

    try:
        payload = {
            "text": req.text,
            "format": "mp3",
            "latency": "balanced",
            "normalize": True,
        }

        if req.voice_id:
            payload["reference_id"] = req.voice_id

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://api.fish.audio/v1/tts",
                headers={
                    "Authorization": f"Bearer {FISH_AUDIO_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        if response.status_code != 200:
            try:
                details = response.json()
            except Exception:
                details = response.text
            return {
                "error": f"Fish Audio API erreur {response.status_code}",
                "details": details
            }

        filename = f"fish_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(response.content)

        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return {"error": "Fish Audio a retourné un fichier vide"}

        audio_url = f"{PUBLIC_BASE_URL}/audio/{filename}"

        sync_filename = filename.replace(".mp3", "_sync.json")
        sync_filepath = os.path.join(AUDIO_DIR, sync_filename)

        words = req.text.strip().split()
        with open(sync_filepath, "w", encoding="utf-8") as f:
            json.dump({"words": words, "timepoints": []}, f)

        return {
            "success": True,
            "audio_url": audio_url,
            "sync_url": f"{PUBLIC_BASE_URL}/audio/{sync_filename}",
            "filename": filename,
            "provider": "fish_audio"
        }

    except Exception as e:
        return {"error": f"Erreur Fish Audio: {str(e)}"}
@app.post("/generate-image")
async def generate_image(req: FluxImageRequest):
    if not FAL_API_KEY:
        return {"error": "FAL_API_KEY manquante — ajoutez-la dans les variables d'environnement Render"}
    if not req.prompt.strip():
        return {"error": "Le prompt est vide"}

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://fal.run/fal-ai/flux-pro/v1.1-ultra",
                headers={
                    "Authorization": f"Key {FAL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "prompt": req.prompt,
                    "image_size": req.image_size,
                    "num_inference_steps": req.num_inference_steps,
                    "guidance_scale": req.guidance_scale,
                    "num_images": req.num_images,
                    "enable_safety_checker": req.enable_safety_checker,
                    "output_format": "jpeg",
                },
            )

        if response.status_code != 200:
            try:
                error_detail = response.json()
            except Exception:
                error_detail = response.text
            return {"error": f"Flux API erreur {response.status_code}", "details": error_detail}

        data = response.json()
        images = data.get("images", [])

        if not images:
            return {"error": "Flux n'a retourné aucune image", "details": data}

        image_urls = [img.get("url", "") for img in images if img.get("url")]

        return {
            "success": True,
            "images": image_urls,
            "image_url": image_urls[0] if image_urls else "",
            "prompt": req.prompt,
            "provider": "flux_pro_fal",
            "seed": data.get("seed"),
        }

    except Exception as e:
        return {"error": f"Erreur Flux: {str(e)}"}


# ── LISTE DES VOIX FISH AUDIO ───────────────────────────────────────────────
@app.get("/fish-voices")
async def list_fish_voices():
    if not FISH_AUDIO_API_KEY:
        return {"error": "FISH_AUDIO_API_KEY manquante"}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                "https://api.fish.audio/v1/model",
                headers={"Authorization": f"Bearer {FISH_AUDIO_API_KEY}"},
                params={"page_size": 20, "sort_by": "task_count"},
            )

        if response.status_code != 200:
            return {"error": f"Fish Audio API erreur {response.status_code}"}

        data = response.json()
        voices = []
        for item in data.get("items", []):
            voices.append({
                "id": item.get("_id", ""),
                "name": item.get("title", ""),
                "language": item.get("languages", []),
                "description": item.get("description", ""),
            })

        return {"success": True, "voices": voices}

    except Exception as e:
        return {"error": f"Erreur Fish Audio voices: {str(e)}"}


# ── GENERATE : Claude + Wan 2.2 ─────────────────────────────────────────────
@app.post("/generate")
async def generate(req: GenerateRequest):
    niche = req.niche.strip() or "general topic"
    lang = (req.langue or "en").lower()
    duration = req.duration if req.duration in [30, 45, 60] else 30

    if not ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY manquante"}

    lang_map = {
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "pt": "Portuguese",
    }
    target_language = lang_map.get(lang, "English")

    if duration == 60:
        nb_scenes = 8
        seconds_per_scene = 7
        scene_structure = """HOOK: [accroche choc qui arrête le scroll - 2 phrases complètes et développées]

CONTEXT: [contexte qui pose le sujet et crée de la curiosité - 2 phrases complètes]

PROBLEM: [décris le problème en détail, avec émotion - 2 phrases complètes]

AGITATION: [amplifie la douleur du problème, rends-le urgent - 2 phrases complètes]

SOLUTION: [présente la solution clairement et en détail - 2 phrases complètes]

PROOF: [donne une preuve concrète, un chiffre, un exemple réel - 2 phrases complètes]

BENEFIT: [décris le bénéfice concret que l'audience va obtenir - 2 phrases complètes]

CTA: [appel à l'action direct et motivant - 2 phrases complètes]"""
    elif duration == 45:
        nb_scenes = 6
        seconds_per_scene = 7
        scene_structure = """HOOK: [accroche choc qui arrête le scroll - 2 phrases complètes et développées]

CONTEXT: [contexte qui pose le sujet - 2 phrases complètes]

PROBLEM: [décris le problème avec émotion - 2 phrases complètes]

SOLUTION: [présente la solution en détail - 2 phrases complètes]

PROOF: [preuve concrète ou exemple réel - 2 phrases complètes]

CTA: [appel à l'action direct - 2 phrases complètes]"""
    else:
        nb_scenes = 4
        seconds_per_scene = 7
        scene_structure = """HOOK: [accroche choc - 2 phrases complètes et développées]

PROBLEM: [problème avec émotion - 2 phrases complètes]

SOLUTION: [solution en détail - 2 phrases complètes]

CTA: [appel à l'action - 2 phrases complètes]"""

    prompt = f"""
Tu es un expert en création de contenu viral pour les réseaux sociaux.
Crée un script vidéo complet de {duration} secondes sur le sujet : "{niche}".
Le script doit être en {target_language}.
La vidéo a {nb_scenes} scènes. Chaque scène dure environ {seconds_per_scene} secondes à l'oral.

RÈGLES IMPORTANTES :
- Chaque scène doit contenir EXACTEMENT 2 phrases complètes et bien développées
- Chaque phrase doit faire entre 15 et 25 mots
- Le ton doit être naturel, conversationnel, persuasif
- Le contenu doit être réel, informatif, pas générique
- Écris tout en {target_language}

Retourne EXACTEMENT dans ce format, sans rien ajouter d'autre :

TITLES:
1. [titre accrocheur 1]
2. [titre accrocheur 2]
3. [titre accrocheur 3]

{scene_structure}
""".strip()

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )

        data = response.json()

        if response.status_code != 200:
            message = "Claude API a échoué"
            if isinstance(data, dict):
                err = data.get("error")
                if isinstance(err, dict):
                    message = err.get("message", message)
            return {"error": message, "details": data}

        content_blocks = data.get("content", [])
        if not content_blocks:
            return {"error": "Claude n'a pas retourné de contenu", "details": data}

        text = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text += block.get("text", "") + "\n"

        text = text.strip()
        if not text:
            return {"error": "Claude a retourné un contenu vide", "details": data}

        lines = text.split("\n")
        titles = []
        scene_keys = ["HOOK", "CONTEXT", "PROBLEM", "AGITATION", "SOLUTION", "PROOF", "BENEFIT", "CTA"]
        scenes = {key: "" for key in scene_keys}

        for line in lines:
            clean = line.strip()
            if clean.startswith("1."):
                titles.append(clean[2:].strip())
            elif clean.startswith("2."):
                titles.append(clean[2:].strip())
            elif clean.startswith("3."):
                titles.append(clean[2:].strip())
            else:
                for key in scene_keys:
                    if clean.upper().startswith(f"{key}:"):
                        scenes[key] = clean.split(":", 1)[1].strip()
                        break

        script_parts = [scenes[key] for key in scene_keys if scenes[key]]
        script = "\n\n".join(script_parts)

        def extract_keywords(scene_text: str, fallback: str, scene_key: str = "") -> str:
            if not scene_text:
                return fallback
            words = scene_text.split()
            stop_words = {
                "le","la","les","un","une","des","de","du","et","en","que","qui",
                "pour","par","sur","dans","avec","est","sont","mais","donc","car",
                "tout","tous","toute","cette","cela","plus","très","bien","aussi",
                "comme","même","fait","peut","faut","doit","avoir","être","faire",
                "the","a","an","is","are","you","your","to","of","in","and","or",
                "it","this","that","we","our","my","not","can","will","all","if",
                "have","has","been","with","they","their","from","but","when","how"
            }
            keywords = [w.strip(".,!?;:") for w in words
                       if len(w.strip(".,!?;:")) > 4
                       and w.lower().strip(".,!?;:") not in stop_words]
            if keywords:
                query = " ".join(keywords[:2]) + " " + fallback
                return query.strip()
            return fallback

        scene_queries = []
        for key in scene_keys:
            if scenes[key]:
                query = extract_keywords(scenes[key], niche, key)
                scene_queries.append(query)

        while len(scene_queries) < 8:
            scene_queries.append(niche)

        async def fetch_pexels_videos(client, query: str, count: int = 2) -> list:
            results = []
            try:
                resp = await client.get(
                    "https://api.pexels.com/videos/search",
                    params={"query": query, "per_page": count + 4, "orientation": "portrait"},
                    headers={"Authorization": PEXELS_API_KEY},
                )
                data = resp.json()
                videos = data.get("videos", [])
                for video in videos:
                    if len(results) >= count:
                        break
                    files = video.get("video_files", [])
                    hd_files = [f for f in files if f.get("quality") == "hd"]
                    if hd_files:
                        results.append(hd_files[0].get("link", ""))
                    elif files:
                        results.append(files[0].get("link", ""))
            except Exception:
                pass
            return results

        async def fetch_pixabay_videos(client, query: str, count: int = 2) -> list:
            results = []
            try:
                resp = await client.get(
                    "https://pixabay.com/api/videos/",
                    params={
                        "key": PIXABAY_API_KEY,
                        "q": query,
                        "per_page": count + 4,
                        "video_type": "film",
                    },
                )
                data = resp.json()
                hits = data.get("hits", [])
                for hit in hits:
                    if len(results) >= count:
                        break
                    videos = hit.get("videos", {})
                    for quality in ["medium", "small", "large"]:
                        v = videos.get(quality, {})
                        url = v.get("url", "")
                        if url:
                            results.append(url)
                            break
            except Exception:
                pass
            return results

        video_urls = [""] * 32

        if PEXELS_API_KEY or PIXABAY_API_KEY:
            try:
                async with httpx.AsyncClient(timeout=45) as client:
                    for i, query in enumerate(scene_queries[:8]):
                        slot = i * 5
                        collected = []

                        if PEXELS_API_KEY:
                            collected += await fetch_pexels_videos(client, query, 5)

                        if PIXABAY_API_KEY and len(collected) < 5:
                            collected += await fetch_pixabay_videos(client, query, 5 - len(collected))

                        if PEXELS_API_KEY and len(collected) < 5 and query != niche:
                            collected += await fetch_pexels_videos(client, niche, 5 - len(collected))

                        for j in range(5):
                            if j < len(collected):
                                video_urls[slot + j] = collected[j]
                            elif collected:
                                video_urls[slot + j] = collected[j % len(collected)]
            except Exception:
                pass

        scene_list = [scenes[key] for key in scene_keys if scenes[key]]

        # ── WAN 2.2 : prompt court et visuel (pas le script complet) ─────────
        # WAN échoue avec les longs textes — prompt court uniquement
        wan_video_url = ""
        if WAN_API_URL:
            try:
                short_prompt = niche if niche else "funny talking character"
                wan_video_url = await generate_wan_video(short_prompt)
            except Exception:
                wan_video_url = ""
        # ───────────────────────────────────────────────────────────────────

        return {
            "titles": titles,
            "script": script,
            "wan_video": wan_video_url,
            "scene1": scene_list[0] if len(scene_list) > 0 else "",
            "scene2": scene_list[1] if len(scene_list) > 1 else "",
            "scene3": scene_list[2] if len(scene_list) > 2 else "",
            "scene4": scene_list[3] if len(scene_list) > 3 else "",
            "scene5": scene_list[4] if len(scene_list) > 4 else "",
            "scene6": scene_list[5] if len(scene_list) > 5 else "",
            "scene7": scene_list[6] if len(scene_list) > 6 else "",
            "scene8": scene_list[7] if len(scene_list) > 7 else "",
            "nb_scenes": len(scene_list),
            "duration": duration,
            "video_url":  video_urls[0],
            "video_url2":  video_urls[1],
            "video_url3":  video_urls[2],
            "video_url4":  video_urls[3],
            "video_url5":  video_urls[4],
            "video_url6":  video_urls[5],
            "video_url7":  video_urls[6],
            "video_url8":  video_urls[7],
            "video_url9":  video_urls[8],
            "video_url10": video_urls[9],
            "video_url11": video_urls[10],
            "video_url12": video_urls[11],
            "video_url13": video_urls[12],
            "video_url14": video_urls[13],
            "video_url15": video_urls[14],
            "video_url16": video_urls[15],
            "video_url17": video_urls[16],
            "video_url18": video_urls[17],
            "video_url19": video_urls[18],
            "video_url20": video_urls[19],
            "video_url21": video_urls[20],
            "video_url22": video_urls[21],
            "video_url23": video_urls[22],
            "video_url24": video_urls[23],
            "video_url25": video_urls[24],
            "video_url26": video_urls[25],
            "video_url27": video_urls[26],
            "video_url28": video_urls[27],
            "video_url29": video_urls[28],
            "video_url30": video_urls[29],
            "video_url31": video_urls[30],
            "video_url32": video_urls[31],
            "raw_claude_text": text
        }

    except Exception as e:
        return {"error": f"Erreur Claude: {str(e)}"}


def build_ssml_with_marks(text: str) -> tuple:
    words = text.strip().split()
    parts = []
    for i, word in enumerate(words):
        parts.append(f'<mark name="w{i}"/>{word}')
    ssml = "<speak>" + " ".join(parts) + "</speak>"
    return ssml, words


def build_srt_from_timepoints(words: list, timepoints: list, out_path: str, words_per_block: int = 5):
    mark_times = {}
    for tp in timepoints:
        mark_name = tp.get("markName", "")
        time_seconds = tp.get("timeSeconds", 0.0)
        if mark_name.startswith("w"):
            try:
                idx = int(mark_name[1:])
                mark_times[idx] = float(time_seconds)
            except ValueError:
                pass

    if not mark_times or not words:
        return False

    last_idx = max(mark_times.keys())
    if len(mark_times) > 1:
        avg_word_dur = mark_times[last_idx] / last_idx if last_idx > 0 else 0.35
    else:
        avg_word_dur = 0.35
    estimated_end = mark_times.get(last_idx, 0) + avg_word_dur * words_per_block

    entries = []
    entry_idx = 1

    for j in range(0, len(words), words_per_block):
        block_words = words[j:j + words_per_block]
        block_text = " ".join(block_words)
        start = mark_times.get(j, None)
        if start is None:
            continue
        next_j = j + words_per_block
        if next_j < len(words) and next_j in mark_times:
            end = mark_times[next_j] - 0.05
        else:
            end = estimated_end - 0.05
        end = max(start + 0.1, end)
        entries.append(
            f"{entry_idx}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{block_text}\n"
        )
        entry_idx += 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries))

    return True


class ScanRequest(BaseModel):
    keyword: str = "money"
    platform: str = "TikTok"
    language: str = "en"


@app.post("/scan-trends")
async def scan_trends(req: ScanRequest):
    if not ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY manquante"}

    lang_label = {"fr": "French", "es": "Spanish", "pt": "Portuguese"}.get(req.language, "English")

    prompt = f"""You are a viral video trend analyst specializing in short-form content for {req.platform}.

Analyze viral trends for the keyword/niche: "{req.keyword}"
Target language: {lang_label}

Return ONLY a JSON array with exactly 5 viral video ideas. No markdown, no explanation, just valid JSON.

Each object must have exactly these fields:
- title: string (catchy viral video title, max 60 chars)
- niche: string (specific niche, max 30 chars)
- platform: string (use "{req.platform}")
- viralScore: number (0-100, based on trend potential)
- bestDuration: string (e.g. "30s", "45s", "60s")
- targetAudience: string (e.g. "25-35 entrepreneurs")
- whyViral: string (1 sentence explaining viral potential)
- hookIdea: string (1 powerful opening hook sentence)
- hashtags: string (5 relevant hashtags separated by spaces)

Focus on what is currently trending and has high viral potential. Be specific and creative."""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 1500,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        if response.status_code != 200:
            err = response.json() if "application/json" in response.headers.get("content-type","") else {}
            return {"error": f"Claude API erreur {response.status_code}", "details": err}
        data = response.json()
        if not data.get("content"):
            return {"error": "Claude n'a retourné aucun contenu"}
        raw = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                raw += block.get("text", "")
        clean = raw.replace("```json", "").replace("```", "").strip()
        if not clean:
            return {"error": "Claude a retourné un contenu vide"}
        try:
            results = json.loads(clean)
        except json.JSONDecodeError as je:
            return {"error": f"Réponse Claude non parseable: {str(je)}", "raw": clean[:200]}
        return {"success": True, "results": results}
    except Exception as e:
        return {"error": f"Erreur scan: {str(e)}"}


# GOOGLE TTS (CONSERVÉ POUR COMPATIBILITÉ)
@app.post("/generate-audio")
async def generate_audio(req: TTSRequest):
    if not GOOGLE_TTS_API_KEY:
        return {"error": "GOOGLE_TTS_API_KEY manquante"}
    if not PUBLIC_BASE_URL:
        return {"error": "PUBLIC_BASE_URL manquante"}
    if not req.text.strip():
        return {"error": "Le texte est vide"}

    google_url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_TTS_API_KEY}"

    voice_name = req.voiceName or ""
    supports_timepoints = any(v in voice_name for v in ["Neural2", "Wavenet", "Standard"])

    words = req.text.strip().split()

    if supports_timepoints:
        def escape_xml(text: str) -> str:
            return (text
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                    .replace("'", "&apos;"))

        ssml_parts = [f'<mark name="w{i}"/>{escape_xml(w)}' for i, w in enumerate(words)]
        ssml = "<speak>" + " ".join(ssml_parts) + "</speak>"
        payload = {
            "input": {"ssml": ssml},
            "voice": {"languageCode": req.languageCode, "name": req.voiceName},
            "audioConfig": {"audioEncoding": "MP3", "speakingRate": req.speakingRate},
            "enableTimePointing": ["SSML_MARK"]
        }
    else:
        payload = {
            "input": {"text": req.text},
            "voice": {"languageCode": req.languageCode, "name": req.voiceName},
            "audioConfig": {"audioEncoding": "MP3", "speakingRate": req.speakingRate},
        }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(google_url, json=payload)
        data = response.json()

        if response.status_code != 200 and supports_timepoints:
            payload_fallback = {
                "input": {"text": req.text},
                "voice": {"languageCode": req.languageCode, "name": req.voiceName},
                "audioConfig": {"audioEncoding": "MP3", "speakingRate": req.speakingRate},
            }
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(google_url, json=payload_fallback)
            data = response.json()
            supports_timepoints = False

        if response.status_code != 200:
            return {"error": "Google TTS a échoué", "details": data}
        audio_content = data.get("audioContent")
        if not audio_content:
            return {"error": "Aucun audio retourné par Google"}

        filename = f"{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(audio_content))
        audio_url = f"{PUBLIC_BASE_URL}/audio/{filename}"

        timepoints = data.get("timepoints", []) if supports_timepoints else []

        sync_filename = filename.replace(".mp3", "_sync.json")
        sync_filepath = os.path.join(AUDIO_DIR, sync_filename)
        with open(sync_filepath, "w", encoding="utf-8") as f:
            json.dump({"words": words, "timepoints": timepoints}, f)

        sync_url = f"{PUBLIC_BASE_URL}/audio/{sync_filename}"

        return {
            "success": True,
            "audio_url": audio_url,
            "sync_url": sync_url,
            "filename": filename,
            "timepoints_count": len(timepoints)
        }
    except Exception as e:
        return {"error": f"Erreur TTS: {str(e)}"}


# FFMPEG: CRÉER LA VIDÉO FINALE
async def _process_video(job_id: str, req: VideoRequest):
    job_dir = None
    try:
        if not PUBLIC_BASE_URL:
            VIDEO_JOBS[job_id] = {"status": "failed", "error": "PUBLIC_BASE_URL manquante"}
            return

        if not ffmpeg_exists():
            VIDEO_JOBS[job_id] = {"status": "failed", "error": "FFmpeg non installé sur le serveur"}
            return

        chosen_duration = req.duration if req.duration in [30, 45, 60] else 30

        if chosen_duration == 60:
            nb_scenes = 8
        elif chosen_duration == 45:
            nb_scenes = 6
        else:
            nb_scenes = 4

        all_video_urls = [
            (req.video_url or "").strip(),
            (req.video_url2 or "").strip(),
            (req.video_url3 or "").strip(),
            (req.video_url4 or "").strip(),
            (req.video_url5 or "").strip(),
            (req.video_url6 or "").strip(),
            (req.video_url7 or "").strip(),
            (req.video_url8 or "").strip(),
            (req.video_url9 or "").strip(),
            (req.video_url10 or "").strip(),
            (req.video_url11 or "").strip(),
            (req.video_url12 or "").strip(),
            (req.video_url13 or "").strip(),
            (req.video_url14 or "").strip(),
            (req.video_url15 or "").strip(),
            (req.video_url16 or "").strip(),
            (req.video_url17 or "").strip(),
            (req.video_url18 or "").strip(),
            (req.video_url19 or "").strip(),
            (req.video_url20 or "").strip(),
            (req.video_url21 or "").strip(),
            (req.video_url22 or "").strip(),
            (req.video_url23 or "").strip(),
            (req.video_url24 or "").strip(),
            (req.video_url25 or "").strip(),
            (req.video_url26 or "").strip(),
            (req.video_url27 or "").strip(),
            (req.video_url28 or "").strip(),
            (req.video_url29 or "").strip(),
            (req.video_url30 or "").strip(),
            (req.video_url31 or "").strip(),
            (req.video_url32 or "").strip(),
            (req.video_url33 or "").strip(),
            (req.video_url34 or "").strip(),
            (req.video_url35 or "").strip(),
            (req.video_url36 or "").strip(),
            (req.video_url37 or "").strip(),
            (req.video_url38 or "").strip(),
            (req.video_url39 or "").strip(),
            (req.video_url40 or "").strip(),
        ]

        all_subtitle_texts = [
            (req.text1 or "").strip()[:200],
            (req.text2 or "").strip()[:200],
            (req.text3 or "").strip()[:200],
            (req.text4 or "").strip()[:200],
            (req.text5 or "").strip()[:200],
            (req.text6 or "").strip()[:200],
            (req.text7 or "").strip()[:200],
            (req.text8 or "").strip()[:200],
        ]

        if req.wan_video:
            valid_video_urls_raw = [req.wan_video]
        else:
            valid_video_urls_raw = [u for u in all_video_urls if u]
        if not valid_video_urls_raw:
            VIDEO_JOBS[job_id] = {"status": "failed", "error": "Aucune vidéo fournie"}
            return

        # MODE WAN : 1 clip par scene, MODE PEXELS : 5 clips par scene
        if req.wan_video:
            CLIPS_PER_SCENE = 1
            nb_clips_total = nb_scenes * CLIPS_PER_SCENE
            clip_urls = [req.wan_video] * nb_scenes
            subtitle_texts = [
                all_subtitle_texts[i] if i < len(all_subtitle_texts) else ""
                for i in range(nb_scenes)
            ]
        else:
            CLIPS_PER_SCENE = 5
            nb_clips_total = nb_scenes * CLIPS_PER_SCENE
            clip_urls = []
            subtitle_texts = []
            for i in range(nb_scenes):
                scene_text = all_subtitle_texts[i] if i < len(all_subtitle_texts) else ""
                collected = []
                for slot_offset in range(len(all_video_urls)):
                    slot = i * CLIPS_PER_SCENE + slot_offset
                    if slot < len(all_video_urls) and all_video_urls[slot]:
                        collected.append(all_video_urls[slot])
                    if len(collected) >= CLIPS_PER_SCENE:
                        break
                while len(collected) < CLIPS_PER_SCENE and valid_video_urls_raw:
                    collected.append(valid_video_urls_raw[
                        (i * CLIPS_PER_SCENE + len(collected)) % len(valid_video_urls_raw)
                    ])
                clip_urls.extend(collected[:CLIPS_PER_SCENE])
                subtitle_texts.append(scene_text)
                for _ in range(CLIPS_PER_SCENE - 1):
                    subtitle_texts.append("")

        total_duration = chosen_duration

        job_dir = os.path.join(WORK_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        voice_url = (req.audio_url or "").strip()
        music_url = (req.music_url or "").strip()

        voice_path = None
        real_total_duration = float(total_duration)

        if voice_url:
            voice_path = os.path.join(job_dir, "voice.mp3")
            await download_audio_file(voice_url, voice_path)
            measured = get_audio_duration(voice_path)
            if measured > 1.0:
                real_total_duration = measured

        clip_duration = real_total_duration / nb_clips_total

        raw_paths = [os.path.join(job_dir, f"raw_{i}.mp4")
                     for i in range(len(clip_urls))]

        await asyncio.gather(*[
            download_file(url, path)
            for url, path in zip(clip_urls, raw_paths)
            if url
        ])

        from concurrent.futures import ThreadPoolExecutor

        norm_paths = [os.path.join(job_dir, f"seg_{i}.mp4")
                      for i in range(len(raw_paths))]

        def normalize_clip(args):
            src, dst = args
            if not os.path.exists(src):
                return
            run_cmd([
                "ffmpeg", "-y", "-i", src,
                "-t", str(clip_duration),
                "-vf", "scale=405:720:force_original_aspect_ratio=increase,"
                       "crop=405:720,fps=25,format=yuv420p",
                "-an", "-c:v", "libx264",
                "-preset", "ultrafast", "-crf", "28", "-r", "25",
                dst
            ])

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(normalize_clip, zip(raw_paths, norm_paths)))

        norm_paths = [p for p in norm_paths if os.path.exists(p)]
        if not norm_paths:
            raise RuntimeError("Aucun clip normalisé produit")

        concat_path = os.path.join(job_dir, "concat.txt")
        with open(concat_path, "w") as f:
            for p in norm_paths:
                f.write(f"file '{os.path.abspath(p)}'\n")

        stitched_raw = os.path.join(job_dir, "stitched_raw.mp4")
        await async_run_cmd([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_path,
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "28", "-r", "25", "-pix_fmt", "yuv420p", "-an",
            stitched_raw
        ])

        stitched_dur = get_audio_duration(stitched_raw)
        stitched_path = os.path.join(job_dir, "stitched.mp4")

        if stitched_dur < real_total_duration - 0.5:
            await async_run_cmd([
                "ffmpeg", "-y", "-stream_loop", "-1", "-i", stitched_raw,
                "-t", str(real_total_duration),
                "-c:v", "libx264", "-preset", "ultrafast",
                "-crf", "28", "-r", "25", "-pix_fmt", "yuv420p", "-an",
                stitched_path
            ])
        else:
            await async_run_cmd([
                "ffmpeg", "-y", "-i", stitched_raw,
                "-t", str(real_total_duration),
                "-c:v", "copy", "-an", stitched_path
            ])

        real_segment_duration = clip_duration

        final_audio_path = None

        if voice_url and voice_path:
            if music_url:
                music_path = os.path.join(job_dir, "music.mp3")
                await download_audio_file(music_url, music_path)
                mixed_path = os.path.join(job_dir, "mixed.m4a")
                await async_run_cmd([
                    "ffmpeg", "-y",
                    "-fflags", "+genpts",
                    "-i", voice_path,
                    "-stream_loop", "-1", "-i", music_path,
                    "-filter_complex",
                    f"[1:a]volume=0.12,atrim=0:{real_total_duration}[bg];"
                    f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                    "-map", "[aout]",
                    "-c:a", "aac", "-b:a", "192k",
                    mixed_path
                ])
                final_audio_path = mixed_path
            else:
                final_audio_path = voice_path

        srt_path = os.path.join(job_dir, "subtitles.srt")
        srt_built = False

        sync_url = (req.sync_url or "").strip()
        if sync_url:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    sync_resp = await client.get(sync_url)
                sync_data = sync_resp.json()
                words_list = sync_data.get("words", [])
                timepoints_list = sync_data.get("timepoints", [])
                if words_list and timepoints_list:
                    srt_built = build_srt_from_timepoints(
                        words_list, timepoints_list, srt_path, words_per_block=5
                    )
            except Exception:
                srt_built = False

        if not srt_built:
            nb_subtitles = len([t for t in subtitle_texts if t.strip()])
            srt_segment_duration = (
                real_total_duration / nb_subtitles if nb_subtitles > 0
                else real_segment_duration
            )
            write_srt(subtitle_texts, srt_segment_duration, srt_path)

        output_filename = f"{job_id}.mp4"
        output_path = os.path.join(VIDEO_DIR, output_filename)

        srt_escaped = escape_srt_path(os.path.abspath(srt_path))
        subtitle_filter = (
            f"subtitles='{srt_escaped}':"
            "force_style='Alignment=2,MarginV=70,"
            "PlayResX=405,PlayResY=720,"
            "FontName=Arial,FontSize=24,Bold=1,"
            "PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,"
            "BorderStyle=3,Outline=2,Shadow=0,"
            "BackColour=&H99000000'"
        )

        if final_audio_path:
            await async_run_cmd([
                "ffmpeg", "-y",
                "-i", stitched_path,
                "-i", final_audio_path,
                "-vf", subtitle_filter,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-r", "25",
                "-c:a", "aac", "-b:a", "128k",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ])
        else:
            await async_run_cmd([
                "ffmpeg", "-y",
                "-i", stitched_path,
                "-vf", subtitle_filter,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-r", "25",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ])

        shutil.rmtree(job_dir, ignore_errors=True)

        video_url = f"{PUBLIC_BASE_URL}/video/{output_filename}"
        VIDEO_JOBS[job_id] = {"status": "done", "video_url": video_url}

    except httpx.HTTPError as e:
        shutil.rmtree(job_dir, ignore_errors=True) if job_dir else None
        VIDEO_JOBS[job_id] = {"status": "failed", "error": f"Erreur téléchargement: {str(e)}"}
    except RuntimeError as e:
        shutil.rmtree(job_dir, ignore_errors=True) if job_dir else None
        VIDEO_JOBS[job_id] = {"status": "failed", "error": f"Erreur FFmpeg: {str(e)}"}
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True) if job_dir else None
        VIDEO_JOBS[job_id] = {"status": "failed", "error": f"Erreur create-video: {str(e)}"}


@app.post("/create-video")
async def create_video(req: VideoRequest):
    if not PUBLIC_BASE_URL:
        return JSONResponse(status_code=400, content={"error": "PUBLIC_BASE_URL manquante"})
    if not ffmpeg_exists():
        return JSONResponse(status_code=500, content={"error": "FFmpeg non installé"})

    job_id = uuid.uuid4().hex
    VIDEO_JOBS[job_id] = {"status": "processing"}

    asyncio.create_task(_process_video(job_id, req))

    return JSONResponse(status_code=200, content={
        "success": True,
        "job_id": job_id,
        "message": "Rendu démarré. Vérifiez /video-status/{job_id}"
    })


class WanGenerateRequest(BaseModel):
    prompt: str
    resolution: str = "480p"
    num_frames: int = 81
    num_inference_steps: int = 20


WAN_SIZE_MAP = {
    "480p": "832*480",
    "720p": "1280*720",
    "1080p": "1280*720",
    "832*480": "832*480",
    "1280*720": "1280*720",
    "480*832": "480*832",
    "720*1280": "720*1280",
}


async def _process_wan_job(job_id: str, prompt: str, wan_size: str, sample_steps: int):
    base = RUNPOD_API_URL.rstrip("/")
    try:
        # Step 1 — submit job to RunPod, get runpod_job_id instantly
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(
                f"{base}/wan/generate",
                data={"prompt": prompt, "size": wan_size, "sample_steps": str(sample_steps)},
            )
        if res.status_code != 200:
            try:
                detail = res.json()
            except Exception:
                detail = res.text[:500]
            WAN_JOBS[job_id] = {"status": "error", "detail": f"RunPod submit erreur {res.status_code}: {detail}"}
            return
        runpod_job_id = res.json().get("job_id")
        if not runpod_job_id:
            WAN_JOBS[job_id] = {"status": "error", "detail": "RunPod n'a pas retourné de job_id"}
            return

        # Step 2 — poll RunPod every 10 seconds, max 180 attempts (30 min)
        for attempt in range(180):
            await asyncio.sleep(10)
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    sr = await client.get(f"{base}/wan/status/{runpod_job_id}")
                sd = sr.json()
            except Exception:
                continue  # transient network error, retry

            if sd.get("status") == "done":
                break
            if sd.get("status") == "error":
                WAN_JOBS[job_id] = {"status": "error", "detail": sd.get("detail", "Erreur RunPod inconnue")}
                return
        else:
            WAN_JOBS[job_id] = {"status": "error", "detail": "Timeout: RunPod n'a pas terminé en 30 minutes"}
            return

        # Step 3 — download finished video from RunPod
        async with httpx.AsyncClient(timeout=300) as client:
            video_resp = await client.get(f"{base}/wan/video/{runpod_job_id}")
        if video_resp.status_code != 200:
            WAN_JOBS[job_id] = {"status": "error", "detail": f"Impossible de télécharger la vidéo ({video_resp.status_code})"}
            return

        video_path = f"/tmp/{job_id}.mp4"
        with open(video_path, "wb") as f:
            f.write(video_resp.content)
        WAN_JOBS[job_id] = {"status": "done", "video_path": video_path}

    except Exception as e:
        WAN_JOBS[job_id] = {"status": "error", "detail": str(e)}


@app.post("/wan/generate")
async def wan_generate(req: WanGenerateRequest):
    if not RUNPOD_API_URL:
        return JSONResponse(status_code=503, content={"error": "RUNPOD_API_URL non configurée"})
    if not req.prompt.strip():
        return JSONResponse(status_code=400, content={"error": "Le prompt est vide"})

    wan_size = WAN_SIZE_MAP.get(req.resolution, "832*480")
    job_id = uuid.uuid4().hex
    WAN_JOBS[job_id] = {"status": "processing"}

    asyncio.create_task(_process_wan_job(job_id, req.prompt, wan_size, req.num_inference_steps))

    return JSONResponse(status_code=200, content={
        "job_id": job_id,
        "status": "processing",
        "poll_url": f"/wan/status/{job_id}",
    })


@app.get("/wan/status/{job_id}")
async def wan_status(job_id: str):
    job = WAN_JOBS.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job introuvable"})
    if job["status"] == "done":
        return JSONResponse(status_code=200, content={
            "status": "done",
            "video_url": f"/wan/video/{job_id}",
        })
    if job["status"] == "error":
        return JSONResponse(status_code=200, content={
            "status": "error",
            "detail": job.get("detail", "Erreur inconnue"),
        })
    return JSONResponse(status_code=200, content={"status": "processing"})


@app.get("/wan/video/{job_id}")
async def wan_video(job_id: str):
    job = WAN_JOBS.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job introuvable"})
    if job["status"] != "done":
        return JSONResponse(status_code=425, content={"error": "Vidéo pas encore prête", "status": job["status"]})
    video_path = job.get("video_path", "")
    if not video_path or not os.path.exists(video_path):
        return JSONResponse(status_code=404, content={"error": "Fichier vidéo introuvable sur le serveur"})

    def iter_file():
        with open(video_path, "rb") as f:
            yield from iter(lambda: f.read(65536), b"")

    return StreamingResponse(
        iter_file(),
        media_type="video/mp4",
        headers={"Content-Disposition": f"inline; filename=wan_{job_id}.mp4"},
    )


class GenerateScriptRequest(BaseModel):
    niche: str
    video_type: str
    narrative_style: str
    duration_seconds: int = 60
    language: str = "fr"
    custom_script: str = ""
    user_plan: str = "free"


@app.post("/generate-script")
async def generate_script(req: GenerateScriptRequest):
    if not ANTHROPIC_API_KEY:
        return JSONResponse(status_code=503, content={"error": "ANTHROPIC_API_KEY manquante"})

    plan_key = req.user_plan.lower().strip()
    allowed_niches = NICHES_BY_PLAN.get(plan_key)
    if allowed_niches is None:
        return JSONResponse(
            status_code=400,
            content={"error": f"Plan inconnu: '{req.user_plan}'. Valeurs acceptées: free, starter, pro, premium"},
        )

    niche_normalized = req.niche.strip()
    if niche_normalized not in allowed_niches:
        return JSONResponse(
            status_code=403,
            content={
                "error": f"La niche '{niche_normalized}' n'est pas disponible pour le plan '{plan_key}'.",
                "allowed_niches": allowed_niches,
            },
        )

    if req.custom_script.strip():
        return {
            "script": req.custom_script.strip(),
            "source": "custom",
            "niche": niche_normalized,
            "narrative_style": req.narrative_style,
        }

    lang_map = {"fr": "français", "en": "anglais", "es": "espagnol", "pt": "portugais"}
    lang_label = lang_map.get(req.language.lower(), req.language)

    prompt = f"""Tu es un expert en création de contenu vidéo viral pour les réseaux sociaux.

Génère un script vidéo de {req.duration_seconds} secondes pour les paramètres suivants :
- Niche : {niche_normalized}
- Type de vidéo : {req.video_type}
- Style narratif : {req.narrative_style}
- Langue : {lang_label}

RÈGLES :
- Le script doit correspondre exactement au style narratif demandé
- Adapte le ton et le rythme au type de vidéo
- Durée cible : {req.duration_seconds} secondes à l'oral (environ {req.duration_seconds * 2} mots)
- Écris uniquement le texte à dire, sans didascalies ni indications de mise en scène
- Pas de titres de section, juste le texte continu à narrer
- Langue : {lang_label} uniquement

Retourne uniquement le script, sans introduction ni commentaire."""

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 1500,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )

        data = response.json()

        if response.status_code != 200:
            err = data.get("error", {})
            return JSONResponse(
                status_code=502,
                content={"error": err.get("message", "Claude API a échoué"), "details": data},
            )

        content_blocks = data.get("content", [])
        script_text = ""
        for block in content_blocks:
            if block.get("type") == "text":
                script_text += block.get("text", "")

        script_text = script_text.strip()
        if not script_text:
            return JSONResponse(status_code=502, content={"error": "Claude a retourné un script vide"})

        return {
            "script": script_text,
            "source": "generated",
            "niche": niche_normalized,
            "narrative_style": req.narrative_style,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Erreur generate-script: {str(e)}"})


VOICE_CATALOG = [
    {
        "voice_id": "a5474df3-4f8e-4e4c-b5e3-d70a7c1c7dc1",
        "voice_name": "Lucas",
        "language": "fr",
        "gender": "male",
        "styles": ["Dynamique et rapide", "Autoritaire et confiant"],
        "preview_url": None,
    },
    {
        "voice_id": "b8c32ef1-7a21-4d3e-9f12-2e8d5b6a4c90",
        "voice_name": "Camille",
        "language": "fr",
        "gender": "female",
        "styles": ["Doux et chaleureux", "Lent et posé"],
        "preview_url": None,
    },
    {
        "voice_id": "c1d45fa2-3b67-4e8f-a034-5c9e7d2b1f83",
        "voice_name": "Théo",
        "language": "fr",
        "gender": "male",
        "styles": ["Humoristique et léger", "Dynamique et rapide"],
        "preview_url": None,
    },
    {
        "voice_id": "d2e56gb3-4c78-5f90-b145-6d0f8e3c2a94",
        "voice_name": "Sophie",
        "language": "fr",
        "gender": "female",
        "styles": ["Éducatif et clair", "Autoritaire et confiant"],
        "preview_url": None,
    },
    {
        "voice_id": "e3f67hc4-5d89-6a01-c256-7e1a9f4d3b05",
        "voice_name": "Nathan",
        "language": "fr",
        "gender": "male",
        "styles": ["Lent et posé", "Éducatif et clair"],
        "preview_url": None,
    },
    {
        "voice_id": "f4a78id5-6e90-7b12-d367-8f2b0a5e4c16",
        "voice_name": "Emma",
        "language": "fr",
        "gender": "female",
        "styles": ["Humoristique et léger", "Doux et chaleureux"],
        "preview_url": None,
    },
    {
        "voice_id": "a1b23cd4-1234-5678-9abc-def012345678",
        "voice_name": "James",
        "language": "en",
        "gender": "male",
        "styles": ["Autoritaire et confiant", "Éducatif et clair"],
        "preview_url": None,
    },
    {
        "voice_id": "b2c34de5-2345-6789-abcd-ef0123456789",
        "voice_name": "Aria",
        "language": "en",
        "gender": "female",
        "styles": ["Doux et chaleureux", "Dynamique et rapide"],
        "preview_url": None,
    },
]


MUSIC_CATALOG = [
    {
        "music_id": "music-epic-001",
        "music_name": "Rise to Glory",
        "music_style": "Épique et motivant",
        "niches": ["motivation", "sport", "finance"],
        "duration": 60,
        "url": "",
    },
    {
        "music_id": "music-epic-002",
        "music_name": "Champion's Path",
        "music_style": "Épique et motivant",
        "niches": ["motivation", "sport", "challenge"],
        "duration": 45,
        "url": "",
    },
    {
        "music_id": "music-soft-001",
        "music_name": "Morning Calm",
        "music_style": "Doux et apaisant",
        "niches": ["santé", "méditation", "religion"],
        "duration": 60,
        "url": "",
    },
    {
        "music_id": "music-soft-002",
        "music_name": "Inner Peace",
        "music_style": "Doux et apaisant",
        "niches": ["mindfulness", "bien-être", "parentalité"],
        "duration": 45,
        "url": "",
    },
    {
        "music_id": "music-energy-001",
        "music_name": "Neon Rush",
        "music_style": "Rythmé et énergique",
        "niches": ["gaming", "challenge", "influenceur"],
        "duration": 60,
        "url": "",
    },
    {
        "music_id": "music-energy-002",
        "music_name": "Beat Drop",
        "music_style": "Rythmé et énergique",
        "niches": ["gaming", "sport", "mode"],
        "duration": 30,
        "url": "",
    },
    {
        "music_id": "music-mystery-001",
        "music_name": "Dark Discovery",
        "music_style": "Mystérieux et intrigant",
        "niches": ["révélation", "technologie", "cryptomonnaie"],
        "duration": 60,
        "url": "",
    },
    {
        "music_id": "music-neutral-001",
        "music_name": "Clear Horizon",
        "music_style": "Neutre et professionnel",
        "niches": ["éducation", "langues", "IA", "business"],
        "duration": 60,
        "url": "",
    },
    {
        "music_id": "music-neutral-002",
        "music_name": "Steady Focus",
        "music_style": "Neutre et professionnel",
        "niches": ["tutoriel", "finance", "technologie"],
        "duration": 45,
        "url": "",
    },
    {
        "music_id": "music-happy-001",
        "music_name": "Sunny Day",
        "music_style": "Joyeux et léger",
        "niches": ["blagues", "cuisine", "voyage"],
        "duration": 60,
        "url": "",
    },
    {
        "music_id": "music-happy-002",
        "music_name": "Good Vibes",
        "music_style": "Joyeux et léger",
        "niches": ["humour", "famille", "parentalité"],
        "duration": 45,
        "url": "",
    },
]


REFERENCE_VIDEOS = {
    "danses": [
        {"id": "dance-001", "name": "Hip-Hop Freestyle", "preview_url": "", "duration": 8, "path": ""},
        {"id": "dance-002", "name": "Salsa Basic", "preview_url": "", "duration": 10, "path": ""},
        {"id": "dance-003", "name": "Contemporary Flow", "preview_url": "", "duration": 12, "path": ""},
    ],
    "chants": [
        {"id": "chant-001", "name": "Gospel Praise", "preview_url": "", "duration": 15, "path": ""},
        {"id": "chant-002", "name": "Pop Chorus", "preview_url": "", "duration": 10, "path": ""},
        {"id": "chant-003", "name": "Acoustic Ballad", "preview_url": "", "duration": 20, "path": ""},
    ],
    "discours": [
        {"id": "discours-001", "name": "Motivational Speech", "preview_url": "", "duration": 30, "path": ""},
        {"id": "discours-002", "name": "Product Pitch", "preview_url": "", "duration": 25, "path": ""},
        {"id": "discours-003", "name": "Story Narration", "preview_url": "", "duration": 20, "path": ""},
    ],
    "gestes": [
        {"id": "geste-001", "name": "Pointing & Explaining", "preview_url": "", "duration": 6, "path": ""},
        {"id": "geste-002", "name": "Hands Open Welcome", "preview_url": "", "duration": 5, "path": ""},
        {"id": "geste-003", "name": "Count on Fingers", "preview_url": "", "duration": 7, "path": ""},
    ],
}

# Flat lookup map for reference video path resolution
_REF_VIDEO_FLAT = {
    v["id"]: v
    for videos in REFERENCE_VIDEOS.values()
    for v in videos
}


class SelectVoiceRequest(BaseModel):
    voice_style: str
    language: str = "fr"
    gender: str = ""
    preview: bool = False


@app.post("/select-voice")
async def select_voice(req: SelectVoiceRequest):
    lang = req.language.lower().strip()
    gender = req.gender.lower().strip()
    style = req.voice_style.strip()

    candidates = [v for v in VOICE_CATALOG if v["language"] == lang]
    if not candidates:
        candidates = VOICE_CATALOG[:]

    style_matches = [v for v in candidates if style in v["styles"]]
    if style_matches:
        candidates = style_matches

    if gender in ("male", "female"):
        gender_matches = [v for v in candidates if v["gender"] == gender]
        if gender_matches:
            candidates = gender_matches

    chosen = candidates[0]

    preview_url = None
    if req.preview and FISH_AUDIO_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"https://api.fish.audio/v1/model/{chosen['voice_id']}",
                    headers={"Authorization": f"Bearer {FISH_AUDIO_API_KEY}"},
                )
            if resp.status_code == 200:
                data = resp.json()
                samples = data.get("samples", [])
                if samples:
                    preview_url = samples[0].get("audio", None)
        except Exception:
            pass

    return {
        "voice_id": chosen["voice_id"],
        "voice_name": chosen["voice_name"],
        "voice_style": style,
        "preview_url": preview_url,
    }


@app.post("/clone-voice")
async def clone_voice(
    audio_file: UploadFile = File(...),
    voice_name: str = Form(...),
):
    if not FISH_AUDIO_API_KEY:
        return {
            "cloned_voice_id": "mock-cloned-voice-id-12345",
            "voice_name": voice_name,
            "status": "ready",
        }

    content_type = audio_file.content_type or "audio/mpeg"
    audio_bytes = await audio_file.read()

    if len(audio_bytes) < 10 * 1024:
        return JSONResponse(
            status_code=400,
            content={"error": "Le fichier audio doit faire au moins 10 secondes (fichier trop petit)"},
        )

    max_bytes = 5 * 60 * 128 * 1024 // 8
    if len(audio_bytes) > max_bytes:
        return JSONResponse(
            status_code=400,
            content={"error": "Le fichier audio ne doit pas dépasser 5 minutes"},
        )

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://api.fish.audio/v1/model",
                headers={"Authorization": f"Bearer {FISH_AUDIO_API_KEY}"},
                data={
                    "title": voice_name,
                    "type": "voice",
                    "train_mode": "fast",
                    "visibility": "private",
                },
                files={"voices": (audio_file.filename or "voice.mp3", audio_bytes, content_type)},
            )

        if response.status_code not in (200, 201):
            try:
                detail = response.json()
            except Exception:
                detail = response.text[:500]
            return JSONResponse(
                status_code=502,
                content={"error": f"Fish Audio cloning erreur {response.status_code}", "details": detail},
            )

        data = response.json()
        model_id = data.get("_id") or data.get("id", "")
        status = "ready" if data.get("state") == "done" else "processing"

        return {
            "cloned_voice_id": model_id,
            "voice_name": voice_name,
            "status": status,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Erreur clone-voice: {str(e)}"})


@app.get("/available-voices")
async def available_voices(style: str = ""):
    if not FISH_AUDIO_API_KEY:
        catalog = VOICE_CATALOG
        if style:
            catalog = [v for v in catalog if style in v["styles"]]
        return {
            "voices": [
                {
                    "voice_id": v["voice_id"],
                    "voice_name": v["voice_name"],
                    "language": v["language"],
                    "gender": v["gender"],
                    "styles": v["styles"],
                }
                for v in catalog
            ],
            "source": "mock",
        }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                "https://api.fish.audio/v1/model",
                headers={"Authorization": f"Bearer {FISH_AUDIO_API_KEY}"},
                params={"page_size": 50, "sort_by": "task_count", "type": "voice"},
            )

        if response.status_code != 200:
            return JSONResponse(
                status_code=502,
                content={"error": f"Fish Audio API erreur {response.status_code}"},
            )

        data = response.json()
        voices = []
        for item in data.get("items", []):
            voice_entry = {
                "voice_id": item.get("_id", ""),
                "voice_name": item.get("title", ""),
                "language": (item.get("languages") or ["unknown"])[0],
                "gender": item.get("gender", "neutral"),
                "styles": [],
                "tags": item.get("tags", []),
            }
            voices.append(voice_entry)

        if style:
            style_lower = style.lower()
            voices = [
                v for v in voices
                if style_lower in v["voice_name"].lower()
                or any(style_lower in t.lower() for t in v.get("tags", []))
            ]

        return {"voices": voices, "count": len(voices), "source": "fish_audio"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Erreur available-voices: {str(e)}"})


class SelectMusicRequest(BaseModel):
    music_style: str
    niche: str = ""
    duration_seconds: int = 60
    user_upload: bool = False


@app.post("/select-music")
async def select_music(req: SelectMusicRequest):
    if req.user_upload:
        return {
            "music_id": "",
            "music_name": "",
            "music_style": req.music_style,
            "duration": req.duration_seconds,
            "url": "",
        }

    style = req.music_style.strip()
    niche_lower = req.niche.lower().strip()

    candidates = [t for t in MUSIC_CATALOG if t["music_style"] == style]
    if not candidates:
        candidates = MUSIC_CATALOG[:]

    if niche_lower:
        niche_matches = [t for t in candidates if any(niche_lower in n for n in t["niches"])]
        if niche_matches:
            candidates = niche_matches

    best = min(
        candidates,
        key=lambda t: abs(t["duration"] - req.duration_seconds),
    )

    return {
        "music_id": best["music_id"],
        "music_name": best["music_name"],
        "music_style": style,
        "duration": best["duration"],
        "url": best["url"],
    }


@app.post("/upload-music")
async def upload_music(
    music_file: UploadFile = File(...),
    music_name: str = Form(""),
):
    allowed_types = {"audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/mp4", "audio/m4a", "audio/x-m4a"}
    allowed_extensions = {".mp3", ".wav", ".m4a"}

    filename = music_file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    content_type = music_file.content_type or ""

    if ext not in allowed_extensions and content_type not in allowed_types:
        return JSONResponse(
            status_code=400,
            content={"error": "Format non supporté. Utilisez mp3, wav ou m4a."},
        )

    audio_bytes = await music_file.read()

    max_bytes = 10 * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        return JSONResponse(
            status_code=400,
            content={"error": "Le fichier dépasse la limite de 10 MB"},
        )

    music_id = uuid.uuid4().hex
    safe_ext = ext if ext in allowed_extensions else ".mp3"
    saved_filename = f"music_{music_id}{safe_ext}"
    saved_path = os.path.join(MUSIC_DIR, saved_filename)

    with open(saved_path, "wb") as f:
        f.write(audio_bytes)

    duration = get_audio_duration(saved_path)
    final_name = music_name.strip() or os.path.splitext(filename)[0] or saved_filename
    file_url = f"{PUBLIC_BASE_URL}/music/{saved_filename}" if PUBLIC_BASE_URL else saved_path

    return {
        "music_id": music_id,
        "music_name": final_name,
        "duration": int(duration),
        "file_path": file_url,
    }


@app.get("/music/{filename}")
async def serve_music(filename: str):
    file_path = os.path.join(MUSIC_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Fichier music introuvable"})
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)


@app.get("/available-music")
async def available_music(style: str = ""):
    catalog = MUSIC_CATALOG
    if style.strip():
        style_lower = style.lower().strip()
        catalog = [
            t for t in MUSIC_CATALOG
            if style_lower in t["music_style"].lower()
        ]

    uploaded = []
    if PUBLIC_BASE_URL:
        try:
            for fname in os.listdir(MUSIC_DIR):
                if fname.startswith("music_") and os.path.splitext(fname)[1].lower() in {".mp3", ".wav", ".m4a"}:
                    fpath = os.path.join(MUSIC_DIR, fname)
                    uploaded.append({
                        "music_id": os.path.splitext(fname)[0].replace("music_", ""),
                        "music_name": fname,
                        "music_style": "Upload utilisateur",
                        "duration": int(get_audio_duration(fpath)),
                        "url": f"{PUBLIC_BASE_URL}/music/{fname}",
                    })
        except Exception:
            pass

    return {
        "catalog": catalog,
        "uploaded": uploaded,
        "total": len(catalog) + len(uploaded),
    }


@app.get("/voice-styles")
async def get_voice_styles():
    return {"voice_styles": VOICE_STYLES}


@app.get("/music-styles")
async def get_music_styles():
    return {"music_styles": MUSIC_STYLES}


@app.get("/suggested-styles")
async def get_suggested_styles(niche: str = ""):
    niche_lower = niche.lower().strip()

    suggested_voice = None
    for style, keywords in VOICE_STYLES.items():
        if any(k in niche_lower for k in keywords):
            suggested_voice = style
            break

    suggested_music = None
    for style, keywords in MUSIC_STYLES.items():
        if any(k in niche_lower for k in keywords):
            suggested_music = style
            break

    suggested_narrative = None
    narrative_map = {
        "Humoristique": ["blague", "humour", "entertainment", "gaming"],
        "Inspirant": ["motivation", "développement personnel", "sport", "fitness"],
        "Éducatif": ["ia", "technologie", "langues", "tutoriel", "enseignement"],
        "Choc & Révélation": ["révélation", "choc", "cryptomonnaie", "finance"],
        "Storytelling": ["amour", "relation", "parentalité", "voyage"],
        "Informatif": ["santé", "religion", "mindfulness"],
        "Conversationnel": ["cuisine", "mode", "beauté", "influenceur"],
        "Dramatique": ["business", "influenceur"],
    }
    for style, keywords in narrative_map.items():
        if any(k in niche_lower for k in keywords):
            suggested_narrative = style
            break

    return {
        "niche": niche,
        "suggested_voice_style": suggested_voice or "Éducatif et clair",
        "suggested_music_style": suggested_music or "Neutre et professionnel",
        "suggested_narrative_style": suggested_narrative or "Informatif",
    }


@app.get("/video-types")
async def get_video_types():
    return {"video_types": VIDEO_TYPES, "count": len(VIDEO_TYPES)}


@app.get("/narrative-styles")
async def get_narrative_styles():
    return {"narrative_styles": NARRATIVE_STYLES, "count": len(NARRATIVE_STYLES)}


@app.get("/niches")
async def get_niches(plan: str = "free"):
    key = plan.lower().strip()
    niches = NICHES_BY_PLAN.get(key)
    if niches is None:
        return JSONResponse(
            status_code=400,
            content={"error": f"Plan inconnu: '{plan}'. Valeurs acceptées: free, starter, pro, premium"},
        )
    return {"plan": key, "niches": niches, "count": len(niches)}


@app.get("/niches/all")
async def get_niches_all():
    result = {}
    seen = {}
    for plan, niches in NICHES_BY_PLAN.items():
        for niche in niches:
            if niche not in seen:
                seen[niche] = plan
    all_niches = [{"niche": niche, "plan": plan} for niche, plan in seen.items()]
    return {"plans": NICHES_BY_PLAN, "all_niches": all_niches, "total": len(all_niches)}


class QwenImageRequest(BaseModel):
    prompt: str
    style: str = "realistic"
    width: int = 1280
    height: int = 720
    negative_prompt: str = ""


@app.post("/qwen/generate-image")
async def qwen_generate_image(req: QwenImageRequest):
    if not QWEN_API_URL:
        return JSONResponse(status_code=503, content={"error": "Qwen service not configured"})
    if not req.prompt.strip():
        return JSONResponse(status_code=400, content={"error": "Le prompt est vide"})

    payload = {
        "prompt": req.prompt,
        "style": req.style,
        "width": req.width,
        "height": req.height,
    }
    if req.negative_prompt.strip():
        payload["negative_prompt"] = req.negative_prompt

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                QWEN_API_URL.rstrip("/"),
                json=payload,
                headers={"Content-Type": "application/json"},
            )

        if response.status_code != 200:
            try:
                detail = response.json()
            except Exception:
                detail = response.text[:500]
            return JSONResponse(
                status_code=response.status_code,
                content={"error": f"Qwen API erreur {response.status_code}", "details": detail},
            )

        data = response.json()
        return {
            "image_url": data.get("image_url", ""),
            "image_path": data.get("image_path", ""),
            "prompt": req.prompt,
        }

    except httpx.TimeoutException:
        return JSONResponse(status_code=504, content={"error": "Qwen generation timed out"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Erreur qwen/generate-image: {str(e)}"})


@app.post("/wan/animate")
async def wan_animate(
    mode: str = Form("animation"),
    resolution: str = Form("1280x720"),
    num_frames: int = Form(81),
    character_image_url: str = Form(""),
    reference_video_id: str = Form(""),
    character_image: UploadFile = File(None),
    reference_video: UploadFile = File(None),
):
    if not WAN_ANIMATE_API_URL:
        return JSONResponse(status_code=503, content={"error": "Wan Animate service not configured"})

    job_dir = os.path.join(WORK_DIR, f"animate_{uuid.uuid4().hex}")
    os.makedirs(job_dir, exist_ok=True)

    try:
        char_path = None
        if character_image and character_image.filename:
            ext = os.path.splitext(character_image.filename)[1].lower()
            if ext not in {".jpg", ".jpeg", ".png"}:
                return JSONResponse(status_code=400, content={"error": "character_image doit être jpg ou png"})
            char_path = os.path.join(job_dir, f"character{ext}")
            with open(char_path, "wb") as f:
                f.write(await character_image.read())
        elif character_image_url.strip():
            char_path = character_image_url.strip()

        ref_path = None
        if reference_video and reference_video.filename:
            ext = os.path.splitext(reference_video.filename)[1].lower()
            if ext != ".mp4":
                return JSONResponse(status_code=400, content={"error": "reference_video doit être mp4"})
            ref_path = os.path.join(job_dir, "reference.mp4")
            with open(ref_path, "wb") as f:
                f.write(await reference_video.read())
        elif reference_video_id.strip():
            entry = _REF_VIDEO_FLAT.get(reference_video_id.strip())
            if not entry:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"reference_video_id inconnu: '{reference_video_id}'"},
                )
            ref_path = entry["path"] or reference_video_id.strip()

        if not char_path:
            return JSONResponse(
                status_code=400,
                content={"error": "Fournissez character_image (fichier) ou character_image_url"},
            )
        if not ref_path:
            return JSONResponse(
                status_code=400,
                content={"error": "Fournissez reference_video (fichier) ou reference_video_id"},
            )

        try:
            async with httpx.AsyncClient(timeout=900) as client:
                if char_path.startswith("http"):
                    dl = await client.get(char_path)
                    char_bytes = dl.content
                    char_filename = "character.jpg"
                else:
                    with open(char_path, "rb") as f:
                        char_bytes = f.read()
                    char_filename = os.path.basename(char_path)

                with open(ref_path, "rb") as f:
                    ref_bytes = f.read()

                response = await client.post(
                    WAN_ANIMATE_API_URL.rstrip("/") + "/wan/animate",
                    files={
                        "character_image": (char_filename, char_bytes, "image/jpeg"),
                        "reference_video": ("reference.mp4", ref_bytes, "video/mp4"),
                    },
                    data={"mode": mode},
                )
        except httpx.TimeoutException:
            return JSONResponse(status_code=504, content={"error": "Wan Animate timed out"})

        if response.status_code != 200:
            try:
                detail = response.json()
            except Exception:
                detail = response.text[:500]
            return JSONResponse(
                status_code=response.status_code,
                content={"error": f"Wan Animate erreur {response.status_code}", "details": detail},
            )

        content_type = response.headers.get("content-type", "")
        if "video" in content_type or "octet-stream" in content_type:
            return StreamingResponse(
                iter([response.content]),
                media_type="video/mp4",
                headers={"Content-Disposition": "inline; filename=animated.mp4"},
            )

        data = response.json()
        return {"video_url": data.get("video_url", "")}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Erreur wan/animate: {str(e)}"})
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


@app.post("/voxtral/transcribe")
async def voxtral_transcribe(
    language: str = Form("fr"),
    output_format: str = Form("srt"),
    add_to_video: bool = Form(False),
    video_path: str = Form(""),
    audio_file: UploadFile = File(None),
    video_file: UploadFile = File(None),
):
    if not VOXTRAL_API_URL:
        return JSONResponse(status_code=503, content={"error": "Voxtral service not configured"})
    if not audio_file and not video_file:
        return JSONResponse(status_code=400, content={"error": "Fournissez audio_file ou video_file"})

    job_dir = os.path.join(WORK_DIR, f"voxtral_{uuid.uuid4().hex}")
    os.makedirs(job_dir, exist_ok=True)

    try:
        audio_path = None

        if video_file and video_file.filename:
            raw_video_path = os.path.join(job_dir, "input.mp4")
            with open(raw_video_path, "wb") as f:
                f.write(await video_file.read())
            audio_path = os.path.join(job_dir, "extracted.mp3")
            await async_run_cmd([
                "ffmpeg", "-y", "-i", raw_video_path,
                "-vn", "-acodec", "libmp3lame", "-ab", "192k",
                audio_path,
            ])
        elif audio_file and audio_file.filename:
            ext = os.path.splitext(audio_file.filename)[1].lower()
            if ext not in {".mp3", ".wav", ".m4a"}:
                return JSONResponse(
                    status_code=400,
                    content={"error": "audio_file doit être mp3, wav ou m4a"},
                )
            audio_path = os.path.join(job_dir, f"input{ext}")
            with open(audio_path, "wb") as f:
                f.write(await audio_file.read())

        if not audio_path or not os.path.exists(audio_path):
            return JSONResponse(status_code=500, content={"error": "Impossible d'obtenir le fichier audio"})

        with open(audio_path, "rb") as af:
            audio_bytes = af.read()

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(
                    VOXTRAL_API_URL.rstrip("/"),
                    content=audio_bytes,
                    headers={
                        "Content-Type": "audio/mpeg",
                        "X-Language": language,
                        "X-Output-Format": output_format,
                    },
                )
        except httpx.TimeoutException:
            return JSONResponse(status_code=504, content={"error": "Voxtral transcription timed out"})

        if response.status_code != 200:
            try:
                detail = response.json()
            except Exception:
                detail = response.text[:500]
            return JSONResponse(
                status_code=response.status_code,
                content={"error": f"Voxtral erreur {response.status_code}", "details": detail},
            )

        try:
            resp_data = response.json()
            subtitles = resp_data.get("subtitles", resp_data.get("text", ""))
        except Exception:
            subtitles = response.text

        output_video_path = None

        if add_to_video and video_path.strip() and subtitles:
            srt_path = os.path.join(job_dir, "subtitles.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(subtitles)

            burned_filename = f"burned_{uuid.uuid4().hex}.mp4"
            burned_path = os.path.join(VIDEO_DIR, burned_filename)
            srt_escaped = escape_srt_path(os.path.abspath(srt_path))
            subtitle_filter = (
                f"subtitles='{srt_escaped}':"
                "force_style='Alignment=2,MarginV=70,"
                "FontName=Arial,FontSize=24,Bold=1,"
                "PrimaryColour=&H00FFFFFF,"
                "OutlineColour=&H00000000,"
                "BorderStyle=3,Outline=2,Shadow=0,"
                "BackColour=&H99000000'"
            )
            await async_run_cmd([
                "ffmpeg", "-y", "-i", video_path.strip(),
                "-vf", subtitle_filter,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-c:a", "copy", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                burned_path,
            ])
            output_video_path = (
                f"{PUBLIC_BASE_URL}/video/{burned_filename}" if PUBLIC_BASE_URL else burned_path
            )

        return {
            "subtitles": subtitles,
            "output_video_path": output_video_path,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Erreur voxtral/transcribe: {str(e)}"})
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


@app.get("/reference-videos")
async def get_reference_videos():
    return {
        category: [
            {
                "id": v["id"],
                "name": v["name"],
                "preview_url": v["preview_url"],
                "duration": v["duration"],
            }
            for v in videos
        ]
        for category, videos in REFERENCE_VIDEOS.items()
    }


@app.get("/health")
async def health():
    async with _runpod_client() as client:
        try:
            r = await client.get("/health")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=f"RunPod unreachable: {exc}")
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"RunPod unreachable: {exc}")


@app.post("/qwen/analyze")
async def qwen_analyze(
    file: UploadFile = File(...),
    prompt: str = Query(default="Describe this image."),
):
    image_bytes = await file.read()
    async with _runpod_client() as client:
        try:
            r = await client.post(
                "/qwen/analyze",
                params={"prompt": prompt},
                files={"file": (file.filename, image_bytes, file.content_type or "image/jpeg")},
            )
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc))


@app.post("/wan/image2video")
async def wan_image2video(
    file: UploadFile = File(...),
    prompt: str = Query(default="Animate this image"),
    negative_prompt: str = Query(default="blurry, low quality"),
    num_frames: int = Query(default=81),
    num_inference_steps: int = Query(default=50),
    guidance_scale: float = Query(default=7.5),
    fps: int = Query(default=16),
):
    image_bytes = await file.read()
    async with _runpod_client() as client:
        try:
            r = await client.post(
                "/wan/image2video",
                params={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_frames": num_frames,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "fps": fps,
                },
                files={"file": (file.filename, image_bytes, file.content_type or "image/jpeg")},
            )
            r.raise_for_status()
            return Response(
                content=r.content,
                media_type="video/mp4",
                headers={"Content-Disposition": "attachment; filename=animated.mp4"},
            )
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc))


# NOTE: /wan/generate is defined above (~line 1495). This file's version accepts a JSON body
# (WanGenerateRequest: prompt/resolution/num_frames/num_inference_steps) and POSTs to
# RUNPOD_API_URL, returning the MP4 directly. api.py's version used WanT2VRequest with
# additional params (negative_prompt, fps, guidance_scale) and routed via _runpod_client().
# The implementation here is kept; use /wan/image2video for image-to-video via RunPod.


@app.get("/video-status/{job_id}")
async def video_status(job_id: str):
    job = VIDEO_JOBS.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job introuvable"})
    return JSONResponse(status_code=200, content=job)


# ── SOCIAL MEDIA PUBLISHING ─────────────────────────────────────────────────

class PublishScheduleRequest(BaseModel):
    video_path: str
    platforms: list
    title: str
    description: str = ""
    scheduled_time: str = ""
    user_plan: str = "free"
    auto_mode: bool = False
    user_id: str


async def _process_publish_job(job_id: str):
    job = PUBLISH_JOBS.get(job_id)
    if not job:
        return

    scheduled_time = job.get("scheduled_time", "")
    if scheduled_time:
        try:
            from datetime import datetime, timezone
            target = datetime.fromisoformat(scheduled_time)
            now = datetime.now(timezone.utc)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            delay = (target - now).total_seconds()
            if delay > 0:
                await asyncio.sleep(delay)
        except Exception:
            pass

    PUBLISH_JOBS[job_id]["status"] = "publishing"

    token_map = {
        "tiktok": ("TIKTOK_ACCESS_TOKEN", "TikTok token not configured"),
        "youtube": ("YOUTUBE_ACCESS_TOKEN", "YouTube token not configured"),
        "instagram": ("INSTAGRAM_ACCESS_TOKEN", "Instagram token not configured"),
        "facebook": ("FACEBOOK_ACCESS_TOKEN", "Facebook token not configured"),
    }

    for platform in job.get("platforms", []):
        env_key, msg = token_map.get(platform, (None, "Platform not supported"))
        if env_key is None:
            PUBLISH_JOBS[job_id]["results"][platform] = {
                "status": "error", "message": "Platform not supported"
            }
            continue
        token = os.getenv(env_key, "")
        if not token:
            PUBLISH_JOBS[job_id]["results"][platform] = {
                "status": "not_configured", "message": msg
            }
        else:
            PUBLISH_JOBS[job_id]["results"][platform] = {
                "status": "published", "message": f"Published to {platform}"
            }

    PUBLISH_JOBS[job_id]["status"] = "done"


@app.post("/publish/schedule")
async def publish_schedule(req: PublishScheduleRequest):
    plan_key = req.user_plan.lower().strip()

    if plan_key == "free":
        return JSONResponse(
            status_code=403,
            content={"error": "Publication automatique non disponible pour le plan gratuit"}
        )

    plan = PUBLISH_PLANS.get(plan_key)
    if not plan:
        return JSONResponse(
            status_code=400,
            content={"error": f"Plan inconnu: '{req.user_plan}'"}
        )

    allowed_platforms = plan.get("platforms", [])
    invalid = [p for p in req.platforms if p not in allowed_platforms]
    if invalid:
        return JSONResponse(
            status_code=403,
            content={"error": f"Plateformes non disponibles pour ce plan: {invalid}"}
        )

    limit_key = "auto_publications_per_day" if req.auto_mode else "manual_publications_per_day"
    daily_limit = plan.get(limit_key, 0)

    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    user_today_count = sum(
        1 for j in PUBLISH_JOBS.values()
        if j.get("user_id") == req.user_id
        and (j.get("scheduled_time", "") or "")[:10] == today
        and j.get("status") not in ("cancelled",)
    )

    if user_today_count >= daily_limit:
        return JSONResponse(
            status_code=429,
            content={"error": f"Limite de {daily_limit} publication(s) par jour atteinte pour ce plan"}
        )

    job_id = uuid.uuid4().hex
    PUBLISH_JOBS[job_id] = {
        "job_id": job_id,
        "user_id": req.user_id,
        "platforms": req.platforms,
        "status": "scheduled",
        "scheduled_time": req.scheduled_time,
        "video_path": req.video_path,
        "title": req.title,
        "description": req.description,
        "results": {},
    }

    asyncio.create_task(_process_publish_job(job_id))

    return JSONResponse(status_code=200, content={
        "job_id": job_id,
        "status": "scheduled",
        "platforms": req.platforms,
        "scheduled_time": req.scheduled_time,
    })


@app.get("/publish/status/{job_id}")
async def publish_status(job_id: str):
    job = PUBLISH_JOBS.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job introuvable"})
    return JSONResponse(status_code=200, content=job)


@app.get("/publish/history/{user_id}")
async def publish_history(user_id: str):
    jobs = [j for j in PUBLISH_JOBS.values() if j.get("user_id") == user_id]
    jobs.sort(key=lambda j: j.get("scheduled_time", ""), reverse=True)
    return JSONResponse(status_code=200, content={"jobs": jobs})


@app.get("/publish/plans")
async def publish_plans():
    return JSONResponse(status_code=200, content=PUBLISH_PLANS)


@app.get("/publish/platforms")
async def publish_platforms():
    return JSONResponse(status_code=200, content=PLATFORM_CONFIGS)


@app.delete("/publish/cancel/{job_id}")
async def publish_cancel(job_id: str):
    job = PUBLISH_JOBS.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job introuvable"})
    if job["status"] not in ("scheduled",):
        return JSONResponse(
            status_code=409,
            content={"error": f"Impossible d'annuler un job avec le statut '{job['status']}'"}
        )
    PUBLISH_JOBS[job_id]["status"] = "cancelled"
    return JSONResponse(status_code=200, content={"job_id": job_id, "status": "cancelled"})
