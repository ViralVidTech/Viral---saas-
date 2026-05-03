from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
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

AUDIO_DIR = "audio"
VIDEO_DIR = "videos"
WORK_DIR = "work"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)


# ── FONCTION WAN 2.2 ────────────────────────────────────────────────────────
async def generate_wan_video(prompt: str) -> str:
    """Appelle Wan 2.2 sur RunPod via POST avec un prompt court et visuel."""
    if not WAN_API_URL:
        return ""

    async with httpx.AsyncClient(timeout=900) as client:
        response = await client.post(
            f"{WAN_API_URL}/generate",
            params={"prompt": prompt}
        )

    if response.status_code != 200:
        print("WAN ERROR:", response.status_code, response.text)
        return ""

    data = response.json()
    video_url = data.get("video_url", "")

    if not video_url:
        print("WAN RESPONSE WITHOUT VIDEO:", data)
        return ""

    return f"{WAN_API_URL}{video_url}"


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


@app.get("/video-status/{job_id}")
async def video_status(job_id: str):
    job = VIDEO_JOBS.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job introuvable"})
    return JSONResponse(status_code=200, content=job)
