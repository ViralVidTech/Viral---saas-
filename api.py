import os
import io
import uuid
import base64
import json
import subprocess
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GOOGLE_TTS_API_KEY  = os.getenv("GOOGLE_TTS_API_KEY", "")
PEXELS_API_KEY      = os.getenv("PEXELS_API_KEY", "")
PIXABAY_API_KEY     = os.getenv("PIXABAY_API_KEY", "")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
PUBLIC_BASE_URL     = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
FAL_API_KEY         = os.getenv("FAL_API_KEY", "")
FISH_AUDIO_API_KEY  = os.getenv("FISH_AUDIO_API_KEY", "")
WAN_API_URL         = os.getenv("WAN_API_URL", "")

RUNPOD_BASE_URL = os.getenv("RUNPOD_API_URL", "https://849ams2zdun0ya-8000.proxy.runpod.net")
RUNPOD_TIMEOUT  = float(os.getenv("RUNPOD_TIMEOUT", "300"))

AUDIO_DIR = "audio"
VIDEO_DIR = "videos"
WORK_DIR  = "work"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(WORK_DIR,  exist_ok=True)

VIDEO_JOBS: dict = {}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="ViralVidTech API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

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
    text1: str = ""; text2: str = ""; text3: str = ""; text4: str = ""
    text5: str = ""; text6: str = ""; text7: str = ""; text8: str = ""
    video_url: str = "";  video_url2: str = "";  video_url3: str = "";  video_url4: str = ""
    video_url5: str = "";  video_url6: str = "";  video_url7: str = "";  video_url8: str = ""
    video_url9: str = "";  video_url10: str = ""; video_url11: str = ""; video_url12: str = ""
    video_url13: str = ""; video_url14: str = ""; video_url15: str = ""; video_url16: str = ""
    video_url17: str = ""; video_url18: str = ""; video_url19: str = ""; video_url20: str = ""
    video_url21: str = ""; video_url22: str = ""; video_url23: str = ""; video_url24: str = ""
    video_url25: str = ""; video_url26: str = ""; video_url27: str = ""; video_url28: str = ""
    video_url29: str = ""; video_url30: str = ""; video_url31: str = ""; video_url32: str = ""
    video_url33: str = ""; video_url34: str = ""; video_url35: str = ""; video_url36: str = ""
    video_url37: str = ""; video_url38: str = ""; video_url39: str = ""; video_url40: str = ""
    audio_url: str = ""; sync_url: str = ""; music_url: str = ""; wan_video: str = ""
    duration: int = 30

class WanT2VRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality"
    num_frames: int = 81
    width: int = 832
    height: int = 480
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    fps: int = 16

class ScanRequest(BaseModel):
    keyword: str = "money"
    platform: str = "TikTok"
    language: str = "en"

# ---------------------------------------------------------------------------
# RunPod client
# ---------------------------------------------------------------------------

def _runpod_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=RUNPOD_BASE_URL, timeout=RUNPOD_TIMEOUT)

# ---------------------------------------------------------------------------
# FFmpeg utilities
# ---------------------------------------------------------------------------

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
                timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=True
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
                timeout=httpx.Timeout(60.0, connect=10.0), follow_redirects=True
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
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000;   ms %= 60000
    s = ms // 1000;    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(subtitle_texts: list, segment_duration: float, out_path: str):
    WORDS_PER_BLOCK = 5
    all_texts = [" ".join((t or "").strip().split()) for t in subtitle_texts if (t or "").strip()]
    if not all_texts:
        open(out_path, "w").close()
        return
    total_words = sum(len(t.split()) for t in all_texts)
    total_duration = len(all_texts) * segment_duration
    seconds_per_word = total_duration / total_words if total_words > 0 else 0.35
    entries = []; idx = 1; current_time = 0.0; ADVANCE = 0.7
    for text in all_texts:
        words = text.split()
        for j in range(0, len(words), WORDS_PER_BLOCK):
            block = words[j:j + WORDS_PER_BLOCK]
            duration = len(block) * seconds_per_word
            start = max(0.0, current_time - ADVANCE)
            end = max(start + 0.1, start + duration - 0.05)
            entries.append(f"{idx}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{' '.join(block)}\n")
            idx += 1; current_time += duration
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries))

def escape_srt_path(path_str: str) -> str:
    return path_str.replace("\\", "\\\\").replace(":", "\\:")

def get_audio_duration(audio_path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0

def build_srt_from_timepoints(words: list, timepoints: list, out_path: str, words_per_block: int = 5):
    mark_times = {}
    for tp in timepoints:
        name = tp.get("markName", "")
        if name.startswith("w"):
            try: mark_times[int(name[1:])] = float(tp.get("timeSeconds", 0))
            except ValueError: pass
    if not mark_times or not words:
        return False
    last_idx = max(mark_times.keys())
    avg_word_dur = mark_times[last_idx] / last_idx if last_idx > 0 else 0.35
    estimated_end = mark_times.get(last_idx, 0) + avg_word_dur * words_per_block
    entries = []; entry_idx = 1
    for j in range(0, len(words), words_per_block):
        block_words = words[j:j + words_per_block]
        start = mark_times.get(j)
        if start is None: continue
        next_j = j + words_per_block
        end = (mark_times[next_j] - 0.05) if (next_j < len(words) and next_j in mark_times) else (estimated_end - 0.05)
        end = max(start + 0.1, end)
        entries.append(f"{entry_idx}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{' '.join(block_words)}\n")
        entry_idx += 1
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries))
    return True

# ---------------------------------------------------------------------------
# Wan legacy helper (appelle WAN_API_URL si définie, sinon RunPod)
# ---------------------------------------------------------------------------

async def generate_wan_video(prompt: str) -> str:
    if not WAN_API_URL:
        return ""
    base = WAN_API_URL.rstrip("/")
    for method in ["post", "get"]:
        try:
            async with httpx.AsyncClient(timeout=900) as client:
                fn = getattr(client, method)
                response = await fn(f"{base}/generate", params={"prompt": prompt})
            if response.status_code == 200:
                video_url = response.json().get("video_url", "")
                if video_url:
                    return video_url if video_url.startswith("http") else f"{base}{video_url}"
        except Exception:
            pass
    return ""

# ---------------------------------------------------------------------------
# Routes — frontend & static files
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>ViralVidTech API is running</h1>")

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    file_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Audio file not found"})
    media = "application/json" if filename.endswith(".json") else "audio/mpeg"
    return FileResponse(file_path, media_type=media, filename=filename)

@app.get("/video/{filename}")
async def serve_video(filename: str):
    file_path = os.path.join(VIDEO_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Video file not found"})
    return FileResponse(file_path, media_type="video/mp4", filename=filename)

# ---------------------------------------------------------------------------
# Health (RunPod proxy)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    async with _runpod_client() as client:
        try:
            r = await client.get("/health")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"RunPod unreachable: {exc}")

# ---------------------------------------------------------------------------
# Claude — génération de script + vidéos stock
# ---------------------------------------------------------------------------

@app.post("/generate")
async def generate(req: GenerateRequest):
    niche = req.niche.strip() or "general topic"
    lang = (req.langue or "en").lower()
    duration = req.duration if req.duration in [30, 45, 60] else 30
    if not ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY manquante"}

    lang_map = {"en": "English", "fr": "French", "es": "Spanish", "pt": "Portuguese"}
    target_language = lang_map.get(lang, "English")

    if duration == 60:
        nb_scenes = 8; seconds_per_scene = 7
        scene_structure = "HOOK: [...]\nCONTEXT: [...]\nPROBLEM: [...]\nAGITATION: [...]\nSOLUTION: [...]\nPROOF: [...]\nBENEFIT: [...]\nCTA: [...]"
    elif duration == 45:
        nb_scenes = 6; seconds_per_scene = 7
        scene_structure = "HOOK: [...]\nCONTEXT: [...]\nPROBLEM: [...]\nSOLUTION: [...]\nPROOF: [...]\nCTA: [...]"
    else:
        nb_scenes = 4; seconds_per_scene = 7
        scene_structure = "HOOK: [...]\nPROBLEM: [...]\nSOLUTION: [...]\nCTA: [...]"

    prompt = f"""Tu es un expert en création de contenu viral pour les réseaux sociaux.
Crée un script vidéo complet de {duration} secondes sur le sujet : "{niche}".
Le script doit être en {target_language}. La vidéo a {nb_scenes} scènes (~{seconds_per_scene}s chacune).
Chaque scène = 2 phrases complètes (15-25 mots chacune), ton naturel et persuasif.

Retourne EXACTEMENT dans ce format :

TITLES:
1. [titre 1]
2. [titre 2]
3. [titre 3]

{scene_structure}""".strip()

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 2000,
                      "messages": [{"role": "user", "content": prompt}]}
            )
        data = response.json()
        if response.status_code != 200:
            err = data.get("error", {})
            return {"error": err.get("message", "Claude API a échoué"), "details": data}

        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
        if not text:
            return {"error": "Claude a retourné un contenu vide"}

        scene_keys = ["HOOK", "CONTEXT", "PROBLEM", "AGITATION", "SOLUTION", "PROOF", "BENEFIT", "CTA"]
        titles = []; scenes = {k: "" for k in scene_keys}
        for line in text.split("\n"):
            clean = line.strip()
            if clean.startswith(("1.", "2.", "3.")):
                titles.append(clean[2:].strip())
            else:
                for key in scene_keys:
                    if clean.upper().startswith(f"{key}:"):
                        scenes[key] = clean.split(":", 1)[1].strip(); break

        script = "\n\n".join(scenes[k] for k in scene_keys if scenes[k])

        stop_words = {
            "le","la","les","un","une","des","de","du","et","en","que","qui","pour","par","sur",
            "dans","avec","est","sont","mais","donc","car","tout","tous","toute","cette","cela",
            "plus","très","bien","aussi","comme","même","fait","peut","faut","doit","avoir","être",
            "the","a","an","is","are","you","your","to","of","in","and","or","it","this","that",
            "we","our","my","not","can","will","all","if","have","has","been","with","they","their",
            "from","but","when","how"
        }

        def extract_keywords(scene_text, fallback):
            words = [w.strip(".,!?;:") for w in (scene_text or "").split()
                     if len(w.strip(".,!?;:")) > 4 and w.lower().strip(".,!?;:") not in stop_words]
            return (" ".join(words[:2]) + " " + fallback).strip() if words else fallback

        scene_queries = [extract_keywords(scenes[k], niche) for k in scene_keys if scenes[k]]
        while len(scene_queries) < 8:
            scene_queries.append(niche)

        async def fetch_pexels(client, query, count=5):
            results = []
            try:
                resp = await client.get(
                    "https://api.pexels.com/videos/search",
                    params={"query": query, "per_page": count + 4, "orientation": "portrait"},
                    headers={"Authorization": PEXELS_API_KEY}
                )
                for video in resp.json().get("videos", []):
                    if len(results) >= count: break
                    files = video.get("video_files", [])
                    hd = [f for f in files if f.get("quality") == "hd"]
                    url = (hd or files or [{}])[0].get("link", "")
                    if url: results.append(url)
            except Exception: pass
            return results

        async def fetch_pixabay(client, query, count=5):
            results = []
            try:
                resp = await client.get(
                    "https://pixabay.com/api/videos/",
                    params={"key": PIXABAY_API_KEY, "q": query, "per_page": count + 4, "video_type": "film"}
                )
                for hit in resp.json().get("hits", []):
                    if len(results) >= count: break
                    for q in ["medium", "small", "large"]:
                        url = hit.get("videos", {}).get(q, {}).get("url", "")
                        if url: results.append(url); break
            except Exception: pass
            return results

        video_urls = [""] * 40
        if PEXELS_API_KEY or PIXABAY_API_KEY:
            try:
                async with httpx.AsyncClient(timeout=45) as client:
                    for i, query in enumerate(scene_queries[:8]):
                        slot = i * 5; collected = []
                        if PEXELS_API_KEY:
                            collected += await fetch_pexels(client, query, 5)
                        if PIXABAY_API_KEY and len(collected) < 5:
                            collected += await fetch_pixabay(client, query, 5 - len(collected))
                        if PEXELS_API_KEY and len(collected) < 5 and query != niche:
                            collected += await fetch_pexels(client, niche, 5 - len(collected))
                        for j in range(5):
                            if slot + j < 40:
                                video_urls[slot + j] = collected[j] if j < len(collected) else (collected[j % len(collected)] if collected else "")
            except Exception: pass

        scene_list = [scenes[k] for k in scene_keys if scenes[k]]
        wan_video_url = ""
        if WAN_API_URL:
            try:
                wan_video_url = await generate_wan_video(niche or "funny talking character")
            except Exception: pass

        result = {
            "titles": titles, "script": script, "wan_video": wan_video_url,
            "nb_scenes": len(scene_list), "duration": duration,
            "raw_claude_text": text,
        }
        for i, scene in enumerate(scene_list[:8]):
            result[f"scene{i+1}"] = scene
        for i in range(8 - len(scene_list)):
            result[f"scene{len(scene_list)+i+1}"] = ""
        for i in range(40):
            key = "video_url" if i == 0 else f"video_url{i+1}"
            result[key] = video_urls[i]
        return result

    except Exception as e:
        return {"error": f"Erreur Claude: {str(e)}"}

# ---------------------------------------------------------------------------
# Scan trends
# ---------------------------------------------------------------------------

@app.post("/scan-trends")
async def scan_trends(req: ScanRequest):
    if not ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY manquante"}
    lang_label = {"fr": "French", "es": "Spanish", "pt": "Portuguese"}.get(req.language, "English")
    prompt = f"""You are a viral video trend analyst for {req.platform}.
Analyze viral trends for: "{req.keyword}" in {lang_label}.
Return ONLY a valid JSON array with exactly 5 objects, no markdown.
Each object: title, niche, platform, viralScore (0-100), bestDuration, targetAudience, whyViral, hookIdea, hashtags."""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 1500,
                      "messages": [{"role": "user", "content": prompt}]}
            )
        if response.status_code != 200:
            return {"error": f"Claude API erreur {response.status_code}"}
        raw = "".join(b.get("text","") for b in response.json().get("content",[]) if b.get("type")=="text")
        clean = raw.replace("```json","").replace("```","").strip()
        try:
            results = json.loads(clean)
        except json.JSONDecodeError as je:
            return {"error": f"JSON invalide: {str(je)}", "raw": clean[:200]}
        return {"success": True, "results": results}
    except Exception as e:
        return {"error": f"Erreur scan: {str(e)}"}

# ---------------------------------------------------------------------------
# Google TTS
# ---------------------------------------------------------------------------

@app.post("/generate-audio")
async def generate_audio(req: TTSRequest):
    if not GOOGLE_TTS_API_KEY:
        return {"error": "GOOGLE_TTS_API_KEY manquante"}
    if not PUBLIC_BASE_URL:
        return {"error": "PUBLIC_BASE_URL manquante"}
    if not req.text.strip():
        return {"error": "Le texte est vide"}

    google_url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_TTS_API_KEY}"
    words = req.text.strip().split()
    supports_timepoints = any(v in req.voiceName for v in ["Neural2", "Wavenet", "Standard"])

    if supports_timepoints:
        def escape_xml(t):
            return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;").replace("'","&apos;")
        ssml = "<speak>" + " ".join(f'<mark name="w{i}"/>{escape_xml(w)}' for i, w in enumerate(words)) + "</speak>"
        payload = {"input": {"ssml": ssml}, "voice": {"languageCode": req.languageCode, "name": req.voiceName},
                   "audioConfig": {"audioEncoding": "MP3", "speakingRate": req.speakingRate},
                   "enableTimePointing": ["SSML_MARK"]}
    else:
        payload = {"input": {"text": req.text}, "voice": {"languageCode": req.languageCode, "name": req.voiceName},
                   "audioConfig": {"audioEncoding": "MP3", "speakingRate": req.speakingRate}}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(google_url, json=payload)
        data = response.json()
        if response.status_code != 200 and supports_timepoints:
            payload_fb = {"input": {"text": req.text}, "voice": {"languageCode": req.languageCode, "name": req.voiceName},
                          "audioConfig": {"audioEncoding": "MP3", "speakingRate": req.speakingRate}}
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(google_url, json=payload_fb)
            data = response.json(); supports_timepoints = False
        if response.status_code != 200:
            return {"error": "Google TTS a échoué", "details": data}

        audio_content = data.get("audioContent")
        if not audio_content:
            return {"error": "Aucun audio retourné par Google"}

        filename = f"{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(audio_content))

        timepoints = data.get("timepoints", []) if supports_timepoints else []
        sync_filename = filename.replace(".mp3", "_sync.json")
        with open(os.path.join(AUDIO_DIR, sync_filename), "w", encoding="utf-8") as f:
            json.dump({"words": words, "timepoints": timepoints}, f)

        return {"success": True, "audio_url": f"{PUBLIC_BASE_URL}/audio/{filename}",
                "sync_url": f"{PUBLIC_BASE_URL}/audio/{sync_filename}",
                "filename": filename, "timepoints_count": len(timepoints)}
    except Exception as e:
        return {"error": f"Erreur TTS: {str(e)}"}

# ---------------------------------------------------------------------------
# Fish Audio TTS
# ---------------------------------------------------------------------------

@app.post("/generate-audio-fish")
async def generate_audio_fish(req: FishTTSRequest):
    if not FISH_AUDIO_API_KEY:
        return {"error": "FISH_AUDIO_API_KEY manquante"}
    if not PUBLIC_BASE_URL:
        return {"error": "PUBLIC_BASE_URL manquante"}
    if not req.text.strip():
        return {"error": "Le texte est vide"}
    try:
        payload = {"text": req.text, "format": "mp3", "latency": "balanced", "normalize": True}
        if req.voice_id:
            payload["reference_id"] = req.voice_id
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://api.fish.audio/v1/tts",
                headers={"Authorization": f"Bearer {FISH_AUDIO_API_KEY}", "Content-Type": "application/json"},
                json=payload
            )
        if response.status_code != 200:
            return {"error": f"Fish Audio API erreur {response.status_code}",
                    "details": response.json() if "json" in response.headers.get("content-type","") else response.text}
        filename = f"fish_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return {"error": "Fish Audio a retourné un fichier vide"}
        words = req.text.strip().split()
        sync_filename = filename.replace(".mp3", "_sync.json")
        with open(os.path.join(AUDIO_DIR, sync_filename), "w", encoding="utf-8") as f:
            json.dump({"words": words, "timepoints": []}, f)
        return {"success": True, "audio_url": f"{PUBLIC_BASE_URL}/audio/{filename}",
                "sync_url": f"{PUBLIC_BASE_URL}/audio/{sync_filename}", "filename": filename, "provider": "fish_audio"}
    except Exception as e:
        return {"error": f"Erreur Fish Audio: {str(e)}"}

@app.get("/fish-voices")
async def list_fish_voices():
    if not FISH_AUDIO_API_KEY:
        return {"error": "FISH_AUDIO_API_KEY manquante"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                "https://api.fish.audio/v1/model",
                headers={"Authorization": f"Bearer {FISH_AUDIO_API_KEY}"},
                params={"page_size": 20, "sort_by": "task_count"}
            )
        if response.status_code != 200:
            return {"error": f"Fish Audio API erreur {response.status_code}"}
        voices = [{"id": i.get("_id",""), "name": i.get("title",""), "language": i.get("languages",[]),
                   "description": i.get("description","")} for i in response.json().get("items", [])]
        return {"success": True, "voices": voices}
    except Exception as e:
        return {"error": f"Erreur Fish Audio voices: {str(e)}"}

# ---------------------------------------------------------------------------
# Flux 2 Pro — génération d'image
# ---------------------------------------------------------------------------

@app.post("/generate-image")
async def generate_image(req: FluxImageRequest):
    if not FAL_API_KEY:
        return {"error": "FAL_API_KEY manquante"}
    if not req.prompt.strip():
        return {"error": "Le prompt est vide"}
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                "https://fal.run/fal-ai/flux-pro/v1.1-ultra",
                headers={"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"},
                json={"prompt": req.prompt, "image_size": req.image_size,
                      "num_inference_steps": req.num_inference_steps,
                      "guidance_scale": req.guidance_scale, "num_images": req.num_images,
                      "enable_safety_checker": req.enable_safety_checker, "output_format": "jpeg"}
            )
        if response.status_code != 200:
            return {"error": f"Flux API erreur {response.status_code}",
                    "details": response.json() if "json" in response.headers.get("content-type","") else response.text}
        data = response.json()
        image_urls = [img.get("url","") for img in data.get("images", []) if img.get("url")]
        if not image_urls:
            return {"error": "Flux n'a retourné aucune image", "details": data}
        return {"success": True, "images": image_urls, "image_url": image_urls[0],
                "prompt": req.prompt, "provider": "flux_pro_fal", "seed": data.get("seed")}
    except Exception as e:
        return {"error": f"Erreur Flux: {str(e)}"}

# ---------------------------------------------------------------------------
# FFmpeg — création vidéo finale (async job)
# ---------------------------------------------------------------------------

async def _process_video(job_id: str, req: VideoRequest):
    job_dir = None
    try:
        if not PUBLIC_BASE_URL:
            VIDEO_JOBS[job_id] = {"status": "failed", "error": "PUBLIC_BASE_URL manquante"}; return
        if not ffmpeg_exists():
            VIDEO_JOBS[job_id] = {"status": "failed", "error": "FFmpeg non installé"}; return

        chosen_duration = req.duration if req.duration in [30, 45, 60] else 30
        nb_scenes = 8 if chosen_duration == 60 else (6 if chosen_duration == 45 else 4)

        all_video_urls = [
            (getattr(req, "video_url" if i == 0 else f"video_url{i+1}") or "").strip()
            for i in range(40)
        ]
        all_subtitle_texts = [
            (getattr(req, f"text{i+1}") or "").strip()[:200] for i in range(8)
        ]

        if req.wan_video:
            clip_urls = [req.wan_video] * nb_scenes
            subtitle_texts = [all_subtitle_texts[i] if i < len(all_subtitle_texts) else "" for i in range(nb_scenes)]
            CLIPS_PER_SCENE = 1
        else:
            valid_urls = [u for u in all_video_urls if u]
            if not valid_urls:
                VIDEO_JOBS[job_id] = {"status": "failed", "error": "Aucune vidéo fournie"}; return
            CLIPS_PER_SCENE = 5
            clip_urls = []; subtitle_texts = []
            for i in range(nb_scenes):
                collected = [all_video_urls[i * CLIPS_PER_SCENE + j]
                             for j in range(CLIPS_PER_SCENE)
                             if i * CLIPS_PER_SCENE + j < len(all_video_urls) and all_video_urls[i * CLIPS_PER_SCENE + j]]
                while len(collected) < CLIPS_PER_SCENE and valid_urls:
                    collected.append(valid_urls[len(collected) % len(valid_urls)])
                clip_urls.extend(collected[:CLIPS_PER_SCENE])
                subtitle_texts.append(all_subtitle_texts[i] if i < len(all_subtitle_texts) else "")
                subtitle_texts.extend([""] * (CLIPS_PER_SCENE - 1))

        nb_clips_total = len(clip_urls)
        job_dir = os.path.join(WORK_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        voice_path = None; real_total_duration = float(chosen_duration)
        voice_url = (req.audio_url or "").strip()
        if voice_url:
            voice_path = os.path.join(job_dir, "voice.mp3")
            await download_audio_file(voice_url, voice_path)
            measured = get_audio_duration(voice_path)
            if measured > 1.0:
                real_total_duration = measured

        clip_duration = real_total_duration / nb_clips_total
        raw_paths = [os.path.join(job_dir, f"raw_{i}.mp4") for i in range(len(clip_urls))]
        await asyncio.gather(*[download_file(url, path) for url, path in zip(clip_urls, raw_paths) if url])

        norm_paths = [os.path.join(job_dir, f"seg_{i}.mp4") for i in range(len(raw_paths))]

        def normalize_clip(args):
            src, dst = args
            if not os.path.exists(src): return
            run_cmd(["ffmpeg", "-y", "-i", src, "-t", str(clip_duration),
                     "-vf", "scale=405:720:force_original_aspect_ratio=increase,crop=405:720,fps=25,format=yuv420p",
                     "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-r", "25", dst])

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(normalize_clip, zip(raw_paths, norm_paths)))

        norm_paths = [p for p in norm_paths if os.path.exists(p)]
        if not norm_paths:
            raise RuntimeError("Aucun clip normalisé produit")

        concat_path = os.path.join(job_dir, "concat.txt")
        with open(concat_path, "w") as f:
            f.writelines(f"file '{os.path.abspath(p)}'\n" for p in norm_paths)

        stitched_raw = os.path.join(job_dir, "stitched_raw.mp4")
        await async_run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_path,
                              "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                              "-r", "25", "-pix_fmt", "yuv420p", "-an", stitched_raw])

        stitched_path = os.path.join(job_dir, "stitched.mp4")
        stitched_dur = get_audio_duration(stitched_raw)
        if stitched_dur < real_total_duration - 0.5:
            await async_run_cmd(["ffmpeg", "-y", "-stream_loop", "-1", "-i", stitched_raw,
                                  "-t", str(real_total_duration), "-c:v", "libx264",
                                  "-preset", "ultrafast", "-crf", "28", "-r", "25", "-pix_fmt", "yuv420p", "-an", stitched_path])
        else:
            await async_run_cmd(["ffmpeg", "-y", "-i", stitched_raw,
                                  "-t", str(real_total_duration), "-c:v", "copy", "-an", stitched_path])

        final_audio_path = None
        music_url = (req.music_url or "").strip()
        if voice_url and voice_path:
            if music_url:
                music_path = os.path.join(job_dir, "music.mp3")
                await download_audio_file(music_url, music_path)
                mixed_path = os.path.join(job_dir, "mixed.m4a")
                await async_run_cmd(["ffmpeg", "-y", "-fflags", "+genpts", "-i", voice_path,
                                     "-stream_loop", "-1", "-i", music_path,
                                     "-filter_complex",
                                     f"[1:a]volume=0.12,atrim=0:{real_total_duration}[bg];"
                                     f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                                     "-map", "[aout]", "-c:a", "aac", "-b:a", "192k", mixed_path])
                final_audio_path = mixed_path
            else:
                final_audio_path = voice_path

        srt_path = os.path.join(job_dir, "subtitles.srt")
        srt_built = False
        sync_url = (req.sync_url or "").strip()
        if sync_url:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    sync_data = (await client.get(sync_url)).json()
                srt_built = build_srt_from_timepoints(
                    sync_data.get("words",[]), sync_data.get("timepoints",[]), srt_path)
            except Exception: pass
        if not srt_built:
            nb_sub = len([t for t in subtitle_texts if t.strip()])
            seg_dur = real_total_duration / nb_sub if nb_sub > 0 else clip_duration
            write_srt(subtitle_texts, seg_dur, srt_path)

        output_filename = f"{job_id}.mp4"
        output_path = os.path.join(VIDEO_DIR, output_filename)
        srt_escaped = escape_srt_path(os.path.abspath(srt_path))
        subtitle_filter = (
            f"subtitles='{srt_escaped}':"
            "force_style='Alignment=2,MarginV=70,PlayResX=405,PlayResY=720,"
            "FontName=Arial,FontSize=24,Bold=1,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,BorderStyle=3,Outline=2,Shadow=0,BackColour=&H99000000'"
        )

        base_cmd = ["ffmpeg", "-y", "-i", stitched_path]
        if final_audio_path:
            base_cmd += ["-i", final_audio_path]
        base_cmd += ["-vf", subtitle_filter]
        if final_audio_path:
            base_cmd += ["-map", "0:v:0", "-map", "1:a:0"]
        base_cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-r", "25",
                     "-c:a", "aac", "-b:a", "128k", "-pix_fmt", "yuv420p",
                     "-movflags", "+faststart", output_path]
        await async_run_cmd(base_cmd)

        shutil.rmtree(job_dir, ignore_errors=True)
        VIDEO_JOBS[job_id] = {"status": "done", "video_url": f"{PUBLIC_BASE_URL}/video/{output_filename}"}

    except Exception as e:
        if job_dir: shutil.rmtree(job_dir, ignore_errors=True)
        VIDEO_JOBS[job_id] = {"status": "failed", "error": str(e)}


@app.post("/create-video")
async def create_video(req: VideoRequest):
    if not PUBLIC_BASE_URL:
        return JSONResponse(status_code=400, content={"error": "PUBLIC_BASE_URL manquante"})
    if not ffmpeg_exists():
        return JSONResponse(status_code=500, content={"error": "FFmpeg non installé"})
    job_id = uuid.uuid4().hex
    VIDEO_JOBS[job_id] = {"status": "processing"}
    asyncio.create_task(_process_video(job_id, req))
    return JSONResponse(status_code=200, content={"success": True, "job_id": job_id,
                                                   "message": "Rendu démarré. Vérifiez /video-status/{job_id}"})

@app.get("/video-status/{job_id}")
async def video_status(job_id: str):
    job = VIDEO_JOBS.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job introuvable"})
    return JSONResponse(status_code=200, content=job)

# ---------------------------------------------------------------------------
# RunPod proxies — Voxtral, Qwen, Wan
# ---------------------------------------------------------------------------

@app.post("/voxtral/transcribe")
async def voxtral_transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    async with _runpod_client() as client:
        try:
            r = await client.post("/voxtral/transcribe",
                                  files={"file": (file.filename, audio_bytes, file.content_type or "audio/wav")})
            r.raise_for_status(); return r.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

@app.post("/qwen/analyze")
async def qwen_analyze(file: UploadFile = File(...), prompt: str = Query(default="Describe this image.")):
    image_bytes = await file.read()
    async with _runpod_client() as client:
        try:
            r = await client.post("/qwen/analyze", params={"prompt": prompt},
                                  files={"file": (file.filename, image_bytes, file.content_type or "image/jpeg")})
            r.raise_for_status(); return r.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

@app.post("/wan/generate")
async def wan_generate(request: WanT2VRequest):
    async with _runpod_client() as client:
        try:
            r = await client.post("/wan/generate", json=request.model_dump())
            r.raise_for_status()
            return Response(content=r.content, media_type="video/mp4",
                            headers={"Content-Disposition": "attachment; filename=output.mp4"})
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
            r = await client.post("/wan/image2video",
                                  params={"prompt": prompt, "negative_prompt": negative_prompt,
                                          "num_frames": num_frames, "num_inference_steps": num_inference_steps,
                                          "guidance_scale": guidance_scale, "fps": fps},
                                  files={"file": (file.filename, image_bytes, file.content_type or "image/jpeg")})
            r.raise_for_status()
            return Response(content=r.content, media_type="video/mp4",
                            headers={"Content-Disposition": "attachment; filename=animated.mp4"})
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
