from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import base64
import httpx
import subprocess
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# CONFIG
GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY", "")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

AUDIO_DIR = "audio"
VIDEO_DIR = "videos"
WORK_DIR = "work"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)


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
    music_url: str = ""


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


async def download_file(url: str, dest_path: str):
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(response.content)


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
    entries = []
    idx = 1
    for i, text in enumerate(subtitle_texts):
        clean = " ".join((text or "").strip().split())
        if not clean:
            continue
        start = i * segment_duration
        end = start + segment_duration - 0.15
        entries.append(
            f"{idx}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{clean}\n"
        )
        idx += 1
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries))


def escape_srt_path(path_str: str) -> str:
    return path_str.replace("\\", "\\\\").replace(":", "\\:")

def get_audio_duration(audio_path: str) -> float:
    """Mesure la durée réelle d'un fichier audio en secondes via ffprobe."""
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

# HOME
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>API is running</h1>"


# SERVIR LES FICHIERS AUDIO
@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    file_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(file_path):
        return {"error": "Audio file not found"}
    return FileResponse(file_path, media_type="audio/mpeg", filename=filename)


# SERVIR LES VIDÉOS GÉNÉRÉES
@app.get("/video/{filename}")
async def serve_video(filename: str):
    file_path = os.path.join(VIDEO_DIR, filename)
    if not os.path.exists(file_path):
        return {"error": "Video file not found"}
    return FileResponse(file_path, media_type="video/mp4", filename=filename)


# GÉNÉRER SCRIPT + VIDÉOS PEXELS
@app.post("/generate")
async def generate(req: GenerateRequest):
    niche = req.niche.strip() or "general topic"
    lang = (req.langue or "en").lower()

    if not ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY manquante"}

    lang_map = {
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "pt": "Portuguese",
    }
    target_language = lang_map.get(lang, "English")

    prompt = f"""
Create viral short-form video content about "{niche}" in {target_language}.

Return exactly in this format:

TITLES:
1. [title 1]
2. [title 2]
3. [title 3]

HOOK: [one punchy sentence - max 8 words]

PROBLEM: [one sentence - max 8 words]

SOLUTION: [one sentence - max 8 words]

CTA: [one call to action - max 8 words]

Important rules:
- Write everything in {target_language}
- Keep it natural, persuasive, and social-media ready
- Do not add explanations
- Do not add markdown
- Do not add anything outside the required format
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
                    "max_tokens": 700,
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
        hook = ""
        problem = ""
        solution = ""
        cta = ""

        for line in lines:
            clean = line.strip()
            if clean.startswith("1."):
                titles.append(clean[2:].strip())
            elif clean.startswith("2."):
                titles.append(clean[2:].strip())
            elif clean.startswith("3."):
                titles.append(clean[2:].strip())
            elif clean.upper().startswith("HOOK:"):
                hook = clean.split(":", 1)[1].strip()
            elif clean.upper().startswith("PROBLEM:"):
                problem = clean.split(":", 1)[1].strip()
            elif clean.upper().startswith("SOLUTION:"):
                solution = clean.split(":", 1)[1].strip()
            elif clean.upper().startswith("CTA:"):
                cta = clean.split(":", 1)[1].strip()

        script_parts = [hook, problem, solution, cta]
        script = "\n\n".join([p for p in script_parts if p])

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
            "raw_claude_text": text
        }

    except Exception as e:
        return {"error": f"Erreur Claude: {str(e)}"}


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
        "voice": {"languageCode": req.languageCode, "name": req.voiceName},
        "audioConfig": {"audioEncoding": "MP3", "speakingRate": req.speakingRate},
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(google_url, json=payload)
        data = response.json()
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
        return {"success": True, "audio_url": audio_url, "filename": filename}
    except Exception as e:
        return {"error": f"Erreur TTS: {str(e)}"}


# FFMPEG: CRÉER LA VIDÉO FINALE
@app.post("/create-video")
async def create_video(req: VideoRequest):
    job_dir = None
    try:
        if not PUBLIC_BASE_URL:
            return JSONResponse(status_code=400, content={"error": "PUBLIC_BASE_URL manquante"})

        if not ffmpeg_exists():
            return JSONResponse(status_code=500, content={
                "error": "FFmpeg n'est pas installé sur le serveur"
            })

        video_urls = [
            (req.video_url or "").strip(),
            (req.video_url2 or "").strip(),
            (req.video_url3 or "").strip(),
            (req.video_url4 or "").strip(),
        ]
        subtitle_texts = [
            (req.text1 or "").strip()[:140],
            (req.text2 or "").strip()[:140],
            (req.text3 or "").strip()[:140],
            (req.text4 or "").strip()[:140],
        ]

        valid_video_urls = [u for u in video_urls if u]
        if not valid_video_urls:
            return JSONResponse(status_code=400, content={"error": "Aucune vidéo n'a été fournie"})

        segment_duration = 5
        total_duration = len(valid_video_urls) * segment_duration

        job_id = uuid.uuid4().hex
        job_dir = os.path.join(WORK_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        # 1) Télécharger les vidéos
        raw_video_paths = []
        for i, url in enumerate(valid_video_urls):
            raw_path = os.path.join(job_dir, f"raw_{i}.mp4")
            await download_file(url, raw_path)
            raw_video_paths.append(raw_path)

        # 2) Normaliser chaque vidéo : 405×720, 5 sec, 30 fps, sans audio
        norm_video_paths = []
        for i, raw_path in enumerate(raw_video_paths):
            norm_path = os.path.join(job_dir, f"seg_{i}.mp4")
            run_cmd([
                "ffmpeg", "-y",
                "-i", raw_path,
                "-t", str(segment_duration),
                "-vf", "scale=405:720:force_original_aspect_ratio=increase,crop=405:720,fps=30,format=yuv420p",
                "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-r", "30",
                norm_path
            ])
            norm_video_paths.append(norm_path)

        # 3) Concaténer les segments
        concat_list_path = os.path.join(job_dir, "concat.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for path in norm_video_paths:
                abs_path = os.path.abspath(path)
                f.write(f"file '{abs_path}'\n")

        stitched_path = os.path.join(job_dir, "stitched.mp4")
        run_cmd([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-r", "30",
            "-pix_fmt", "yuv420p", "-an",
            stitched_path
        ])

        # 4) Préparer l'audio (voix + musique optionnelle)
        final_audio_path = None
        voice_url = (req.audio_url or "").strip()
        music_url = (req.music_url or "").strip()

        if voice_url:
            voice_path = os.path.join(job_dir, "voice.mp3")
            await download_file(voice_url, voice_path)

            if music_url:
                music_path = os.path.join(job_dir, "music.mp3")
                await download_file(music_url, music_path)
                mixed_path = os.path.join(job_dir, "mixed.m4a")
                run_cmd([
                    "ffmpeg", "-y",
                    "-i", voice_path,
                    "-stream_loop", "-1", "-i", music_path,
                    "-filter_complex",
                    f"[1:a]volume=0.12,atrim=0:{total_duration}[bg];"
                    f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                    "-map", "[aout]",
                    "-c:a", "aac", "-b:a", "192k",
                    mixed_path
                ])
                final_audio_path = mixed_path
            else:
                voice_aac_path = os.path.join(job_dir, "voice_only.m4a")
                run_cmd([
                    "ffmpeg", "-y",
                    "-i", voice_path,
                    "-t", str(total_duration),
                    "-c:a", "aac", "-b:a", "192k",
                    voice_aac_path
                ])
                final_audio_path = voice_aac_path

        # 5) Générer le fichier SRT synchronisé avec la durée RÉELLE de la voix
        # Si on a une voix, on mesure sa durée exacte et on divise par le nombre
        # de sous-titres pour que chaque texte apparaisse au bon moment.
        # Si pas de voix, on utilise les 5 secondes fixes par segment.
        srt_path = os.path.join(job_dir, "subtitles.srt")
        nb_subtitles = len([t for t in subtitle_texts[:len(valid_video_urls)] if t.strip()])

        if voice_url and nb_subtitles > 0:
            voice_duration = get_audio_duration(os.path.join(job_dir, "voice.mp3"))
            if voice_duration > 0:
                srt_segment_duration = voice_duration / nb_subtitles
            else:
                srt_segment_duration = segment_duration
        else:
            srt_segment_duration = segment_duration

        write_srt(subtitle_texts[:len(valid_video_urls)], srt_segment_duration, srt_path)

        # 6) Rendu final : sous-titres brûlés, framerate forcé à 30
        output_filename = f"{job_id}.mp4"
        output_path = os.path.join(VIDEO_DIR, output_filename)

        srt_escaped = escape_srt_path(os.path.abspath(srt_path))
        subtitle_filter = (
            f"subtitles='{srt_escaped}':"
            "force_style='Alignment=2,MarginV=28,"
            "FontName=Arial,FontSize=17,Bold=1,"
            "PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,"
            "BorderStyle=1,Outline=2,Shadow=0'"
        )

        if final_audio_path:
            run_cmd([
                "ffmpeg", "-y",
                "-i", stitched_path,
                "-i", final_audio_path,
                "-vf", subtitle_filter,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-r", "30",                  # framerate final forcé à 30
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-shortest",
                output_path
            ])
        else:
            run_cmd([
                "ffmpeg", "-y",
                "-i", stitched_path,
                "-vf", subtitle_filter,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-r", "30",                  # framerate final forcé à 30
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ])

        # Nettoyage du dossier temporaire de travail
        shutil.rmtree(job_dir, ignore_errors=True)

        video_url = f"{PUBLIC_BASE_URL}/video/{output_filename}"

        return JSONResponse(status_code=200, content={
            "success": True,
            "video_url": video_url,
            "message": "Vidéo générée avec FFmpeg"
        })

    except httpx.HTTPError as e:
        shutil.rmtree(job_dir, ignore_errors=True) if job_dir else None
        return JSONResponse(status_code=500, content={
            "error": f"Erreur téléchargement: {str(e)}"
        })
    except RuntimeError as e:
        shutil.rmtree(job_dir, ignore_errors=True) if job_dir else None
        return JSONResponse(status_code=500, content={
            "error": f"Erreur FFmpeg: {str(e)}"
        })
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True) if job_dir else None
        return JSONResponse(status_code=500, content={
            "error": f"Erreur create-video: {str(e)}"
        })
