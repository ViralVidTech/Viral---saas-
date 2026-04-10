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
    duration: int = 30  # durée choisie : 30, 45 ou 60 secondes


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
    audio_url: str = ""
    music_url: str = ""
    duration: int = 30  # 30, 45 ou 60 secondes


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
    """
    Génère des sous-titres synchronisés en calculant la durée par mot.
    
    Méthode :
    - On compte le nombre total de mots dans tous les textes
    - On calcule : durée_par_mot = durée_totale / nb_total_mots
    - Chaque bloc de 5 mots dure donc : 5 × durée_par_mot
    - Les sous-titres suivent le rythme réel de la voix
    """
    WORDS_PER_BLOCK = 5

    # Collecter tous les mots de toutes les scènes
    all_texts = []
    for text in subtitle_texts:
        clean = " ".join((text or "").strip().split())
        if clean:
            all_texts.append(clean)

    if not all_texts:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("")
        return

    # Compter le total de mots
    total_words = sum(len(t.split()) for t in all_texts)
    total_duration = len(all_texts) * segment_duration

    # Durée par mot = durée totale / nombre total de mots
    if total_words > 0:
        seconds_per_word = total_duration / total_words
    else:
        seconds_per_word = 0.35  # fallback : 0.35s par mot

    entries = []
    idx = 1
    current_time = 0.0

    for text in all_texts:
        words = text.split()

        # Découper en blocs de WORDS_PER_BLOCK mots
        for j in range(0, len(words), WORDS_PER_BLOCK):
            block = words[j:j + WORDS_PER_BLOCK]
            block_text = " ".join(block)
            block_word_count = len(block)

            start = current_time
            duration = block_word_count * seconds_per_word
            end = start + duration - 0.05  # petit écart pour éviter chevauchement

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

    # Adapter le nombre de scènes selon la durée choisie
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
    else:  # 30 secondes
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
        # Toutes les scènes possibles selon la durée
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

        # Construire le script dans l'ordre naturel, en ignorant les scènes vides
        script_parts = [scenes[key] for key in scene_keys if scenes[key]]
        script = "\n\n".join(script_parts)

        video_urls = ["", "", "", "", "", "", "", ""]

        if PEXELS_API_KEY:
            try:
                # Récupérer jusqu'à 8 vidéos depuis Pexels
                async with httpx.AsyncClient(timeout=30) as client:
                    pexels_response = await client.get(
                        "https://api.pexels.com/videos/search",
                        params={"query": niche, "per_page": 8},
                        headers={"Authorization": PEXELS_API_KEY},
                    )
                pexels_data = pexels_response.json()
                videos = pexels_data.get("videos", [])
                for i, video in enumerate(videos[:8]):
                    files = video.get("video_files", [])
                    hd_files = [f for f in files if f.get("quality") == "hd"]
                    if hd_files:
                        video_urls[i] = hd_files[0].get("link", "")
                    elif files:
                        video_urls[i] = files[0].get("link", "")
            except Exception:
                pass

        # Retourner chaque scène séparément pour les sous-titres
        scene_list = [scenes[key] for key in scene_keys if scenes[key]]

        return {
            "titles": titles,
            "script": script,
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
            "video_url2": video_urls[1],
            "video_url3": video_urls[2],
            "video_url4": video_urls[3],
            "video_url5": video_urls[4],
            "video_url6": video_urls[5],
            "video_url7": video_urls[6],
            "video_url8": video_urls[7],
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

        # Durée choisie par l'utilisateur
        chosen_duration = req.duration if req.duration in [30, 45, 60] else 30

        # Nombre de scènes selon la durée
        # 30 sec → 4 scènes, 45 sec → 6 scènes, 60 sec → 8 scènes
        if chosen_duration == 60:
            nb_scenes = 8
        elif chosen_duration == 45:
            nb_scenes = 6
        else:
            nb_scenes = 4

        # Récupérer toutes les URLs vidéo disponibles (jusqu'à 8)
        all_video_urls = [
            (req.video_url or "").strip(),
            (req.video_url2 or "").strip(),
            (req.video_url3 or "").strip(),
            (req.video_url4 or "").strip(),
            (req.video_url5 or "").strip(),
            (req.video_url6 or "").strip(),
            (req.video_url7 or "").strip(),
            (req.video_url8 or "").strip(),
        ]

        # Récupérer tous les textes de sous-titres (jusqu'à 8)
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

        # Garder seulement les URLs valides, limiter au nombre de scènes voulu
        valid_video_urls_raw = [u for u in all_video_urls if u]
        if not valid_video_urls_raw:
            return JSONResponse(status_code=400, content={"error": "Aucune vidéo n'a été fournie"})

        # Si on a moins de vidéos que de scènes, on recycle les vidéos disponibles
        video_urls = []
        subtitle_texts = []
        for i in range(nb_scenes):
            video_urls.append(valid_video_urls_raw[i % len(valid_video_urls_raw)])
            subtitle_texts.append(all_subtitle_texts[i] if i < len(all_subtitle_texts) else "")

        valid_video_urls = video_urls

        # Durée de chaque segment = durée totale / nombre de scènes
        segment_duration = chosen_duration / nb_scenes
        total_duration = chosen_duration

        job_id = uuid.uuid4().hex
        job_dir = os.path.join(WORK_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        voice_url = (req.audio_url or "").strip()
        music_url = (req.music_url or "").strip()

        # ── ÉTAPE 1 : Télécharger la voix et mesurer sa durée réelle
        # La durée de la voix est la SEULE référence pour toute la vidéo.
        # Elle prime sur la durée choisie dans l'interface.
        voice_path = None
        real_total_duration = float(total_duration)  # fallback sans voix

        if voice_url:
            voice_path = os.path.join(job_dir, "voice.mp3")
            await download_file(voice_url, voice_path)
            measured = get_audio_duration(voice_path)
            if measured > 1.0:
                real_total_duration = measured

        # Durée de chaque segment vidéo = durée voix / nb scènes
        real_segment_duration = real_total_duration / nb_scenes

        # ── ÉTAPE 2 : Télécharger les vidéos sources
        raw_video_paths = []
        for i, url in enumerate(valid_video_urls):
            raw_path = os.path.join(job_dir, f"raw_{i}.mp4")
            await download_file(url, raw_path)
            raw_video_paths.append(raw_path)

        # ── ÉTAPE 3 : Normaliser chaque segment à la bonne durée
        norm_video_paths = []
        for i, raw_path in enumerate(raw_video_paths):
            norm_path = os.path.join(job_dir, f"seg_{i}.mp4")
            run_cmd([
                "ffmpeg", "-y",
                "-i", raw_path,
                "-t", str(real_segment_duration),
                "-vf", "scale=405:720:force_original_aspect_ratio=increase,crop=405:720,fps=30,format=yuv420p",
                "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-r", "30",
                norm_path
            ])
            norm_video_paths.append(norm_path)

        # ── ÉTAPE 4 : Concaténer les segments
        concat_list_path = os.path.join(job_dir, "concat.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for path in norm_video_paths:
                abs_path = os.path.abspath(path)
                f.write(f"file '{abs_path}'\n")

        # Vidéo concaténée brute (durée théorique = real_total_duration)
        stitched_raw_path = os.path.join(job_dir, "stitched_raw.mp4")
        run_cmd([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-r", "30",
            "-pix_fmt", "yuv420p", "-an",
            stitched_raw_path
        ])

        # ── ÉTAPE 5 : Vérifier la durée réelle de la vidéo concaténée
        # Si elle est plus courte que la voix (à cause d'arrondi ou de vidéos
        # sources trop courtes), on boucle la vidéo pour combler le manque.
        stitched_duration = get_audio_duration(stitched_raw_path)
        stitched_path = os.path.join(job_dir, "stitched.mp4")

        if stitched_duration < real_total_duration - 0.5:
            # La vidéo est plus courte que la voix → on la boucle
            run_cmd([
                "ffmpeg", "-y",
                "-stream_loop", "-1",
                "-i", stitched_raw_path,
                "-t", str(real_total_duration),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-r", "30",
                "-pix_fmt", "yuv420p", "-an",
                stitched_path
            ])
        else:
            # La vidéo est assez longue → on la coupe exactement
            run_cmd([
                "ffmpeg", "-y",
                "-i", stitched_raw_path,
                "-t", str(real_total_duration),
                "-c:v", "copy", "-an",
                stitched_path
            ])

        # ── ÉTAPE 6 : Préparer l'audio final (voix + musique optionnelle)
        final_audio_path = None

        if voice_url and voice_path:
            if music_url:
                music_path = os.path.join(job_dir, "music.mp3")
                await download_file(music_url, music_path)
                mixed_path = os.path.join(job_dir, "mixed.m4a")
                run_cmd([
                    "ffmpeg", "-y",
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
                # Pas de musique → on garde la voix telle quelle sans la couper
                final_audio_path = voice_path

        # ── ÉTAPE 7 : SRT synchronisé sur la durée réelle de la voix
        srt_path = os.path.join(job_dir, "subtitles.srt")
        nb_subtitles = len([t for t in subtitle_texts if t.strip()])
        srt_segment_duration = (
            real_total_duration / nb_subtitles if nb_subtitles > 0
            else real_segment_duration
        )
        write_srt(subtitle_texts, srt_segment_duration, srt_path)

        # ── ÉTAPE 8 : Rendu final avec sous-titres brûlés
        output_filename = f"{job_id}.mp4"
        output_path = os.path.join(VIDEO_DIR, output_filename)

        srt_escaped = escape_srt_path(os.path.abspath(srt_path))
        # Alignment=2  = centré horizontalement, ancrage en BAS du texte
        # MarginV=30   = 30px depuis le bas — valeur petite et stable
        # PlayResY=720 = dit à FFmpeg que la vidéo fait 720px de haut
        #                pour que MarginV soit interprété correctement
        subtitle_filter = (
            f"subtitles='{srt_escaped}':"
            "force_style='Alignment=2,MarginV=30,"
            "PlayResX=405,PlayResY=720,"
            "FontName=Arial,FontSize=18,Bold=1,"
            "PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,"
            "BorderStyle=3,Outline=2,Shadow=0,"
            "BackColour=&H99000000'"
        )

        if final_audio_path:
            run_cmd([
                "ffmpeg", "-y",
                "-i", stitched_path,
                "-i", final_audio_path,
                "-vf", subtitle_filter,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-r", "30",
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                # PAS de -shortest : la vidéo dure exactement comme la voix
                output_path
            ])
        else:
            run_cmd([
                "ffmpeg", "-y",
                "-i", stitched_path,
                "-vf", subtitle_filter,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-r", "30",
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
