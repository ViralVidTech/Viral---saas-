from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import base64
import json
import httpx
import subprocess
import shutil
import asyncio
import threading

app = FastAPI()

# Dictionnaire en mémoire pour suivre les jobs de rendu vidéo
# { job_id: {"status": "processing"|"done"|"failed", "video_url": "...", "error": "..."} }
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
    # 5 vidéos par scène × 8 scènes = 40 URLs vidéo
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


async def download_file(url: str, dest_path: str, retries: int = 3, delay: float = 2.0):
    """
    Télécharge un fichier avec tentatives automatiques.
    Pour les vidéos stock, on limite à 15MB max pour accélérer le téléchargement.
    """
    MAX_VIDEO_BYTES = 15 * 1024 * 1024  # 15 MB max par vidéo
    last_error = None
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                follow_redirects=True
            ) as client:
                # Streaming pour limiter la taille et accélérer
                async with client.stream("GET", url) as response:
                    if response.status_code in (502, 503, 504):
                        await asyncio.sleep(delay * (attempt + 1))
                        continue
                    response.raise_for_status()
                    downloaded = 0
                    with open(dest_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            f.write(chunk)
                            downloaded += len(chunk)
                            # Limiter la taille pour les vidéos stock
                            if downloaded >= MAX_VIDEO_BYTES:
                                break
                return  # succès
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                await asyncio.sleep(delay * (attempt + 1))
    raise last_error or RuntimeError(f"Échec téléchargement après {retries} tentatives: {url}")


async def download_audio_file(url: str, dest_path: str, retries: int = 4, delay: float = 3.0):
    """Téléchargement audio sans limite de taille."""
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

    # Avancer tous les sous-titres de 0.5s pour compenser
    # le délai naturel entre l'estimation et la prononciation réelle
    ADVANCE = 0.6

    for text in all_texts:
        words = text.split()

        # Découper en blocs de WORDS_PER_BLOCK mots
        for j in range(0, len(words), WORDS_PER_BLOCK):
            block = words[j:j + WORDS_PER_BLOCK]
            block_text = " ".join(block)
            block_word_count = len(block)

            duration = block_word_count * seconds_per_word
            start = max(0.0, current_time - ADVANCE)
            end   = max(start + 0.1, start + duration - 0.05)

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

        video_urls = [""] * 32

        # Construire les requêtes de recherche pour chaque scène
        # On prend les 3 premiers mots significatifs de chaque scène
        # pour trouver des vidéos qui correspondent au contenu
        def extract_keywords(scene_text: str, fallback: str, scene_key: str = "") -> str:
            """
            Extrait les mots-clés les plus visuels d'une scène.
            Privilégie les noms et adjectifs concrets qui donnent
            de bonnes images de stock (personnes, lieux, actions).
            """
            if not scene_text:
                return fallback
            words = scene_text.split()
            stop_words = {
                # Français
                "le","la","les","un","une","des","de","du","et","en","que","qui",
                "pour","par","sur","dans","avec","est","sont","mais","donc","car",
                "tout","tous","toute","cette","cela","plus","très","bien","aussi",
                "comme","même","fait","peut","faut","doit","avoir","être","faire",
                # Anglais
                "the","a","an","is","are","you","your","to","of","in","and","or",
                "it","this","that","we","our","my","not","can","will","all","if",
                "have","has","been","with","they","their","from","but","when","how"
            }
            # Garder uniquement les mots significatifs de plus de 4 caractères
            keywords = [w.strip(".,!?;:") for w in words
                       if len(w.strip(".,!?;:")) > 4
                       and w.lower().strip(".,!?;:") not in stop_words]

            # Ajouter la niche comme contexte pour ancrer la recherche
            if keywords:
                # Combiner 2 mots-clés de la scène + la niche
                query = " ".join(keywords[:2]) + " " + fallback
                return query.strip()
            return fallback

        # Préparer une requête par scène — combinaison mots-clés + niche
        scene_queries = []
        for key in scene_keys:
            if scenes[key]:
                query = extract_keywords(scenes[key], niche, key)
                scene_queries.append(query)

        # Compléter avec la niche si on n'a pas assez de scènes
        while len(scene_queries) < 8:
            scene_queries.append(niche)

        async def fetch_pexels_videos(client, query: str, count: int = 2) -> list:
            """Cherche plusieurs vidéos Pexels pour une requête donnée."""
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
            """Cherche plusieurs vidéos Pixabay pour une requête donnée."""
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

        # video_urls : 16 slots — 2 vidéos par scène (slot 0-1 = scène1, 2-3 = scène2...)
        video_urls = [""] * 32

        if PEXELS_API_KEY or PIXABAY_API_KEY:
            try:
                async with httpx.AsyncClient(timeout=45) as client:
                    for i, query in enumerate(scene_queries[:8]):
                        slot = i * 5  # chaque scène occupe 5 slots consécutifs
                        collected = []

                        # 1. Pexels avec mots-clés de la scène
                        if PEXELS_API_KEY:
                            collected += await fetch_pexels_videos(client, query, 5)

                        # 2. Pixabay si pas assez
                        if PIXABAY_API_KEY and len(collected) < 5:
                            collected += await fetch_pixabay_videos(client, query, 5 - len(collected))

                        # 3. Fallback niche si toujours pas assez
                        if PEXELS_API_KEY and len(collected) < 5 and query != niche:
                            collected += await fetch_pexels_videos(client, niche, 5 - len(collected))

                        # Remplir les 5 slots de cette scène
                        for j in range(5):
                            if j < len(collected):
                                video_urls[slot + j] = collected[j]
                            elif collected:
                                video_urls[slot + j] = collected[j % len(collected)]
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
    """
    Transforme un texte en SSML avec une <mark> devant chaque mot.
    Retourne (ssml_string, liste_de_mots).
    Google TTS retournera ensuite le timestamp exact de chaque mark.
    """
    words = text.strip().split()
    parts = []
    for i, word in enumerate(words):
        parts.append(f'<mark name="w{i}"/>{word}')
    ssml = "<speak>" + " ".join(parts) + "</speak>"
    return ssml, words


def build_srt_from_timepoints(words: list, timepoints: list, out_path: str, words_per_block: int = 5):
    """
    Construit le fichier SRT à partir des timestamps EXACTS
    fournis par Google TTS pour chaque mot.
    """
    # Construire un dict : index_mot → temps_début (en secondes)
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

    # Durée totale estimée = dernier timestamp + durée moyenne par mot
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

        # Début du bloc = timestamp du premier mot du bloc
        start = mark_times.get(j, None)
        if start is None:
            continue

        # Fin du bloc = timestamp du premier mot du prochain bloc (ou fin estimée)
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

    # Texte simple pour toutes les voix (Chirp3, Neural2, Standard, Wavenet)
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

        # Sauvegarder le fichier audio
        filename = f"{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(audio_content))
        audio_url = f"{PUBLIC_BASE_URL}/audio/{filename}"

        # Sauvegarder les mots du texte pour la synchronisation des sous-titres
        words = req.text.strip().split()
        sync_filename = filename.replace(".mp3", "_sync.json")
        sync_filepath = os.path.join(AUDIO_DIR, sync_filename)
        with open(sync_filepath, "w", encoding="utf-8") as f:
            json.dump({"words": words, "timepoints": []}, f)

        sync_url = f"{PUBLIC_BASE_URL}/audio/{sync_filename}"

        return {
            "success": True,
            "audio_url": audio_url,
            "sync_url": sync_url,
            "filename": filename
        }
    except Exception as e:
        return {"error": f"Erreur TTS: {str(e)}"}


# FFMPEG: CRÉER LA VIDÉO FINALE
async def _process_video(job_id: str, req: VideoRequest):
    """Traitement vidéo en arrière-plan — appelé dans un thread séparé."""
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

        # Récupérer les 40 URLs vidéo (5 par scène × 8 scènes max)
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

        valid_video_urls_raw = [u for u in all_video_urls if u]
        if not valid_video_urls_raw:
            return JSONResponse(status_code=400, content={"error": "Aucune vidéo n'a été fournie"})

        # Construire la liste finale des clips vidéo :
        # 2 clips par scène — chaque clip dure segment_duration / 2
        # Exemple pour 4 scènes de 9s : 8 clips de 4.5s chacun
        CLIPS_PER_SCENE = 4
        nb_clips_total = nb_scenes * CLIPS_PER_SCENE

        video_urls = []
        subtitle_texts = []

        for i in range(nb_scenes):
            scene_text = all_subtitle_texts[i] if i < len(all_subtitle_texts) else ""
            for j in range(CLIPS_PER_SCENE):
                slot = i * CLIPS_PER_SCENE + j
                if slot < len(all_video_urls) and all_video_urls[slot]:
                    video_urls.append(all_video_urls[slot])
                elif valid_video_urls_raw:
                    # Recycle si pas de vidéo disponible pour ce slot
                    video_urls.append(valid_video_urls_raw[slot % len(valid_video_urls_raw)])
                # Le texte du sous-titre ne s'affiche que sur le 1er clip de chaque scène
                subtitle_texts.append(scene_text if j == 0 else "")

        valid_video_urls = video_urls

        # Durée de chaque clip = durée totale / nb clips totaux
        segment_duration = chosen_duration / nb_clips_total
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
            await download_audio_file(voice_url, voice_path)
            measured = get_audio_duration(voice_path)
            if measured > 1.0:
                real_total_duration = measured

        # Durée de chaque CLIP vidéo = durée voix / nb clips totaux (5 par scène)
        real_segment_duration = real_total_duration / nb_clips_total

        # ── ÉTAPE 2 : Télécharger les vidéos EN PARALLÈLE
        raw_video_paths = [os.path.join(job_dir, f"raw_{i}.mp4")
                           for i in range(len(valid_video_urls))]

        await asyncio.gather(*[
            download_file(url, path)
            for url, path in zip(valid_video_urls, raw_video_paths)
        ])

        # ── ÉTAPE 3 : Normaliser chaque segment EN PARALLÈLE
        # On utilise un thread pool pour lancer plusieurs FFmpeg simultanément
        norm_video_paths = [os.path.join(job_dir, f"seg_{i}.mp4")
                            for i in range(len(raw_video_paths))]

        def normalize_one(args):
            raw_path, norm_path = args
            run_cmd([
                "ffmpeg", "-y",
                "-i", raw_path,
                "-t", str(real_segment_duration),
                "-vf", "scale=405:720:force_original_aspect_ratio=increase,crop=405:720,fps=25,format=yuv420p",
                "-an",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                "-r", "25",
                norm_path
            ])

        # Lancer max 4 normalisations en parallèle pour ne pas surcharger le CPU
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(normalize_one, zip(raw_video_paths, norm_video_paths)))

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
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-r", "25",
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
                await download_audio_file(music_url, music_path)
                mixed_path = os.path.join(job_dir, "mixed.m4a")
                run_cmd([
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
                # Pas de musique → on garde la voix telle quelle sans la couper
                final_audio_path = voice_path

        # ── ÉTAPE 7 : SRT avec synchronisation EXACTE via timepoints Google TTS
        srt_path = os.path.join(job_dir, "subtitles.srt")
        srt_built = False

        # Essayer d'utiliser les timepoints exacts de Google TTS
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

        # Fallback : estimation par mots si pas de timepoints disponibles
        if not srt_built:
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
            "force_style='Alignment=2,MarginV=70,"
            "PlayResX=405,PlayResY=720,"
            "FontName=Arial,FontSize=24,Bold=1,"
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
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-r", "25",
                "-c:a", "aac", "-b:a", "128k",
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
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-r", "25",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ])

        # Nettoyage du dossier temporaire de travail
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
    """Lance le rendu vidéo en arrière-plan et retourne immédiatement un job_id."""
    if not PUBLIC_BASE_URL:
        return JSONResponse(status_code=400, content={"error": "PUBLIC_BASE_URL manquante"})
    if not ffmpeg_exists():
        return JSONResponse(status_code=500, content={"error": "FFmpeg non installé"})

    job_id = uuid.uuid4().hex
    VIDEO_JOBS[job_id] = {"status": "processing"}

    # Lancer le traitement dans un thread séparé pour ne pas bloquer
    loop = asyncio.get_event_loop()
    asyncio.ensure_future(_process_video(job_id, req))

    return JSONResponse(status_code=200, content={
        "success": True,
        "job_id": job_id,
        "message": "Rendu démarré. Vérifiez /video-status/{job_id}"
    })


@app.get("/video-status/{job_id}")
async def video_status(job_id: str):
    """Retourne le statut d'un job de rendu vidéo."""
    job = VIDEO_JOBS.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job introuvable"})
    return JSONResponse(status_code=200, content=job)
