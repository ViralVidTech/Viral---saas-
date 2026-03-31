from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import random
import os

app = FastAPI()

origins = [
    "https://viralvidtech-frontend.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestData(BaseModel):
    niche: str
    langue: str

def generate_titles(niche: str):
    hooks = [
        f"Tu ne croiras jamais ça sur {niche}",
        f"La vérité cachée sur {niche}",
        f"Voici pourquoi {niche} va changer ta vie",
        f"Les secrets de {niche} révélés",
        f"Ce que personne ne te dit sur {niche}"
    ]
    return random.sample(hooks, 3)

def generate_script(niche: str):
    return f"""
Hook : Tu fais cette erreur avec {niche} sans le savoir...

Développement :
La majorité des gens ne comprennent pas à quel point {niche} peut impacter leur vie.
Mais aujourd’hui, tu vas découvrir une vérité choquante.

Conclusion :
Si tu veux réussir avec {niche}, commence dès maintenant.

CTA :
Abonne-toi pour plus de conseils.
"""

def create_video(filename: str = "video.mp4"):
    # Import déplacé ici pour éviter que Render plante au démarrage
    from moviepy.editor import ColorClip

    duration = 30
    clip = ColorClip(size=(720, 1280), color=(20, 20, 20), duration=duration)

    clip.write_videofile(
        filename,
        fps=24,
        codec="libx264",
        audio=False,
        logger=None
    )

    return filename

@app.get("/")
def home():
    return {"message": "SaaS Viral actif"}

@app.post("/generate")
def generate(data: RequestData):
    return {
        "titles": generate_titles(data.niche),
        "script": generate_script(data.niche)
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/create-video")
def create_video_endpoint(data: RequestData):
    file_path = "output.mp4"

    try:
        create_video(file_path)
        return FileResponse(
            path=file_path,
            media_type="video/mp4",
            filename="video.mp4"
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "video_creation_failed",
                "detail": str(e)
            }
        )
