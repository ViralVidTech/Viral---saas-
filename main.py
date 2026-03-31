from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modèle de requête pour /generate ---
class GenerateRequest(BaseModel):
    niche: str
    langue: str = "en"

# --- Route principale : sert l'interface HTML ---
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# --- Route API : génère titres + script ---
@app.post("/generate")
async def generate(req: GenerateRequest):
    niche = req.niche.strip() or "general"
    lang = req.langue

    # Titres et script de démonstration (remplace par ton appel OpenAI si besoin)
    titles = [
        f"How to make money with {niche} in 2025",
        f"The {niche} secret nobody talks about",
        f"I tried {niche} for 30 days — here's what happened",
    ]
    script = (
        f"Hook: Did you know that {niche} is changing everything right now?\n\n"
        f"Problem: Most people ignore {niche} and lose out on huge opportunities.\n\n"
        f"Solution: Here's exactly how you can leverage {niche} starting today.\n\n"
        f"CTA: Follow for more {niche} content every week!"
    )

    return {"titles": titles, "script": script}
