from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI()

class RequestData(BaseModel):
    niche: str
    langue: str

def generate_titles(niche):
    hooks = [
        f"Tu ne croiras jamais ça sur {niche}",
        f"La vérité cachée sur {niche}",
        f"Voici pourquoi {niche} va changer ta vie",
        f"Les secrets de {niche} révélés",
        f"Ce que personne ne te dit sur {niche}"
    ]
    return random.sample(hooks, 3)

def generate_script(niche):
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

@app.get("/")
def home():
    return {"message": "SaaS Viral actif 🚀"}

@app.post("/generate")
def generate(data: RequestData):
    return {
        "titles": generate_titles(data.niche),
        "script": generate_script(data.niche)
    }
