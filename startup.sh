#!/bin/bash
set -euo pipefail
echo "=== ViralVidTech Vast.ai Startup ==="
cd /workspace

# ── Arrêter les serveurs existants ─────────────────────────────────────────
pkill -f runpod_server.py 2>/dev/null || true
pkill -f ltx_server.py   2>/dev/null || true
sleep 2

# ── Nettoyage : mauvaises installations LTX et Wan Animate ────────────────
pip uninstall -y ltx-pipelines ltx-core 2>/dev/null || true
rm -rf /workspace/LTX-2
rm -rf /workspace/wan2.2-animate
# Supprimer l'ancien fichier mal nommé (sans -1.1) s'il existe encore
rm -f /workspace/ltx-2.3/ltx-2.3-22b-distilled.safetensors
pip cache purge -q

# ── Dépendances de base : Wan T2V + outils système ────────────────────────
pip install fastapi uvicorn python-multipart httpx peft decord librosa \
    einops timm imageio imageio-ffmpeg easydict dashscope diffusers ftfy \
    opencv-python-headless mistral_common huggingface_hub uv -q
pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3+cu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" -q

# ── LTX-2 — installation propre avec uv ───────────────────────────────────
# Vérification : aucun dossier imbriqué LTX-2/LTX-2 ne doit subsister
NESTED=$(find /workspace -maxdepth 4 -type d -path "*/LTX-2/LTX-2*" 2>/dev/null | head -5)
if [ -n "$NESTED" ]; then
    echo "Nettoyage des chemins imbriqués LTX-2 :"
    echo "$NESTED"
    # Remonter au parent commun et tout supprimer
    find /workspace -maxdepth 2 -type d -name "LTX-2" -exec rm -rf {} + 2>/dev/null || true
fi

git clone https://github.com/Lightricks/LTX-2.git /workspace/LTX-2

# Vérification anti-nesting post-clone
if find /workspace/LTX-2 -maxdepth 2 -type d -name "LTX-2" 2>/dev/null | grep -q .; then
    echo "ERREUR : structure imbriquée LTX-2/LTX-2 détectée après clone — abandon"
    exit 1
fi
echo "Clone propre confirmé : $(ls /workspace/LTX-2)"

cd /workspace/LTX-2
uv sync --frozen
# httpx / fastapi / uvicorn nécessaires pour ltx_server.py
/workspace/LTX-2/.venv/bin/pip install httpx fastapi uvicorn -q
cd /workspace

# ── Patch upstream LTX-2 bug #216 : multigpu module manquant ──────────────
# blocks.py line 74 importe ltx_pipelines.multigpu.delegating_builder qui
# n'existe pas dans le repo officiel (issue Lightricks/LTX-2#216, ouvert).
# On injecte un try/except idempotent directement dans la source ; l'install
# editable (uv sync workspace) fait que ce fichier IS le module — pas besoin
# de toucher le site-packages.
echo "=== Patch blocks.py (Lightricks/LTX-2 issue #216 — multigpu manquant) ==="
/workspace/LTX-2/.venv/bin/python3 - <<'PYEOF'
import pathlib, textwrap, sys
f = pathlib.Path(
    "/workspace/LTX-2/packages/ltx-pipelines/src"
    "/ltx_pipelines/utils/blocks.py"
)
if not f.exists():
    print(f"ERREUR : {f} introuvable", file=sys.stderr)
    sys.exit(1)

OLD = "from ltx_pipelines.multigpu.delegating_builder import DelegatingBuilder"
NEW = textwrap.dedent("""
    try:
        from ltx_pipelines.multigpu.delegating_builder import DelegatingBuilder
    except ImportError:
        # Stub — multigpu non livré par le repo officiel (issue #216).
        # DelegatingBuilder est utilisé uniquement comme annotation de type
        # dans DiffusionStage.__init__ ; il n'est jamais instancié en
        # mode single-GPU.
        from typing import Generic, TypeVar
        _T = TypeVar("_T")
        class DelegatingBuilder(Generic[_T]):  # type: ignore[no-redef]
            pass
""").strip()

src = f.read_text()
if OLD in src:
    f.write_text(src.replace(OLD, NEW))
    print("blocks.py patché — try/except multigpu injecté")
else:
    print("blocks.py déjà patché — rien à faire")
PYEOF

# ── Checkpoints officiels (téléchargement si absents) ─────────────────────
mkdir -p /workspace/ltx-2.3

if [ ! -f /workspace/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors ]; then
    echo "Téléchargement du checkpoint principal LTX-2.3..."
    huggingface-cli download Lightricks/LTX-2.3 \
        ltx-2.3-22b-distilled-1.1.safetensors \
        --local-dir /workspace/ltx-2.3
fi

if [ ! -f /workspace/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors ]; then
    echo "Téléchargement du spatial upscaler LTX-2.3..."
    huggingface-cli download Lightricks/LTX-2.3 \
        ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
        --local-dir /workspace/ltx-2.3
fi

if [ ! -d /workspace/gemma ] || [ -z "$(ls -A /workspace/gemma 2>/dev/null)" ]; then
    echo "Téléchargement Gemma text encoder..."
    huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
        --local-dir /workspace/gemma
fi

# ── Test minimal : vérification des imports LTX dans le venv ──────────────
echo "=== Test imports LTX ==="
/workspace/LTX-2/.venv/bin/python3 -c "
import ltx_core
import ltx_pipelines
print('LTX OK — ltx_core et ltx_pipelines importés avec succès')
"
if [ $? -ne 0 ]; then
    echo "ERREUR : imports LTX échoués — abandon du démarrage"
    exit 1
fi

# ── Répertoire de sortie ───────────────────────────────────────────────────
mkdir -p /workspace/outputs

# ── Téléchargement des serveurs depuis GitHub ─────────────────────────────
curl -fsSL -o /workspace/runpod_server.py \
    https://raw.githubusercontent.com/ViralVidTech/Viral---saas-/main/runpod_server.py
curl -fsSL -o /workspace/ltx_server.py \
    https://raw.githubusercontent.com/ViralVidTech/Viral---saas-/main/ltx_server.py

# ── Démarrage des serveurs ─────────────────────────────────────────────────
# port 8000 — Wan T2V (pipeline 3) + proxy transparent vers LTX
nohup python3 /workspace/runpod_server.py > /workspace/server.log 2>&1 &
sleep 3
# port 8001 — LTX-2.3 ; lancé avec le venv uv
nohup /workspace/LTX-2/.venv/bin/python3 /workspace/ltx_server.py > /workspace/ltx.log 2>&1 &
sleep 5

cat /workspace/server.log
cat /workspace/ltx.log
echo "=== Serveurs démarrés (8000 = Wan T2V + proxy LTX | 8001 = LTX 2.3) ==="
