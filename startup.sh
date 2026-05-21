#!/bin/bash
set -euo pipefail
echo "=== ViralVidTech Vast.ai Startup ==="
cd /workspace

VENV_PYTHON="/workspace/LTX-2/.venv/bin/python3"

# ── Arrêter les serveurs existants ─────────────────────────────────────────
pkill -f runpod_server.py 2>/dev/null || true
pkill -f ltx_server.py   2>/dev/null || true
sleep 2

# ── Télécharger les serveurs en premier (toujours la dernière version) ────
echo "=== Téléchargement des serveurs depuis GitHub ==="
curl -fsSL -o /workspace/runpod_server.py \
    https://raw.githubusercontent.com/ViralVidTech/Viral---saas-/main/runpod_server.py
curl -fsSL -o /workspace/ltx_server.py \
    https://raw.githubusercontent.com/ViralVidTech/Viral---saas-/main/ltx_server.py

# ── LTX-2 : installer seulement si absent ─────────────────────────────────
# CONTRAINTE : ne jamais réinstaller si le venv existe déjà.
if [ -f "$VENV_PYTHON" ]; then
    echo "=== LTX-2 déjà installé — skip clone+sync ==="
else
    echo "=== Première installation LTX-2 ==="

    # Nettoyage des résidus
    pip uninstall -y ltx-pipelines ltx-core 2>/dev/null || true
    rm -rf /workspace/LTX-2
    rm -rf /workspace/wan2.2-animate
    rm -f  /workspace/ltx-2.3/ltx-2.3-22b-distilled.safetensors
    pip cache purge -q

    # Dépendances système
    pip install fastapi uvicorn python-multipart httpx peft decord librosa \
        einops timm imageio imageio-ffmpeg easydict dashscope diffusers ftfy \
        opencv-python-headless mistral_common huggingface_hub uv -q
    pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3+cu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" -q

    # Clone + vérification anti-nesting
    git clone https://github.com/Lightricks/LTX-2.git /workspace/LTX-2
    if find /workspace/LTX-2 -maxdepth 2 -type d -name "LTX-2" 2>/dev/null | grep -q .; then
        echo "ERREUR : structure imbriquée LTX-2/LTX-2 détectée — abandon"
        exit 1
    fi
    echo "Clone propre : $(ls /workspace/LTX-2)"

    cd /workspace/LTX-2
    uv sync --frozen
    /workspace/LTX-2/.venv/bin/pip install httpx fastapi uvicorn -q
    cd /workspace
fi

# ── Patch blocks.py (issue #216, idempotent — toujours appliqué) ──────────
echo "=== Patch blocks.py (issue #216) ==="
"$VENV_PYTHON" - <<'PYEOF'
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

# ── Checkpoints (téléchargement si absents) ────────────────────────────────
mkdir -p /workspace/ltx-2.3

if [ ! -f /workspace/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors ]; then
    echo "Téléchargement checkpoint principal LTX-2.3..."
    huggingface-cli download Lightricks/LTX-2.3 \
        ltx-2.3-22b-distilled-1.1.safetensors \
        --local-dir /workspace/ltx-2.3
fi

if [ ! -f /workspace/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors ]; then
    echo "Téléchargement spatial upscaler LTX-2.3..."
    huggingface-cli download Lightricks/LTX-2.3 \
        ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
        --local-dir /workspace/ltx-2.3
fi

if [ ! -d /workspace/gemma ] || [ -z "$(ls -A /workspace/gemma 2>/dev/null)" ]; then
    echo "Téléchargement Gemma text encoder..."
    huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
        --local-dir /workspace/gemma
fi

# ── Répertoire de sortie ───────────────────────────────────────────────────
mkdir -p /workspace/outputs

# ── Fonction de health check avec retry ───────────────────────────────────
_health_check() {
    local url="$1" name="$2" tries="${3:-12}" delay="${4:-5}"
    for i in $(seq 1 "$tries"); do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo "  ✓ $name opérationnel"
            return 0
        fi
        echo "  [$i/$tries] En attente de $name..."
        sleep "$delay"
    done
    echo "ERREUR : $name n'a pas répondu après $((tries * delay))s — logs ci-dessous"
    return 1
}

# ── a) Démarrer ltx_server EN PREMIER (port 8001) ─────────────────────────
echo ""
echo "=== Démarrage ltx_server (port 8001) ==="
nohup "$VENV_PYTHON" /workspace/ltx_server.py > /workspace/ltx.log 2>&1 &
LTX_PID=$!
echo "  PID=$LTX_PID"

# ── b) Attendre que ltx_server soit prêt (max 60s) ────────────────────────
_health_check "http://localhost:8001/health" "ltx_server" 12 5 || {
    echo "--- ltx.log ---"
    tail -30 /workspace/ltx.log
    exit 1
}

# ── c) Démarrer runpod_server (port 8000) ─────────────────────────────────
echo ""
echo "=== Démarrage runpod_server (port 8000) ==="
nohup python3 /workspace/runpod_server.py > /workspace/server.log 2>&1 &
RUNPOD_PID=$!
echo "  PID=$RUNPOD_PID"

# ── d) Attendre que runpod_server soit prêt (max 30s) ─────────────────────
_health_check "http://localhost:8000/health" "runpod_server" 6 5 || {
    echo "--- server.log ---"
    tail -15 /workspace/server.log
    exit 1
}

# ── Résumé final ──────────────────────────────────────────────────────────
echo ""
echo "=========================================="
LTX_HEALTH=$(curl -s http://localhost:8001/health)
LTX_STATUS=$(echo "$LTX_HEALTH" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("status","?"))' 2>/dev/null || echo "?")
LTX_DEVICE=$(echo "$LTX_HEALTH" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("device","?"))' 2>/dev/null || echo "?")
RUN_STATUS=$(curl -s http://localhost:8000/health | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("status","?"))' 2>/dev/null || echo "?")
echo "  ltx_server    :8001 → $LTX_STATUS | GPU: $LTX_DEVICE"
echo "  runpod_server :8000 → $RUN_STATUS"
echo "=========================================="
echo "=== Startup terminé ==="
