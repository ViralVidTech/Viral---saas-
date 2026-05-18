#!/bin/bash
echo "=== ViralVidTech Vast.ai Startup ==="
cd /workspace
pkill -f runpod_server.py 2>/dev/null
sleep 2
pip install fastapi uvicorn python-multipart httpx peft decord librosa einops timm imageio imageio-ffmpeg easydict dashscope diffusers ftfy opencv-python-headless mistral_common -q
pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3+cu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" -q
mkdir -p /workspace/outputs
curl -o /workspace/runpod_server.py https://raw.githubusercontent.com/ViralVidTech/Viral---saas-/main/runpod_server.py
nohup python3 /workspace/runpod_server.py > /workspace/server.log 2>&1 &
sleep 5
cat /workspace/server.log
echo "=== Serveur démarré ==="
