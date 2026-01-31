#!/usr/bin/env bash
set -euo pipefail

# Default to GPU image if not specified
IMAGE="${1:-}"
PORT="${PORT:-8080}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "${ROOT_DIR}/models"

# Detect if NVIDIA GPU is available and docker-nvidia-container is set up
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  GPU_FLAGS="--gpus all"
  DEFAULT_IMAGE="localai/localai:latest-aio-gpu-nvidia-cuda-12"
else
  GPU_FLAGS=""
  DEFAULT_IMAGE="localai/localai:latest-aio-cpu"
  echo "No NVIDIA GPU detected or nvidia-smi failed. Falling back to CPU."
fi

IMAGE="${IMAGE:-$DEFAULT_IMAGE}"

echo "Starting LocalAI..."
echo "  Image: ${IMAGE}"
echo "  Port : ${PORT}"
echo "  GPU  : ${GPU_FLAGS:-None}"
echo "  Models mount: ${ROOT_DIR}/models -> /models"

docker rm -f local-ai >/dev/null 2>&1 || true

docker run -ti --name local-ai \
  -p "${PORT}:8080" \
  ${GPU_FLAGS} \
  -v "${ROOT_DIR}/models:/models" \
  "${IMAGE}"
