#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-localai/localai:latest-aio-gpu-nvidia-cuda-12}"
PORT="${PORT:-8080}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "${ROOT_DIR}/models"

echo "Starting LocalAI..."
echo "  Image: ${IMAGE}"
echo "  Port : ${PORT}"
echo "  Models mount: ${ROOT_DIR}/models -> /models"

docker rm -f local-ai >/dev/null 2>&1 || true

docker run -ti --name local-ai \
  -p "${PORT}:8080" \
  --gpus all \
  -v "${ROOT_DIR}/models:/models" \
  "${IMAGE}"
