# models/

This folder is **for model persistence** so you don't download/install models repeatedly.

## Recommended workflow

1. Run LocalAI with this folder mounted to `/models` in the container:

### Linux/macOS
```bash
docker run -ti --name local-ai -p 8080:8080 --gpus all \
  -v "$(pwd)/models:/models" \
  localai/localai:latest-aio-gpu-nvidia-cuda-12
```

### Windows (PowerShell)
```powershell
docker run -ti --name local-ai -p 8080:8080 --gpus all `
  -v ${PWD}/models:/models `
  localai/localai:latest-aio-gpu-nvidia-cuda-12
```

2. Install TTS models (example: VibeVoice) once, then reuse:
```bash
# inside the container (or via local-ai CLI if available)
local-ai run models install vibevoice
```

3. List installed models:
```bash
curl http://localhost:8080/v1/models
```

> This repository ignores heavy model files by default (see `.gitignore`).
