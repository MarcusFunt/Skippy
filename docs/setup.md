# Setup

## 0) Prereqs

- Python 3.10+
- Docker (recommended) + NVIDIA GPU support (for LocalAI on your GTX 1070)
- A running LocalAI server on `http://localhost:8080`

## 1) Start LocalAI with persistent models

From the repo root:

### Linux/macOS
```bash
bash scripts/run_localai_docker.sh
```

### Windows (PowerShell)
```powershell
./scripts/run_localai_docker.ps1
```

This mounts `./models` into the container as `/models`, so installed models persist across restarts.

## 2) Install models (once)

Inside the LocalAI container, install what you want. A good default is:

- LLM (already present in AIO as `gpt-4` mapping; you can change)
- STT: Whisper (`whisper-1`)
- TTS: `vibevoice` (low-latency, modern) or `piper` (very light)

Example (inside container):
```bash
local-ai run models install vibevoice
```

You can list installed models:
```bash
curl http://localhost:8080/v1/models
```

## 3) Install Skippy

From the repo root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## 4) Run Skippy (GUI)

```bash
python -m skippy --config config/default.toml
```

## 5) Push-to-talk

- Default hotkey: **Right Shift** (`SHIFT_R`)
- You can also hold the on-screen button.

Change hotkey in `config/default.toml`:

```toml
[audio]
ptt_key = "SPACE"  # SHIFT_R, SPACE, F9, etc.
```
