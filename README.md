# Skippy — LocalAI Speech-to-Speech (Push-to-Talk)

A **local** speech-to-speech desktop app:
**Hold-to-talk → transcribe (STT) → LLM → TTS → play audio**, with a polished **DearPyGUI** interface.

- Backend: **LocalAI** (runs on your machine)
- UI: **DearPyGUI**
- Input: push-to-talk hotkey + on-screen button
- Output: local TTS playback
- Persistence: `./models` folder for LocalAI model caching (no repeated installs)

---

## Quick start

### 1) Start LocalAI with persistent models

**Linux/macOS**
```bash
bash scripts/run_localai_docker.sh
```

**Windows (PowerShell)**
```powershell
./scripts/run_localai_docker.ps1
```

This mounts `./models` into LocalAI as `/models`.

### 2) Install models (once)

Inside the LocalAI container, install a TTS model (example: VibeVoice):

```bash
local-ai run models install vibevoice
```

List installed models:

```bash
curl http://localhost:8080/v1/models
```

### 3) Install and run Skippy

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python -m skippy --config config/default.toml
```

---

## Controls

- **PTT hotkey**: `SHIFT_R` (Right Shift) by default  
- Or hold the **“Hold to talk”** button in the UI

Change in `config/default.toml`:

```toml
[audio]
ptt_key = "SPACE"
```

---

## Configuration

- `config/default.toml` — LocalAI URL, model names, audio devices, hotkey, persona file
- `config/personas/*.md` — editable personalities (system prompt)

---

## Docs

- [Setup](docs/setup.md)
- [Troubleshooting](docs/troubleshooting.md)
- [API Notes](docs/api_notes.md)

---

## Project structure

```
Skippy/
  skippy/               # app code
  config/               # config + personas
  models/               # persistent LocalAI models (ignored by git)
  scripts/              # helper scripts (docker + run)
  docs/                 # documentation
```

---

## Notes

- Skippy defaults to LocalAI's `/tts` endpoint (often WAV-like output).
  If your TTS returns mp3/opus, install **ffmpeg** so the fallback decoder can handle it.
- If streaming looks weird, turn off **Stream reply**.
