# API Notes (LocalAI)

Skippy uses these endpoints:

- STT: `POST /v1/audio/transcriptions`
- Chat: `POST /v1/chat/completions` (optionally streaming)
- TTS: `POST /tts` (default) OR `POST /v1/audio/speech` (OpenAI-style)

You can list models:
- `GET /v1/models`

If you change model names in LocalAI, update `config/default.toml`.
