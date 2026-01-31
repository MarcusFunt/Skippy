# Troubleshooting

## LocalAI not reachable
- Make sure LocalAI is running and exposed at the same base URL in `config/default.toml`.
- Test:
  ```bash
  curl http://localhost:8080/v1/models
  ```

## No audio input / wrong microphone
- Set `audio.device_in` in `config/default.toml` to a substring of your device name.
- On Windows, some devices require exclusive mode disabled in Sound settings.

## No audio output
- Set `audio.device_out` similarly.

## TTS plays garbage / decode fails
Skippy assumes the `/tts` endpoint returns WAV-like audio. If your TTS backend returns mp3/opus:

1) Install **ffmpeg** and ensure it is in PATH
2) Skippy will use `pydub` as a fallback decoder

Or switch to `/tts` outputting WAV by changing your LocalAI TTS backend/config.

## Streaming looks odd
Some LocalAI builds historically streamed characters instead of tokens. If streaming looks broken, disable it:
- Uncheck **Stream reply**.

## High latency
- Use a smaller local LLM (quantized gguf) and/or reduce context.
- Prefer a lightweight TTS (Piper) if your chosen TTS is heavy.
- Keep STT to `tiny` or `base` Whisper for speed if needed.
