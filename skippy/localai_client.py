from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests
from sseclient import SSEClient  # type: ignore


@dataclass
class LocalAIClient:
    base_url: str
    timeout_s: int = 180

    def _url(self, path: str) -> str:
        return self.base_url.rstrip("/") + path

    def list_models(self) -> List[str]:
        try:
            r = requests.get(self._url("/v1/models"), timeout=self.timeout_s)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to list models from LocalAI: {e}")
        data = r.json()
        # OpenAI-like: {"data":[{"id":"..."}]}
        models = []
        for item in data.get("data", []):
            mid = item.get("id")
            if mid:
                models.append(mid)
        return sorted(set(models))

    # ---- STT ----
    def transcribe(self, wav_bytes: bytes, model: str, language: Optional[str] = None) -> str:
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data: Dict[str, Any] = {"model": model}
        if language:
            data["language"] = language
        try:
            r = requests.post(self._url("/v1/audio/transcriptions"), files=files, data=data, timeout=self.timeout_s)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"STT transcription failed: {e}")
        out = r.json()
        # most implementations return {"text": "..."}
        return (out.get("text") or "").strip()

    # ---- LLM ----
    def chat(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        payload = {"model": model, "messages": messages, "temperature": temperature}
        try:
            r = requests.post(self._url("/v1/chat/completions"), json=payload, timeout=self.timeout_s)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"LLM chat failed: {e}")
        out = r.json()
        return out["choices"][0]["message"]["content"].strip()

    def chat_stream(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7) -> Iterable[str]:
        payload = {"model": model, "messages": messages, "temperature": temperature, "stream": True}
        headers = {"Accept": "text/event-stream"}
        try:
            r = requests.post(self._url("/v1/chat/completions"), json=payload, headers=headers, stream=True, timeout=self.timeout_s)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"LLM chat stream failed: {e}")

        with r:
            client = SSEClient(r)
            for event in client.events():
                if not event.data:
                    continue
                if event.data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(event.data)
                except json.JSONDecodeError:
                    continue
                # OpenAI-style delta
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield text

    # ---- TTS ----
    def tts_raw(self, tts_model: str, text: str, extra_params: Optional[Dict[str, Any]] = None) -> bytes:
        payload: Dict[str, Any] = {"model": tts_model, "input": text}
        if extra_params:
            payload.update(extra_params)
        try:
            r = requests.post(self._url("/tts"), json=payload, timeout=self.timeout_s)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"TTS (raw) failed: {e}")
        return r.content

    def tts_openai(self, tts_model: str, text: str, voice: str = "alloy", response_format: Optional[str] = None) -> bytes:
        payload: Dict[str, Any] = {"model": tts_model, "input": text, "voice": voice}
        if response_format:
            payload["response_format"] = response_format
        try:
            r = requests.post(self._url("/v1/audio/speech"), json=payload, timeout=self.timeout_s)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"TTS (openai) failed: {e}")
        return r.content
