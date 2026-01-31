from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf


def device_by_name(name: str, kind: str = "input") -> Optional[int]:
    if not name:
        return None
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        if name.lower() in str(d.get("name", "")).lower():
            if kind == "input" and d.get("max_input_channels", 0) > 0:
                return idx
            if kind == "output" and d.get("max_output_channels", 0) > 0:
                return idx
    return None


@dataclass
class Recorder:
    sample_rate: int = 16000
    channels: int = 1
    device: Optional[int] = None
    max_seconds: int = 30

    _frames: list = None
    _stream: Optional[sd.InputStream] = None
    _start_time: float = 0.0
    _running: bool = False

    def start(self) -> None:
        if self._running:
            return
        self._frames = []
        self._start_time = time.time()
        self._running = True

        def callback(indata, frames, time_info, status):
            if status:
                # Avoid noisy prints in callback; callers can handle debug if desired.
                pass
            if not self._running:
                return
            self._frames.append(indata.copy())
            if time.time() - self._start_time >= self.max_seconds:
                self.stop()

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=callback,
            device=self.device,
        )
        self._stream.start()

    def stop(self) -> bytes:
        if not self._running:
            return b""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logging.error(f"Error stopping audio stream: {e}")
            self._stream = None

        if not self._frames:
            audio = np.zeros((0, self.channels), dtype=np.float32)
        else:
            audio = np.concatenate(self._frames, axis=0)

        buf = io.BytesIO()
        # Whisper works well with 16k PCM16 WAV
        sf.write(buf, audio, self.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    @property
    def running(self) -> bool:
        return self._running


def decode_audio_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    """Decode common audio bytes into float32 numpy + sample rate.

    Preferred output is WAV from LocalAI /tts, but we try to decode other formats too.
    """
    # First try soundfile (handles WAV/FLAC/OGG depending on libsndfile build)
    try:
        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return arr, sr
    except Exception as e:
        logging.debug(f"soundfile decoding failed: {e}")

    # Fallback: pydub (requires ffmpeg installed on system)
    try:
        try:
            from pydub import AudioSegment  # type: ignore
        except ImportError:
            raise RuntimeError("pydub not installed. Please install with 'pip install pydub'.")

        seg = AudioSegment.from_file(io.BytesIO(data))
        sr = seg.frame_rate
        ch = seg.channels
        samples = np.array(seg.get_array_of_samples())
        # pydub interleaves channels
        if ch > 1:
            samples = samples.reshape((-1, ch))
        else:
            samples = samples.reshape((-1, 1))
        # normalize to float32
        maxv = float(1 << (8 * seg.sample_width - 1))
        arr = (samples.astype(np.float32) / maxv)
        return arr, sr
    except Exception as e:
        raise RuntimeError(
            f"Could not decode audio bytes: {e}. If your TTS returns mp3/opus, install ffmpeg "
            "and ensure pydub can find it, or switch LocalAI to return WAV (recommended)."
        ) from e


def play_audio(data: bytes, device: Optional[int] = None) -> None:
    arr, sr = decode_audio_bytes(data)
    sd.play(arr, sr, device=device)
    sd.wait()


def stop_playback() -> None:
    """Stop any ongoing sounddevice playback."""
    try:
        sd.stop()
    except Exception:
        pass
