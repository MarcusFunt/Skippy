from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


@dataclass
class LocalAIConfig:
    base_url: str = "http://localhost:8080"
    timeout_s: int = 180


@dataclass
class ModelsConfig:
    llm: str = "gpt-4"
    stt: str = "whisper-1"
    tts: str = "vibevoice"
    tts_endpoint: str = "tts"  # "tts" or "openai"


@dataclass
class TTSConfig:
    extra_params: Dict[str, Any] = field(default_factory=dict)
    response_format: str = "wav"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    device_in: str = ""
    device_out: str = ""
    ptt_key: str = "SHIFT_R"
    max_record_seconds: int = 30


@dataclass
class UIConfig:
    theme: str = "dark"
    font_size: int = 17
    show_debug: bool = False


@dataclass
class PersonaConfig:
    persona_file: str = "config/personas/engineer.md"


@dataclass
class AppConfig:
    localai: LocalAIConfig = field(default_factory=LocalAIConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)

    root_dir: Path = Path.cwd()
    config_path: Optional[Path] = None


def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_config(path: str | Path) -> AppConfig:
    p = Path(path).expanduser().resolve()
    try:
        content = p.read_text(encoding="utf-8")
        data = tomllib.loads(content)
    except FileNotFoundError:
        print(f"Config file not found: {p}. Using defaults.")
        data = {}
    except Exception as e:
        print(f"Error loading config {p}: {e}. Using defaults.")
        data = {}

    cfg = AppConfig()
    cfg.root_dir = p.parent
    cfg.config_path = p

    cfg.localai.base_url = _get(data, "localai", "base_url", default=cfg.localai.base_url)
    cfg.localai.timeout_s = int(_get(data, "localai", "timeout_s", default=cfg.localai.timeout_s))

    cfg.models.llm = _get(data, "models", "llm", default=cfg.models.llm)
    cfg.models.stt = _get(data, "models", "stt", default=cfg.models.stt)
    cfg.models.tts = _get(data, "models", "tts", default=cfg.models.tts)
    cfg.models.tts_endpoint = _get(data, "models", "tts_endpoint", default=cfg.models.tts_endpoint)

    cfg.tts.extra_params = dict(_get(data, "tts", "extra_params", default=cfg.tts.extra_params) or {})
    cfg.tts.response_format = _get(data, "tts", "response_format", default=cfg.tts.response_format)

    cfg.audio.sample_rate = int(_get(data, "audio", "sample_rate", default=cfg.audio.sample_rate))
    cfg.audio.channels = int(_get(data, "audio", "channels", default=cfg.audio.channels))
    cfg.audio.device_in = str(_get(data, "audio", "device_in", default=cfg.audio.device_in))
    cfg.audio.device_out = str(_get(data, "audio", "device_out", default=cfg.audio.device_out))
    cfg.audio.ptt_key = str(_get(data, "audio", "ptt_key", default=cfg.audio.ptt_key))
    cfg.audio.max_record_seconds = int(_get(data, "audio", "max_record_seconds", default=cfg.audio.max_record_seconds))

    cfg.ui.theme = str(_get(data, "ui", "theme", default=cfg.ui.theme))
    cfg.ui.font_size = int(_get(data, "ui", "font_size", default=cfg.ui.font_size))
    cfg.ui.show_debug = bool(_get(data, "ui", "show_debug", default=cfg.ui.show_debug))

    cfg.persona.persona_file = str(_get(data, "persona", "persona_file", default=cfg.persona.persona_file))

    return cfg


def read_persona_text(cfg: AppConfig) -> str:
    persona_path = (cfg.root_dir / cfg.persona.persona_file).resolve()
    if persona_path.exists():
        return persona_path.read_text(encoding="utf-8").strip()
    # Fall back to minimal persona if file missing
    return "You are Skippy, a helpful local voice assistant. Be concise and practical."
