from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg

from .audio import Recorder, device_by_name, play_audio
from .config import AppConfig, load_config, read_persona_text
from .localai_client import LocalAIClient


def _keycode(name: str) -> int:
    # Minimal mapping for common push-to-talk choices
    n = name.strip().upper()
    mapping = {
        "SPACE": dpg.mvKey_Spacebar,
        "ENTER": dpg.mvKey_Return,
        "SHIFT_L": dpg.mvKey_LShift,
        "SHIFT_R": dpg.mvKey_RShift,
        "CTRL_L": dpg.mvKey_LControl,
        "CTRL_R": dpg.mvKey_RControl,
        "ALT_L": dpg.mvKey_LAlt,
        "ALT_R": dpg.mvKey_RAlt,
        "F6": dpg.mvKey_F6,
        "F7": dpg.mvKey_F7,
        "F8": dpg.mvKey_F8,
        "F9": dpg.mvKey_F9,
        "F10": dpg.mvKey_F10,
        "F11": dpg.mvKey_F11,
        "F12": dpg.mvKey_F12,
    }
    return mapping.get(n, dpg.mvKey_RShift)


@dataclass
class ChatState:
    messages: List[Dict[str, str]] = field(default_factory=list)
    temperature: float = 0.7
    stream: bool = False
    busy: bool = False
    cancel_flag: bool = False


class SkippyApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.client = LocalAIClient(cfg.localai.base_url, timeout_s=cfg.localai.timeout_s)

        self.persona_text = read_persona_text(cfg)
        self.state = ChatState(messages=[{"role": "system", "content": self.persona_text}])

        self.ui_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        self.recorder = Recorder(
            sample_rate=cfg.audio.sample_rate,
            channels=cfg.audio.channels,
            device=device_by_name(cfg.audio.device_in, kind="input"),
            max_seconds=cfg.audio.max_record_seconds,
        )
        self.out_device = device_by_name(cfg.audio.device_out, kind="output")

        # DPG item ids
        self.ids: Dict[str, int] = {}

    # ---------------- UI helpers ----------------
    def _enqueue(self, op: str, payload: Any) -> None:
        self.ui_queue.put((op, payload))

    def _add_log(self, who: str, text: str) -> None:
        # Append message widget to log child
        self._enqueue("add_log", {"who": who, "text": text})

    def _set_status(self, text: str) -> None:
        self._enqueue("status", text)

    def _set_busy(self, busy: bool) -> None:
        self._enqueue("busy", busy)

    def _set_assistant_stream_text(self, text: str) -> None:
        self._enqueue("assistant_stream", text)

    # ---------------- Model ops ----------------
    def refresh_models(self) -> None:
        def worker():
            try:
                models = self.client.list_models()
                self._enqueue("models_list", models)
                self._set_status(f"Loaded {len(models)} models from LocalAI")
            except Exception as e:
                self._set_status(f"Model list error: {e}")
        threading.Thread(target=worker, daemon=True).start()

    # ---------------- Audio / PTT ----------------
    def ptt_start(self) -> None:
        if self.state.busy:
            return
        self.recorder.start()
        self._set_status("Recording… release to send")

    def ptt_stop(self) -> None:
        if self.state.busy:
            return
        wav_bytes = self.recorder.stop()
        if not wav_bytes:
            self._set_status("No audio captured")
            return

        self.state.busy = True
        self.state.cancel_flag = False
        self._set_busy(True)
        self._set_status("Transcribing…")

        threading.Thread(target=self._process_turn, args=(wav_bytes,), daemon=True).start()

    def cancel(self) -> None:
        self.state.cancel_flag = True
        self._set_status("Cancel requested…")

    def _process_turn(self, wav_bytes: bytes) -> None:
        try:
            user_text = self.client.transcribe(wav_bytes, model=self.cfg.models.stt)
            user_text = (user_text or "").strip()
            if not user_text:
                self._set_status("Transcription was empty")
                return

            # Add user message
            self._add_log("You", user_text)
            self.state.messages.append({"role": "user", "content": user_text})

            # LLM
            self._set_status("Thinking…")
            assistant_text = ""

            if self.state.stream:
                self._enqueue("assistant_new_stream", None)
                for chunk in self.client.chat_stream(
                    model=self.cfg.models.llm,
                    messages=self.state.messages,
                    temperature=self.state.temperature,
                ):
                    if self.state.cancel_flag:
                        break
                    assistant_text += chunk
                    self._set_assistant_stream_text(assistant_text)
                assistant_text = assistant_text.strip()
                if assistant_text:
                    # finalize log entry
                    self._enqueue("assistant_finalize_stream", assistant_text)
            else:
                assistant_text = self.client.chat(
                    model=self.cfg.models.llm,
                    messages=self.state.messages,
                    temperature=self.state.temperature,
                )
                if self.state.cancel_flag:
                    return
                self._add_log("Skippy", assistant_text)

            if not assistant_text:
                self._set_status("No assistant response")
                return

            self.state.messages.append({"role": "assistant", "content": assistant_text})

            # TTS
            self._set_status("Speaking…")
            tts_bytes = self._tts_bytes(assistant_text)

            if self.state.cancel_flag:
                return

            play_audio(tts_bytes, device=self.out_device)
            self._set_status("Ready")
        except Exception as e:
            self._set_status(f"Error: {e}")
        finally:
            self.state.busy = False
            self._set_busy(False)

    def _tts_bytes(self, text: str) -> bytes:
        if self.cfg.models.tts_endpoint.lower() == "openai":
            # OpenAI-compatible endpoint (may return wav depending on LocalAI version/config)
            voice = str(self.cfg.tts.extra_params.get("voice", "alloy"))
            fmt = self.cfg.tts.response_format or None
            return self.client.tts_openai(self.cfg.models.tts, text=text, voice=voice, response_format=fmt)
        # Default: LocalAI /tts endpoint (commonly WAV)
        return self.client.tts_raw(self.cfg.models.tts, text=text, extra_params=self.cfg.tts.extra_params)

    # ---------------- UI build ----------------
    def build(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="Skippy", width=1180, height=720, small_icon=None, large_icon=None)

        self._setup_fonts()
        self._setup_theme()

        with dpg.window(tag="Primary", no_title_bar=False):
            self._build_menu()
            self._build_layout()

        dpg.set_primary_window("Primary", True)

        # Global key handlers for PTT
        ptt = _keycode(self.cfg.audio.ptt_key)
        with dpg.handler_registry():
            dpg.add_key_down_handler(key=ptt, callback=lambda: self.ptt_start(), tag="ptt_down")
            dpg.add_key_release_handler(key=ptt, callback=lambda: self.ptt_stop(), tag="ptt_up")

        # UI tick
        dpg.set_render_callback(self._tick)

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def _setup_fonts(self) -> None:
        # Use default font; allow scaling via font size
        fs = int(self.cfg.ui.font_size)
        with dpg.font_registry():
            default_font = dpg.add_font_default(size=fs)
        dpg.bind_font(default_font)

    def _setup_theme(self) -> None:
        # Modern-ish dark theme
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 8)

                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 14, 12)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 8)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 10)

        dpg.bind_theme(theme)

    def _build_menu(self) -> None:
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Reload config", callback=self._reload_config)
                dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())
            with dpg.menu(label="Tools"):
                dpg.add_menu_item(label="Refresh models", callback=lambda: self.refresh_models())
                dpg.add_menu_item(label="Cancel current", callback=lambda: self.cancel())
            with dpg.menu(label="Help"):
                dpg.add_menu_item(label="About", callback=self._about_modal)

    def _build_layout(self) -> None:
        with dpg.group(horizontal=True):
            # LEFT: chat
            with dpg.child_window(width=760, height=-1, border=True):
                dpg.add_text("Conversation")
                dpg.add_separator()
                self.ids["chat_log"] = dpg.add_child_window(height=-110, border=False)
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    self.ids["status"] = dpg.add_text("Ready")
                    dpg.add_spacer(width=20)
                    self.ids["busy"] = dpg.add_text("")

                with dpg.group(horizontal=True):
                    self.ids["ptt_btn"] = dpg.add_button(label="Hold to talk (or use hotkey)", width=360)
                    dpg.bind_item_handler_registry(self.ids["ptt_btn"], self._ptt_mouse_handlers())
                    self.ids["cancel_btn"] = dpg.add_button(label="Cancel", width=120, callback=lambda: self.cancel())
                    self.ids["stream_chk"] = dpg.add_checkbox(label="Stream reply", default_value=self.state.stream, callback=self._toggle_stream)
                    self.ids["temp_slider"] = dpg.add_slider_float(label="Temp", min_value=0.0, max_value=1.2, default_value=self.state.temperature, width=220, callback=self._set_temp)

            # RIGHT: settings
            with dpg.child_window(width=-1, height=-1, border=True):
                dpg.add_text("Settings")
                dpg.add_separator()
                dpg.add_text("LocalAI")
                self.ids["base_url"] = dpg.add_input_text(label="Base URL", default_value=self.cfg.localai.base_url, readonly=True, width=-1)
                with dpg.group(horizontal=True):
                    self.ids["refresh_models"] = dpg.add_button(label="Refresh models", callback=lambda: self.refresh_models())
                    self.ids["models_hint"] = dpg.add_text("")

                dpg.add_spacer(height=6)
                dpg.add_text("Models")
                self.ids["llm_combo"] = dpg.add_combo(label="LLM", items=[self.cfg.models.llm], default_value=self.cfg.models.llm, width=-1, callback=self._set_llm)
                self.ids["stt_combo"] = dpg.add_combo(label="STT", items=[self.cfg.models.stt], default_value=self.cfg.models.stt, width=-1, callback=self._set_stt)
                self.ids["tts_combo"] = dpg.add_combo(label="TTS", items=[self.cfg.models.tts], default_value=self.cfg.models.tts, width=-1, callback=self._set_tts)
                self.ids["tts_endpoint"] = dpg.add_combo(label="TTS endpoint", items=["tts", "openai"], default_value=self.cfg.models.tts_endpoint, width=-1, callback=self._set_tts_endpoint)

                dpg.add_spacer(height=6)
                dpg.add_text("Persona")
                self.ids["persona_path"] = dpg.add_input_text(label="Persona file", default_value=self.cfg.persona.persona_file, width=-1, callback=self._persona_path_changed)
                self.ids["persona_text"] = dpg.add_input_text(multiline=True, height=200, width=-1, default_value=self.persona_text, callback=self._persona_text_changed)

                dpg.add_spacer(height=6)
                dpg.add_text("Push-to-talk")
                self.ids["ptt_key"] = dpg.add_input_text(label="Hotkey", default_value=self.cfg.audio.ptt_key, width=-1, readonly=True)
                dpg.add_text("Change hotkey in config/default.toml (audio.ptt_key)")

        # Seed chat with persona line for clarity
        self._add_log("Skippy", "Ready. Hold the button or PTT hotkey and speak.")

        # Auto-refresh models once
        self.refresh_models()

    def _ptt_mouse_handlers(self) -> int:
        with dpg.item_handler_registry() as reg:
            dpg.add_item_clicked_handler(callback=lambda sender, app_data, user_data: None)
            dpg.add_item_active_handler(callback=lambda: self.ptt_start())
            dpg.add_item_deactivated_handler(callback=lambda: self.ptt_stop())
        return reg

    # ---------------- Callbacks ----------------
    def _toggle_stream(self, sender, app_data):
        self.state.stream = bool(app_data)

    def _set_temp(self, sender, app_data):
        self.state.temperature = float(app_data)

    def _set_llm(self, sender, app_data):
        self.cfg.models.llm = str(app_data)

    def _set_stt(self, sender, app_data):
        self.cfg.models.stt = str(app_data)

    def _set_tts(self, sender, app_data):
        self.cfg.models.tts = str(app_data)

    def _set_tts_endpoint(self, sender, app_data):
        self.cfg.models.tts_endpoint = str(app_data)

    def _persona_path_changed(self, sender, app_data):
        # Don't live-write to disk; just load if exists.
        try:
            p = (self.cfg.root_dir / str(app_data)).resolve()
            if p.exists():
                self.persona_text = p.read_text(encoding="utf-8").strip()
                dpg.set_value(self.ids["persona_text"], self.persona_text)
                self.state.messages[0] = {"role": "system", "content": self.persona_text}
                self._set_status("Persona loaded")
        except Exception as e:
            self._set_status(f"Persona load error: {e}")

    def _persona_text_changed(self, sender, app_data):
        # Live-update system prompt in memory
        self.persona_text = str(app_data)
        if self.state.messages and self.state.messages[0].get("role") == "system":
            self.state.messages[0]["content"] = self.persona_text

    def _reload_config(self) -> None:
        if not self.cfg.config_path:
            self._set_status("No config path set")
            return
        try:
            new_cfg = load_config(self.cfg.config_path)
            self.cfg = new_cfg
            self.client = LocalAIClient(new_cfg.localai.base_url, timeout_s=new_cfg.localai.timeout_s)
            self.persona_text = read_persona_text(new_cfg)
            self.state = ChatState(messages=[{"role": "system", "content": self.persona_text}])
            self.recorder = Recorder(
                sample_rate=new_cfg.audio.sample_rate,
                channels=new_cfg.audio.channels,
                device=device_by_name(new_cfg.audio.device_in, kind="input"),
                max_seconds=new_cfg.audio.max_record_seconds,
            )
            self.out_device = device_by_name(new_cfg.audio.device_out, kind="output")
            dpg.set_value(self.ids["base_url"], new_cfg.localai.base_url)
            dpg.set_value(self.ids["persona_path"], new_cfg.persona.persona_file)
            dpg.set_value(self.ids["persona_text"], self.persona_text)
            dpg.set_value(self.ids["ptt_key"], new_cfg.audio.ptt_key)
            self._set_status("Config reloaded")
            self.refresh_models()
        except Exception as e:
            self._set_status(f"Config reload error: {e}")

    def _about_modal(self) -> None:
        win_id = dpg.generate_uuid()
        with dpg.window(
            label="About Skippy",
            modal=True,
            no_resize=True,
            width=520,
            height=260,
            pos=(280, 180),
            tag=win_id,
        ):
            dpg.add_text("Skippy — LocalAI Speech-to-Speech (Push-to-Talk)\\n")
            dpg.add_text("• Desktop GUI: DearPyGUI\\n• Backend: LocalAI (local LLM + STT + TTS)\\n")
            dpg.add_text("Tip: mount LocalAI /models to Skippy/models to avoid re-downloading.")
            dpg.add_spacer(height=10)
            dpg.add_button(label="Close", width=120, callback=lambda: dpg.delete_item(win_id))


    # ---------------- UI tick ----------------
    def _tick(self):
        # Process queued UI ops (thread-safe)
        while True:
            try:
                op, payload = self.ui_queue.get_nowait()
            except queue.Empty:
                break

            if op == "status":
                dpg.set_value(self.ids["status"], str(payload))
            elif op == "busy":
                dpg.set_value(self.ids["busy"], "⏳" if payload else "")
                dpg.configure_item(self.ids["ptt_btn"], enabled=not payload)
            elif op == "add_log":
                who = payload["who"]
                text = payload["text"]
                self._append_chat_bubble(who, text)
            elif op == "models_list":
                models: List[str] = payload
                # Populate combos with fetched models, keep current if possible
                for key, cur in [("llm_combo", self.cfg.models.llm), ("stt_combo", self.cfg.models.stt), ("tts_combo", self.cfg.models.tts)]:
                    dpg.configure_item(self.ids[key], items=models)
                    if cur in models:
                        dpg.set_value(self.ids[key], cur)
                    elif models:
                        dpg.set_value(self.ids[key], models[0])
                dpg.set_value(self.ids["models_hint"], f"{len(models)} models")
            elif op == "assistant_new_stream":
                self._start_stream_bubble()
            elif op == "assistant_stream":
                self._update_stream_bubble(str(payload))
            elif op == "assistant_finalize_stream":
                self._finalize_stream_bubble(str(payload))

    def _append_chat_bubble(self, who: str, text: str) -> None:
        parent = self.ids["chat_log"]
        with dpg.group(parent=parent):
            header = f"{who}"
            dpg.add_text(header)
            dpg.add_text(text, wrap=720)
            dpg.add_spacer(height=8)
        dpg.set_y_scroll(parent, 10**9)

    def _start_stream_bubble(self) -> None:
        parent = self.ids["chat_log"]
        with dpg.group(parent=parent) as grp:
            dpg.add_text("Skippy")
            self.ids["stream_text"] = dpg.add_text("", wrap=720)
            dpg.add_spacer(height=8)
        dpg.set_y_scroll(parent, 10**9)

    def _update_stream_bubble(self, text: str) -> None:
        if "stream_text" in self.ids:
            dpg.set_value(self.ids["stream_text"], text)

    def _finalize_stream_bubble(self, text: str) -> None:
        self._update_stream_bubble(text)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Skippy — LocalAI speech-to-speech (push-to-talk)")
    parser.add_argument("--config", default="config/default.toml", help="Path to config TOML")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    app = SkippyApp(cfg)
    app.build()

    try:
        dpg.start_dearpygui()
    finally:
        dpg.destroy_context()
    return 0
