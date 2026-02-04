from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import dearpygui.dearpygui as dpg

from .agent_graph import SkippyAgentGraph
from .audio import Recorder, device_by_name, play_audio, stop_playback
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
        self.agent = SkippyAgentGraph(self.client)

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

        self._lock = threading.Lock()

    # ---------------- UI helpers ----------------
    def _enqueue(self, op: str, payload: Any) -> None:
        self.ui_queue.put((op, payload))

    def _add_log(self, who: str, text: str) -> None:
        # Append message widget to log child
        self._enqueue("add_log", {"who": who, "text": text})

    def _set_status(self, text: str) -> None:
        self._enqueue("status", text)

    def _show_error(self, title: str, message: str) -> None:
        self._enqueue("error_modal", {"title": title, "message": message})

    def _set_busy(self, busy: bool) -> None:
        self._enqueue("busy", busy)

    def _set_assistant_stream_text(self, text: str) -> None:
        self._enqueue("assistant_stream", text)

    def _get_messages(self) -> List[Dict[str, str]]:
        with self._lock:
            return list(self.state.messages)

    def _append_message(self, role: str, content: str) -> None:
        with self._lock:
            self.state.messages.append({"role": role, "content": content})

    def _clear_messages(self) -> None:
        with self._lock:
            self.state.messages = [{"role": "system", "content": self.persona_text}]

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
        with self._lock:
            if self.state.busy:
                return
        self.recorder.start()
        self._enqueue("ptt_active", True)
        self._set_status("Recording… release to send")

    def ptt_stop(self) -> None:
        with self._lock:
            if self.state.busy:
                return
        self._enqueue("ptt_active", False)
        wav_bytes = self.recorder.stop()
        if not wav_bytes:
            self._set_status("No audio captured")
            return

        with self._lock:
            self.state.busy = True
            self.state.cancel_flag = False
        self._set_busy(True)
        self._set_status("Transcribing…")

        threading.Thread(target=self._process_turn, args=(wav_bytes,), daemon=True).start()

    def cancel(self) -> None:
        with self._lock:
            self.state.cancel_flag = True
        stop_playback()
        self._set_status("Cancel requested…")

    def _process_turn(self, wav_bytes: bytes) -> None:
        try:
            with self._lock:
                stt_model = self.cfg.models.stt
                llm_model = self.cfg.models.llm
                tts_model = self.cfg.models.tts
                tts_endpoint = self.cfg.models.tts_endpoint
                tts_extra_params = dict(self.cfg.tts.extra_params)
                tts_response_format = self.cfg.tts.response_format
                out_device = self.out_device
                volume = self.cfg.audio.volume
                stream = self.state.stream
                temp = self.state.temperature

            user_text = self.client.transcribe(wav_bytes, model=stt_model)
            user_text = (user_text or "").strip()
            if not user_text:
                self._set_status("Transcription was empty")
                return

            # Add user message
            self._add_log("You", user_text)
            self._append_message("user", user_text)

            # LLM
            self._set_status("Thinking…")
            assistant_text = ""
            messages = self._get_messages()

            if stream:
                self._enqueue("assistant_new_stream", None)

            def stream_callback(text: str) -> None:
                self._set_assistant_stream_text(text)

            def cancel_check() -> bool:
                with self._lock:
                    return self.state.cancel_flag

            assistant_text = self.agent.run(
                model=llm_model,
                messages=messages,
                temperature=temp,
                stream=stream,
                stream_callback=stream_callback if stream else None,
                cancel_check=cancel_check,
            )

            with self._lock:
                if self.state.cancel_flag:
                    return

            assistant_text = assistant_text.strip()
            if stream:
                if assistant_text:
                    self._enqueue("assistant_finalize_stream", assistant_text)
            else:
                if assistant_text:
                    self._add_log("Skippy", assistant_text)

            if not assistant_text:
                self._set_status("No assistant response")
                return

            self._append_message("assistant", assistant_text)

            # TTS
            self._set_status("Speaking…")
            if tts_endpoint.lower() == "openai":
                voice = str(tts_extra_params.get("voice", "alloy"))
                tts_bytes = self.client.tts_openai(
                    tts_model, text=assistant_text, voice=voice, response_format=tts_response_format
                )
            else:
                tts_bytes = self.client.tts_raw(
                    tts_model, text=assistant_text, extra_params=tts_extra_params
                )

            with self._lock:
                if self.state.cancel_flag:
                    return

            play_audio(tts_bytes, device=out_device, volume=volume)
            self._set_status("Ready")
        except Exception as e:
            self._set_status(f"Error: {e}")
        finally:
            with self._lock:
                self.state.busy = False
            self._set_busy(False)

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
            dpg.add_key_press_handler(key=ptt, callback=lambda: self.ptt_start())
            dpg.add_key_release_handler(key=ptt, callback=lambda: self.ptt_stop())

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
        if self.cfg.ui.theme.lower() == "light":
            self._setup_light_theme()
            return

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

        with dpg.theme() as self.recording_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 50, 50))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHover, (230, 70, 70))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (150, 30, 30))

    def _setup_light_theme(self) -> None:
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 14, 12)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 8)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 10)

                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (240, 240, 240))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (225, 225, 225))
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (240, 240, 240))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (180, 180, 180))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (255, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHover, (230, 230, 230))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (210, 210, 210))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (200, 200, 200))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (180, 180, 180))
                dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, (220, 220, 220))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 200, 200))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHover, (180, 180, 180))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (160, 160, 160))
                dpg.add_theme_color(dpg.mvThemeCol_Header, (200, 200, 200))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHover, (180, 180, 180))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (160, 160, 160))

        dpg.bind_theme(theme)

    def _build_menu(self) -> None:
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Reload config", callback=self._reload_config)
                dpg.add_menu_item(label="Save config", callback=self._save_config_callback)
                dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())
            with dpg.menu(label="Tools"):
                dpg.add_menu_item(label="Refresh models", callback=lambda: self.refresh_models())
                dpg.add_menu_item(label="Cancel current", callback=lambda: self.cancel())
                dpg.add_menu_item(label="Clear conversation", callback=lambda: self._clear_conv_callback())
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
                    self.ids["ptt_btn"] = dpg.add_button(label="Hold to talk (or use hotkey)", width=300)
                    dpg.bind_item_handler_registry(self.ids["ptt_btn"], self._ptt_mouse_handlers())
                    dpg.add_text("Vol")
                    self.ids["volume_slider"] = dpg.add_slider_float(default_value=self.cfg.audio.volume, min_value=0.0, max_value=2.0, width=120, callback=self._set_volume)
                    self.ids["cancel_btn"] = dpg.add_button(label="Cancel", width=120, callback=lambda: self.cancel())
                    self.ids["clear_btn"] = dpg.add_button(label="Clear", width=80, callback=lambda: self._clear_conv_callback())
                    with self._lock:
                        stream_val = self.state.stream
                        temp_val = self.state.temperature
                    self.ids["stream_chk"] = dpg.add_checkbox(label="Stream reply", default_value=stream_val, callback=self._toggle_stream)
                    self.ids["temp_slider"] = dpg.add_slider_float(label="Temp", min_value=0.0, max_value=1.2, default_value=temp_val, width=220, callback=self._set_temp)

            # RIGHT: settings
            with dpg.child_window(width=-1, height=-1, border=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("Settings")
                    dpg.add_spacer(width=200)
                    dpg.add_button(label="Save current as default", callback=self._save_config_callback)
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

                dpg.add_spacer(height=6)
                dpg.add_text("Audio Devices")
                devices = self._get_device_names()
                self.ids["device_in"] = dpg.add_combo(label="Input", items=devices["input"], default_value=self.cfg.audio.device_in, width=-1, callback=self._set_device_in)
                self.ids["device_out"] = dpg.add_combo(label="Output", items=devices["output"], default_value=self.cfg.audio.device_out, width=-1, callback=self._set_device_out)

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
    def _clear_conv_callback(self):
        self._clear_messages()
        dpg.delete_item(self.ids["chat_log"], children_only=True)
        self._add_log("Skippy", "Conversation cleared.")
        self._set_status("Ready")

    def _toggle_stream(self, sender, app_data):
        with self._lock:
            self.state.stream = bool(app_data)

    def _set_temp(self, sender, app_data):
        with self._lock:
            self.state.temperature = float(app_data)

    def _set_llm(self, sender, app_data):
        with self._lock:
            self.cfg.models.llm = str(app_data)

    def _set_stt(self, sender, app_data):
        with self._lock:
            self.cfg.models.stt = str(app_data)

    def _set_tts(self, sender, app_data):
        with self._lock:
            self.cfg.models.tts = str(app_data)

    def _set_tts_endpoint(self, sender, app_data):
        with self._lock:
            self.cfg.models.tts_endpoint = str(app_data)

    def _set_device_in(self, sender, app_data):
        with self._lock:
            self.cfg.audio.device_in = str(app_data)
            self.recorder.device = device_by_name(self.cfg.audio.device_in, kind="input")

    def _set_device_out(self, sender, app_data):
        with self._lock:
            self.cfg.audio.device_out = str(app_data)
            self.out_device = device_by_name(self.cfg.audio.device_out, kind="output")

    def _set_volume(self, sender, app_data):
        with self._lock:
            self.cfg.audio.volume = float(app_data)

    def _get_device_names(self) -> Dict[str, List[str]]:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devs = []
        output_devs = []
        for d in devices:
            name = d.get("name", "Unknown")
            if d.get("max_input_channels", 0) > 0:
                input_devs.append(name)
            if d.get("max_output_channels", 0) > 0:
                output_devs.append(name)
        return {"input": sorted(set(input_devs)), "output": sorted(set(output_devs))}

    def _persona_path_changed(self, sender, app_data):
        # Don't live-write to disk; just load if exists.
        try:
            p = (self.cfg.root_dir / str(app_data)).resolve()
            if p.exists():
                self.persona_text = p.read_text(encoding="utf-8").strip()
                dpg.set_value(self.ids["persona_text"], self.persona_text)
                with self._lock:
                    if self.state.messages and self.state.messages[0].get("role") == "system":
                        self.state.messages[0]["content"] = self.persona_text
                    else:
                        self.state.messages.insert(0, {"role": "system", "content": self.persona_text})
                self._set_status("Persona loaded")
        except Exception as e:
            self._set_status(f"Persona load error: {e}")
            self._show_error("Persona Load Error", str(e))

    def _persona_text_changed(self, sender, app_data):
        # Live-update system prompt in memory
        self.persona_text = str(app_data)
        with self._lock:
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
            self.agent = SkippyAgentGraph(self.client)
            self.persona_text = read_persona_text(new_cfg)
            with self._lock:
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
            self._show_error("Config Reload Error", str(e))

    def _save_config_callback(self) -> None:
        if not self.cfg.config_path:
            self._set_status("No config path set")
            return
        try:
            with self._lock:
                self.cfg.save()
            self._set_status(f"Config saved to {self.cfg.config_path.name}")
        except Exception as e:
            self._set_status(f"Config save error: {e}")
            self._show_error("Config Save Error", str(e))

    def _error_modal(self, title: str, message: str) -> None:
        win_id = dpg.generate_uuid()
        with dpg.window(
            label=title,
            modal=True,
            no_resize=True,
            width=500,
            height=-1,
            pos=(300, 200),
            tag=win_id,
        ):
            dpg.add_text(message, wrap=480)
            dpg.add_spacer(height=10)
            dpg.add_button(label="OK", width=120, callback=lambda: dpg.delete_item(win_id))

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
                with self._lock:
                    combos = [
                        ("llm_combo", self.cfg.models.llm, "llm"),
                        ("stt_combo", self.cfg.models.stt, "stt"),
                        ("tts_combo", self.cfg.models.tts, "tts"),
                    ]
                    for key, cur, attr in combos:
                        dpg.configure_item(self.ids[key], items=models)
                        if cur in models:
                            dpg.set_value(self.ids[key], cur)
                        elif models:
                            new_val = models[0]
                            dpg.set_value(self.ids[key], new_val)
                            setattr(self.cfg.models, attr, new_val)
                dpg.set_value(self.ids["models_hint"], f"{len(models)} models")
            elif op == "assistant_new_stream":
                self._start_stream_bubble()
            elif op == "assistant_stream":
                self._update_stream_bubble(str(payload))
            elif op == "assistant_finalize_stream":
                self._finalize_stream_bubble(str(payload))
            elif op == "error_modal":
                self._error_modal(payload["title"], payload["message"])
            elif op == "ptt_active":
                if payload:
                    dpg.bind_item_theme(self.ids["ptt_btn"], self.recording_theme)
                    dpg.configure_item(self.ids["ptt_btn"], label="RECORDING...")
                else:
                    dpg.bind_item_theme(self.ids["ptt_btn"], 0)
                    dpg.configure_item(self.ids["ptt_btn"], label="Hold to talk (or use hotkey)")

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
        stop_playback()
        dpg.destroy_context()
    return 0
