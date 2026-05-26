#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import soundfile as sf
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import var
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static, TextArea

PROJECT_DIR = Path(__file__).parent.resolve()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CancelledError(Exception):
    pass


# ---------------------------------------------------------------------------
# Component C: AudioPlayer
# ---------------------------------------------------------------------------

class AudioPlayer:
    def __init__(self) -> None:
        system = platform.system()
        if system == "Windows":
            self._use_winsound = True
        elif system == "Darwin":
            self._cmd = ["afplay", "{}"]
            self._use_winsound = False
        else:
            self._cmd = ["aplay", "{}"]
            self._use_winsound = False
        self._process: subprocess.Popen | None = None

    def play(self, filepath: str) -> None:
        self.stop()
        if self._use_winsound:
            import winsound
            winsound.PlaySound(filepath, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            cmd = [arg.format(filepath) for arg in self._cmd]
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    def stop(self) -> None:
        if self._use_winsound:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        elif self._process:
            self._process.terminate()
            self._process = None


# ---------------------------------------------------------------------------
# Component B: SessionLog
# ---------------------------------------------------------------------------

class SessionLog:
    def __init__(self, log_dir: str | Path, log_history: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.log_history = log_history
        self.entries: list[dict] = []
        self.log_dir.mkdir(exist_ok=True)

    def add_entry(
        self, text: str, output_file: str, voice_used: str,
        device: str, duration_ms: int,
    ) -> dict:
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "text": text,
            "file": output_file,
            "voice": Path(voice_used).name if voice_used else "unknown",
            "device": device,
            "duration_ms": duration_ms,
        }
        self.entries.append(entry)

        if self.log_history:
            date_str = datetime.now().strftime("%Y-%m-%d")
            log_file = self.log_dir / f"{date_str}.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return entry

    def recent_entries(self, count: int = 50) -> list[dict]:
        return self.entries[-count:]


# ---------------------------------------------------------------------------
# Component A: ModelEngine
# ---------------------------------------------------------------------------

class ModelEngine:
    def __init__(self, model_dir: str | Path) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        self._cancelled = False
        self._generation_count = 0
        self._prompt_cache: dict[str, list] = {}
        self._active_voice: str | None = None
        self.model_dir = str(model_dir)

        from rich.progress import Progress, SpinnerColumn, TextColumn
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        ) as progress:
            progress.add_task("Loading model...", total=None)

            self.model = Qwen3TTSModel.from_pretrained(
                self.model_dir,
                device_map="cpu",
                dtype=torch.float32,
            )

        self._prompt_cache.clear()
        self._active_voice = None

    def load_voice(self, sample_path: str) -> None:
        self._active_voice = sample_path
        if sample_path in self._prompt_cache:
            return

        txt_path = Path(sample_path).with_suffix(".txt")
        if not txt_path.exists():
            raise FileNotFoundError(f"Transcript not found: {txt_path}")
        ref_text = txt_path.read_text(encoding="utf-8").strip()
        if not ref_text:
            raise ValueError(f"Transcript empty: {txt_path}")

        prompt_items = self.model.create_voice_clone_prompt(
            ref_audio=sample_path,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        self._prompt_cache[sample_path] = prompt_items

    def generate(self, text: str) -> tuple[list, int]:
        if not self._active_voice or self._active_voice not in self._prompt_cache:
            raise RuntimeError("No voice loaded — press F2 to select a voice")

        self._cancelled = False
        prompt = self._prompt_cache[self._active_voice]

        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language="auto",
            voice_clone_prompt=prompt,
        )

        if self._cancelled:
            raise CancelledError("Generation cancelled")

        self._generation_count += 1
        return wavs, sr

    def reload(self, size: str) -> None:
        from qwen_tts import Qwen3TTSModel
        import gc

        dir_name = f"{size}-base"
        new_dir = PROJECT_DIR / "models" / dir_name
        if not new_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {new_dir}")

        from rich.progress import Progress, SpinnerColumn, TextColumn
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        ) as progress:
            progress.add_task(f"Loading {size} model...", total=None)
            new_model = Qwen3TTSModel.from_pretrained(
                str(new_dir), device_map="cpu", dtype=torch.float32,
            )

        self.model = new_model
        self.model_dir = str(new_dir)
        self._prompt_cache.clear()
        self._active_voice = None
        gc.collect()

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def generation_count(self) -> int:
        return self._generation_count

    @property
    def active_voice(self) -> str | None:
        return self._active_voice

    @property
    def model_size(self) -> str:
        dir_name = Path(self.model_dir).name
        if "0.6b" in dir_name:
            return "0.6B"
        if "1.7b" in dir_name:
            return "1.7B"
        return dir_name


# ---------------------------------------------------------------------------
# TUI — Voice Selector Modal
# ---------------------------------------------------------------------------

class VoiceSelector(ModalScreen[str | None]):
    CSS = """
    VoiceSelector {
        align: center middle;
    }

    #voice-modal {
        width: 50;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1;
    }

    #voice-modal > Static {
        text-style: bold;
        padding-bottom: 1;
    }

    #voice-list {
        height: auto;
        max-height: 16;
    }

    .hint {
        color: $text-muted;
        padding-top: 1;
        text-align: center;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="voice-modal"):
            yield Static("Select Voice")
            yield ListView(id="voice-list")
            yield Static("[Esc] Cancel", classes="hint")

    def on_mount(self) -> None:
        list_view = self.query_one("#voice-list", ListView)
        samples_dir = PROJECT_DIR / "samples"
        if samples_dir.exists():
            for wav in sorted(samples_dir.glob("*.wav")):
                txt = wav.with_suffix(".txt")
                status = "✓" if txt.exists() else "✗"
                list_view.append(ListItem(Label(f"{status}  {wav.name}")))
        list_view.index = 0

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item:
            label = event.item.query_one(Label).render()
            label_str = str(label)
            parts = label_str.split("  ", 1)
            if len(parts) == 2:
                self.dismiss(parts[1])

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)


# ---------------------------------------------------------------------------
# TUI — Model Selector Modal
# ---------------------------------------------------------------------------

class ModelSelector(ModalScreen[str | None]):
    CSS = """
    ModelSelector {
        align: center middle;
    }

    #model-modal {
        width: 40;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1;
    }

    #model-modal > Static {
        text-style: bold;
        padding-bottom: 1;
    }

    .hint {
        color: $text-muted;
        padding-top: 1;
        text-align: center;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="model-modal"):
            yield Static("Select Model")
            yield ListView(
                ListItem(Label("0.6B (faster)")),
                ListItem(Label("1.7B (better)")),
                id="model-list",
            )
            yield Static("[Esc] Cancel", classes="hint")

    def on_mount(self) -> None:
        self.query_one("#model-list", ListView).index = 0

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item:
            label = str(event.item.query_one(Label).render())
            self.dismiss("0.6b" if "0.6" in label else "1.7b")

    def on_key(self, event) -> None:
        if event.key == "escape":
            self.dismiss(None)


# ---------------------------------------------------------------------------
# TUI — Status Panel
# ---------------------------------------------------------------------------

class StatusPanel(Static):
    model_size = var("")
    voice_name = var("none")
    voice_status = var("")
    state = var("Ready")
    last_file = var("")

    def render(self) -> str:
        return (
            f"Model: Qwen3-TTS-12Hz-{self.model_size}-Base\n"
            f"Device: CPU\n"
            f"Voice: {self.voice_name} {self.voice_status}\n"
            f"CPU mode — generation may be slow (10-60s)\n"
            f"State: {self.state}\n"
            f"Last: {self.last_file}\n"
        )


# ---------------------------------------------------------------------------
# TUI — History Panel
# ---------------------------------------------------------------------------

class HistoryPanel(ListView):
    def add_entry(self, entry: dict) -> None:
        ts = entry["time"]
        text = entry["text"]
        display = text[:50] + "..." if len(text) > 50 else text
        self.append(ListItem(Label(f"{ts}  {display}")))
        self.scroll_end(animate=False)


# ---------------------------------------------------------------------------
# TUI — Custom Header
# ---------------------------------------------------------------------------

class AppHeader(Static):
    voice_name = var("none")
    gen_count = var(0)

    def render(self) -> str:
        return (
            f" Qwen3-TTS  |  CPU ⚠  |  Voice: {self.voice_name}  |  #{self.gen_count}"
        )


# ---------------------------------------------------------------------------
# TUI — Main App
# ---------------------------------------------------------------------------

class TTSApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    AppHeader {
        height: 1;
        dock: top;
        background: $primary-background;
        color: $text;
        padding: 0 1;
    }

    #main-area {
        height: 1fr;
    }

    #history-panel {
        width: 40%;
        border: solid $secondary;
        padding: 0 1;
    }

    StatusPanel {
        width: 60%;
        border: solid $secondary;
        padding: 1 2;
    }

    #input-area {
        dock: bottom;
        height: 6;
        max-height: 12;
        margin: 0 1;
        border: solid $secondary;
    }

    Footer {
        dock: bottom;
    }
    """

    BINDINGS = [
        Binding("f2", "select_voice", "Voice", show=True),
        Binding("f3", "open_folder", "Folder", show=True),
        Binding("f4", "play_last", "Play last", show=True),
        Binding("f5", "select_model", "Model", show=True),
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("ctrl+q", "quit", "Quit", show=False),
    ]

    def __init__(
        self,
        model_engine: ModelEngine,
        session_log: SessionLog,
        audio_player: AudioPlayer,
        config: dict,
    ) -> None:
        super().__init__()
        self.model_engine = model_engine
        self.session_log = session_log
        self.audio_player = audio_player
        self.config = config
        self._last_output: str | None = None
        self._switching_model = False

    def compose(self) -> ComposeResult:
        yield AppHeader()
        with Horizontal(id="main-area"):
            yield HistoryPanel(id="history-panel")
            yield StatusPanel()
        yield TextArea(id="input-area")
        yield Footer()

    def on_mount(self) -> None:
        header = self.query_one(AppHeader)
        header.voice_name = self._voice_display_name()
        header.gen_count = self.model_engine.generation_count

        status = self.query_one(StatusPanel)
        status.model_size = self.model_engine.model_size
        self._refresh_voice_status()

        text_area = self.query_one("#input-area", TextArea)
        text_area.placeholder = "Type text and press Enter to generate  |  Shift+Enter for newline"

    def _voice_display_name(self) -> str:
        v = self.model_engine.active_voice
        return Path(v).stem if v else "none"

    def _refresh_voice_status(self) -> None:
        status = self.query_one(StatusPanel)
        v = self.model_engine.active_voice
        status.voice_name = self._voice_display_name()
        if v:
            txt = Path(v).with_suffix(".txt")
            status.voice_status = "✓" if txt.exists() else "✗ (missing transcript!)"
        else:
            status.voice_status = "✗ (none selected)"

    # -- input submission --

    def on_key(self, event) -> None:
        if event.key == "enter" and not event.shift:
            text_area = self.query_one("#input-area", TextArea)
            if text_area.has_focus:
                event.stop()
                text = text_area.text.strip()
                if not text:
                    return
                text_area.text = ""
                self._start_generation(text)

    def _start_generation(self, text: str) -> None:
        if self._switching_model:
            self.query_one(StatusPanel).state = "Model loading, please wait"
            return
        if not self.model_engine.active_voice:
            self.query_one(StatusPanel).state = "No voice — press F2 to select"
            return
        sample_path = Path(self.model_engine.active_voice)
        if not sample_path.exists():
            self.query_one(StatusPanel).state = "Voice file missing"
            return
        txt_path = sample_path.with_suffix(".txt")
        if not txt_path.exists():
            self.query_one(StatusPanel).state = "Transcript missing"
            return

        self.query_one("#input-area", TextArea).disabled = True
        self.query_one(StatusPanel).state = "Generating..."
        self.current_text = text
        self.generate_worker(text)

    @work(thread=True, exclusive="generate")
    async def generate_worker(self, text: str) -> None:
        try:
            start = time.time()
            wavs, sr = self.model_engine.generate(text)
            elapsed = int((time.time() - start) * 1000)

            output_dir = PROJECT_DIR / self.config.get("output", {}).get("folder", "output")
            output_dir.mkdir(exist_ok=True)

            naming = self.config.get("output", {}).get("naming", "timestamp")
            if naming == "prompt-prefix":
                words = re.sub(r'[^a-z0-9\s-]', '', text.lower()).split()[:4]
                prefix = "-".join(words) if words else "speech"
                filename = f"{prefix}_{datetime.now().strftime('%H-%M-%S')}.wav"
            else:
                filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.wav"

            output_path = output_dir / filename
            sf.write(str(output_path), wavs[0], sr)
            self._last_output = str(output_path)

            entry = self.session_log.add_entry(
                text=text,
                output_file=str(output_path),
                voice_used=self.model_engine.active_voice or "",
                device="cpu",
                duration_ms=elapsed,
            )

            self.call_from_thread(self._on_done, entry, str(output_path))
        except CancelledError:
            self.call_from_thread(self._on_cancelled)
        except Exception as exc:
            self.call_from_thread(self._on_error, str(exc))

    def _on_done(self, entry: dict, output_path: str) -> None:
        self.query_one("#input-area", TextArea).disabled = False
        self.query_one("#input-area", TextArea).focus()

        status = self.query_one(StatusPanel)
        status.state = "Ready"
        status.last_file = Path(output_path).name

        self.query_one(HistoryPanel).add_entry(entry)

        header = self.query_one(AppHeader)
        header.gen_count = self.model_engine.generation_count

        if self.config.get("output", {}).get("auto_play", False):
            self.audio_player.play(output_path)

    def _on_cancelled(self) -> None:
        self.query_one("#input-area", TextArea).disabled = False
        self.query_one("#input-area", TextArea).focus()
        status = self.query_one(StatusPanel)
        status.state = "Cancelled"
        self.set_timer(2.0, lambda: setattr(status, "state", "Ready"))

    def _on_error(self, msg: str) -> None:
        self.query_one("#input-area", TextArea).disabled = False
        self.query_one("#input-area", TextArea).focus()
        self.query_one(StatusPanel).state = f"Error: {msg}"

    # -- actions --

    def action_select_voice(self) -> None:
        def on_voice_selected(result: str | None) -> None:
            if result:
                sample_path = str(PROJECT_DIR / "samples" / result)
                try:
                    self.model_engine.load_voice(sample_path)
                    self._refresh_voice_status()
                    self.query_one(AppHeader).voice_name = self._voice_display_name()
                    self.query_one(StatusPanel).state = f"Ready (voice: {Path(result).stem})"
                except (FileNotFoundError, ValueError) as e:
                    self.query_one(StatusPanel).state = f"Voice error: {e}"

        self.push_screen(VoiceSelector(), on_voice_selected)

    def action_open_folder(self) -> None:
        output_dir = PROJECT_DIR / self.config.get("output", {}).get("folder", "output")
        output_dir.mkdir(exist_ok=True)
        system = platform.system()
        try:
            if system == "Windows":
                subprocess.Popen(["explorer", str(output_dir.resolve())])
            elif system == "Darwin":
                subprocess.Popen(["open", str(output_dir.resolve())])
            else:
                subprocess.Popen(["xdg-open", str(output_dir.resolve())])
        except Exception:
            self.query_one(StatusPanel).state = "Could not open folder"

    def action_play_last(self) -> None:
        if self._last_output and Path(self._last_output).exists():
            self.audio_player.play(self._last_output)
            self.query_one(StatusPanel).state = "Playing..."
            self.set_timer(0.5, lambda: setattr(self.query_one(StatusPanel), "state", "Ready"))
        else:
            self.query_one(StatusPanel).state = "No file to play"

    def action_cancel(self) -> None:
        self.model_engine.cancel()
        status = self.query_one(StatusPanel)
        if status.state == "Generating...":
            status.state = "Cancelling..."
        else:
            status.state = "Ready"

    # -- model switching --

    def action_select_model(self) -> None:
        if self._switching_model:
            return
        status = self.query_one(StatusPanel)
        if status.state == "Generating...":
            status.state = "Cancel generation first (Esc)"
            self.set_timer(2.0, lambda: setattr(status, "state", "Ready"))
            return
        self.push_screen(ModelSelector(), self._on_model_selected)

    def _on_model_selected(self, result: str | None) -> None:
        if result is None:
            return
        current = self.model_engine.model_size.lower().replace("b", ".", 1)
        if result == current:
            return
        self._start_switch(result)

    def _start_switch(self, size: str) -> None:
        self._switching_model = True
        self.query_one("#input-area", TextArea).disabled = True
        status = self.query_one(StatusPanel)
        status.state = f"Loading {size} model..."
        self.switch_worker(size)

    @work(thread=True, exclusive="switch")
    async def switch_worker(self, size: str) -> None:
        try:
            self.model_engine.reload(size)
            self.call_from_thread(self._on_switch_done, size)
        except Exception as exc:
            self.call_from_thread(self._on_switch_error, str(exc))

    def _on_switch_done(self, size: str) -> None:
        self._switching_model = False
        self.query_one("#input-area", TextArea).disabled = False
        self.query_one("#input-area", TextArea).focus()

        status = self.query_one(StatusPanel)
        status.model_size = self.model_engine.model_size
        status.voice_name = "none"
        status.voice_status = "✗ (none selected)"
        status.state = "Ready — voice: press F2"

        header = self.query_one(AppHeader)
        header.voice_name = "none"

        self.config["model"]["size"] = size
        self._write_config()

    def _on_switch_error(self, msg: str) -> None:
        self._switching_model = False
        self.query_one("#input-area", TextArea).disabled = False
        self.query_one("#input-area", TextArea).focus()
        self.query_one(StatusPanel).state = f"Switch failed: {msg}"

    def _write_config(self) -> None:
        config_path = PROJECT_DIR / "config.toml"
        lines = []
        for section, values in self.config.items():
            lines.append(f"[{section}]")
            for k, v in values.items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                elif isinstance(v, bool):
                    lines.append(f"{k} = {'true' if v else 'false'}")
                else:
                    lines.append(f"{k} = {v}")
            lines.append("")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    config_path = PROJECT_DIR / "config.toml"

    if not config_path.exists():
        print("✗ config.toml not found. Run bootstrap.py first.")
        sys.exit(1)

    import tomllib

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    valid_sizes = {"0.6b", "1.7b"}
    size = config.get("model", {}).get("size", "0.6b")
    if size not in valid_sizes:
        print(f"✗ Invalid model.size: {size!r} (valid: {', '.join(sorted(valid_sizes))})")
        sys.exit(1)

    device = config.get("model", {}).get("device", "cpu")
    if device != "cpu":
        print(f"✗ Only device=\"cpu\" is supported (got {device!r})")
        sys.exit(1)

    valid_formats = {"wav", "mp3"}
    fmt = config.get("output", {}).get("format", "wav")
    if fmt not in valid_formats:
        print(f"✗ Invalid output.format: {fmt!r} (valid: {', '.join(sorted(valid_formats))})")
        sys.exit(1)

    valid_naming = {"timestamp", "prompt-prefix"}
    naming = config.get("output", {}).get("naming", "timestamp")
    if naming not in valid_naming:
        print(f"✗ Invalid output.naming: {naming!r} (valid: {', '.join(sorted(valid_naming))})")
        sys.exit(1)

    return config


def _get_model_dir(config: dict) -> Path:
    size = config.get("model", {}).get("size", "0.6b")
    dir_name = f"{size}-base"
    model_dir = PROJECT_DIR / "models" / dir_name

    if not model_dir.exists():
        print(f"✗ Model directory not found: {model_dir}")
        print("  Run bootstrap.py first")
        sys.exit(1)

    return model_dir


def _resolve_initial_voice(config: dict) -> str | None:
    default = config.get("voice", {}).get("default", "samples/")
    default_path = PROJECT_DIR / default

    if default_path.is_dir():
        wavs = sorted(default_path.glob("*.wav"))
        for wav in wavs:
            txt = wav.with_suffix(".txt")
            if txt.exists():
                return str(wav)
        if wavs:
            return str(wavs[0])
        return None
    elif default_path.exists() and default_path.suffix == ".wav":
        txt = default_path.with_suffix(".txt")
        if txt.exists():
            return str(default_path)
        return str(default_path)
    return None


def main() -> None:
    lock_path = PROJECT_DIR / "bootstrap.lock"
    if not lock_path.exists():
        print("✗ bootstrap.lock not found. Run bootstrap.py first.")
        sys.exit(1)

    config = _load_config()
    model_dir = _get_model_dir(config)

    print("Loading model...")
    engine = ModelEngine(model_dir)
    session_log = SessionLog(
        log_dir=PROJECT_DIR / "logs",
        log_history=config.get("session", {}).get("log_history", True),
    )
    player = AudioPlayer()

    initial_voice = _resolve_initial_voice(config)
    if initial_voice:
        try:
            engine.load_voice(initial_voice)
            print(f"  Voice: {Path(initial_voice).stem}")
        except (FileNotFoundError, ValueError) as e:
            print(f"  ⚠ Could not load initial voice: {e}")

    app = TTSApp(
        model_engine=engine,
        session_log=session_log,
        audio_player=player,
        config=config,
    )
    app.run()


if __name__ == "__main__":
    main()
