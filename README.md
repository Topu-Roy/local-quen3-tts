# Qwen3-TTS TUI

A **Textual** terminal user interface for **Qwen3-TTS** voice cloning on CPU. Clone any voice from a 3-second audio sample and synthesize speech across 10 languages.

![CPU only](https://img.shields.io/badge/CPU-only-orange)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

---

## Features

- **Voice cloning** — Provide a `.wav` reference + `.txt` transcript; the model reproduces that voice
- **10 languages** — Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- **Auto language detection** — Just type; the model detects the language automatically
- **Session history** — Every generation logged to `logs/YYYY-MM-DD.jsonl`
- **Multi-line input** — Shift+Enter for newlines; Generate button to submit
- **Two model sizes** — `0.6B` (faster) or `1.7B` (better quality); switch at runtime with **F5**
- **Auto-format conversion** — Drop `.webm`, `.mp3`, `.ogg`, etc. into `samples/`; bootstrap auto-converts to `.wav`

---

## Requirements

- **Python 3.12.x only** (not 3.11, not 3.13+)
- **~8GB free disk space** (both models + venv)
- **ffmpeg** (optional — required for mp3 output and non-wav sample conversion)
- **Microsoft Visual C++ Redistributable** (Windows only)

---

## Setup

### Step 1 — Verify Python version

```bash
python --version   # must be 3.12.x
```

If `python` points to 3.13+ or 3.11, use `python3.12` instead in all commands below.

### Step 2 — Clone and enter the repo

```bash
git clone https://github.com/Topu-Roy/local-quen3-tts.git
cd local-quen3-tts
```

### Step 3 — First bootstrap pass

```bash
python bootstrap.py
```

This creates the virtual environment in `venv/`, then exits. You will see:

```
→ Run: source venv/bin/activate.fish    (or .bash, .csh)
→ Then re-run: python bootstrap.py
```

### Step 4 — Activate the virtual environment

```bash
source venv/bin/activate.fish    # if using fish
source venv/bin/activate         # if using bash/zsh
source venv/bin/activate.csh     # if using csh/tcsh
.\venv\Scripts\activate          # if using Windows
```

### Step 5 — Second bootstrap pass

```bash
python bootstrap.py
```

This installs PyTorch CPU, Qwen3-TTS from GitHub, all Python dependencies, downloads both models (0.6B + 1.7B) and the tokenizer from Hugging Face, validates voice samples, and writes `bootstrap.lock`.

Bootstrap runs through 7 phases:

| Phase | What it does |
|---|---|
| 1 | Checks Python 3.12, creates venv (first pass), loads config |
| 2 | Detects CPU cores, RAM, platform — warns if low memory |
| 3 | Installs PyTorch CPU, pip deps, clones + installs Qwen3-TTS from GitHub |
| 4 | Checks for ffmpeg — forces `format = "wav"` if missing |
| 5 | Downloads tokenizer + both base models from Hugging Face (~5GB total) |
| 6 | Scans `samples/`, auto-converts non-wav files, validates `.wav`+`.txt` pairs |
| 7 | Writes `bootstrap.lock`, prints launch command |

---

## Voice Samples

Place audio samples in `samples/`. Each needs a matching transcript:

```
samples/
├── my-voice.wav       # audio (16 kHz mono preferred, 3–30 seconds)
└── my-voice.txt       # verbatim transcript of what is spoken in the audio
```

Supported input formats: `.wav`, `.webm`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.opus` — non-wav files are auto-converted by bootstrap phase 6.

---

## Configuration

Edit `config.toml` (auto-generated from `config.toml.default` on first bootstrap run):

```toml
[model]
size = "0.6b"           # "0.6b" or "1.7b" — initial model; switch at runtime with F5
device = "cpu"

[voice]
default = "samples/"    # first valid .wav found in samples/, or explicit path

[output]
format = "wav"          # "wav" or "mp3" (mp3 requires ffmpeg)
naming = "timestamp"    # "timestamp" or "prompt-prefix"
folder = "output"
auto_play = true        # play generated audio immediately after generation

[session]
log_history = true      # save each generation to logs/YYYY-MM-DD.jsonl
show_history = true     # show the history panel in the TUI
```

---

## Usage

```bash
python tts.py
```

### Keybindings

| Key | Action |
|---|---|
| `F2` | Select voice from available samples |
| `F3` | Open output folder in file manager |
| `F4` | Play last generated audio |
| `F5` | Switch model (0.6B ↔ 1.7B) at runtime |
| `Escape` | Cancel current generation |
| `Ctrl+Q` | Quit |

> **Note:** Use the **Generate** button on screen to submit text for speech synthesis. Enter inserts a newline; Shift+Enter also inserts a newline.

### Output naming

- **Timestamp mode:** `output/2026-05-25_14-32-01.wav`
- **Prompt-prefix mode:** `output/hello-how-are-you_14-32-01.wav`

### Session logs

Each generation is logged to `logs/YYYY-MM-DD.jsonl`:

```json
{"time": "14:32:01", "text": "Hello world", "file": "output/2026-05-25_14-32-01.wav", "voice": "my-voice.wav", "device": "cpu", "duration_ms": 3241}
```

---

## Project Structure

```
local-quen3-tts/
├── bootstrap.py          # One-time setup (venv, deps, models, samples)
├── tts.py                # Main app (TUI + ModelEngine + SessionLog + AudioPlayer)
├── config.toml.default   # Default config template
├── requirements.txt      # Python dependencies
├── APP.md                # Full implementation plan
├── VERSION               # Semantic version (0.5.0)
├── .gitignore
├── samples/              # Voice samples (.wav + .txt per voice)
├── logs/                 # Session history (.jsonl)
└── output/               # Generated audio (.wav)
```

Generated by bootstrap (not committed):

```
├── venv/                 # Python virtual environment
├── models/
│   ├── tokenizer/        # Qwen3-TTS-Tokenizer-12Hz
│   ├── 0.6b-base/        # Qwen3-TTS-12Hz-0.6B-Base
│   └── 1.7b-base/        # Qwen3-TTS-12Hz-1.7B-Base
├── Qwen3-TTS/            # Cloned GitHub source
├── config.toml           # User config (copied from .default)
└── bootstrap.lock        # Bootstrap completion marker
```

---

## Technical Stack

| Component | Technology |
|---|---|
| Inference | [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (via `qwen_tts` package) |
| TUI | [Textual](https://textual.textualize.io/) |
| Audio playback | `winsound` (Windows) / `afplay` (macOS) / `aplay` (Linux) |
| Model download | Hugging Face Hub |
| Audio processing | ffmpeg, scipy, soundfile |

---

## Out of Scope

- No audio recording (provide pre-recorded samples)
- No batch mode
- No web UI or API server
- No waveform visualization
- No builtin/default voice (all voices require a reference sample)
- No CUDA/GPU support

---

## License

Apache 2.0 — see [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for the underlying model license.
