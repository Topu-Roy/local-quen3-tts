## Full Implementation Plan (Corrected)

---

### 1. Dependencies & Pinning Strategy

**`requirements.txt` will have three sections:**

- **Core ML** — PyTorch (CPU version), torchaudio, `soundfile` for writing wav output
- **TUI** — Textual (pinned), Rich (comes with Textual but pin it anyway)
- **Utilities** — `tomllib` (stdlib in 3.11+, else `tomli` for older Python), `sounddevice` for audio playback, `huggingface_hub` for model download, `scipy` for wav validation

**Key decision:** Qwen3-TTS is installed from GitHub source (not PyPI). Bootstrap clones `https://github.com/QwenLM/Qwen3-TTS.git` and runs `pip install -e ./Qwen3-TTS`. This is done in Phase 3 after PyTorch is installed.

No CUDA — CPU-only PyTorch with no special index URL needed. No flash-attention (CUDA-only).

---

### 2. `bootstrap.py` — Detailed Flow

**Phase 1 — Environment checks**
- Check Python is 3.9+, exit with message if not
- Check if `venv/` exists inside the project folder
  - If not: create it automatically, then print the OS-specific activation command and exit telling the user to activate and re-run bootstrap
  - If yes but not currently activated: detect this (check `sys.prefix`), print activation command, exit
- Check `config.toml` exists — if not, copy from a bundled `config.toml.default`

**Phase 2 — Hardware detection**
- Detect CPU core count and RAM → warn if under 8GB RAM (will be very slow)
- Print a clean hardware summary table
- Print a visible warning box: "CPU mode — generation will be slow (10-30s per sentence)"

**Phase 3 — Dependency installation**
- Install PyTorch CPU via pip: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`
- Install remaining `requirements.txt` packages
- Clone Qwen3-TTS: `git clone https://github.com/QwenLM/Qwen3-TTS.git` → `Qwen3-TTS/`
- Install from source: `pip install -e ./Qwen3-TTS`
- Re-validate after install — confirm `from qwen_tts import Qwen3TTSModel` actually works

**Phase 4 — ffmpeg check**
- Run `ffmpeg -version` as a subprocess
- If found: note version, mp3 output is available
- If not found: update `config.toml` to force `output_format = "wav"`, print a note explaining why

**Phase 5 — Model download**
- Check if `models/tokenizer/` is populated with expected files
- Read `model.size` from `config.toml` to determine which model to download
- Check if `models/tokenizer/` is populated with expected files
- Check if `models/{size}-base/` is populated with expected files (e.g., `models/0.6b-base/` or `models/1.7b-base/`)
- For each missing model:
  - Show a progress bar using Rich (huggingface_hub provides download progress hooks)
  - Download to a temp subfolder first, then move on completion — prevents partial downloads looking like successful ones
  - After move: verify key weight files exist and are non-zero bytes
- Models to download (tokenizer always; base model based on config):
  - `huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./models/tokenizer`
  - `huggingface-cli download Qwen/Qwen3-TTS-12Hz-{size}-Base --local-dir ./models/{size}-base`
- If all models already present: skip with a "models already downloaded" message

**Phase 6 — Sample validation**
- Scan `samples/` folder
- For each `.wav` file found:
  - Load with scipy, check: sample rate (warn if not 16kHz), duration (warn if under 3s or over 30s), channels (warn if stereo — mono preferred)
  - Check for matching `.txt` transcript file (e.g., `samples/john.wav` → `samples/john.txt`)
  - Print a per-file status: filename + ✓ or ✗ + reason
- If `samples/` is empty: print a note explaining voice cloning requires at least one sample with transcript

**Phase 7 — Final report**
- Print a summary: what passed, what warned, what failed
- Print exact command to launch: `python tts.py`
- Write a `bootstrap.lock` file with timestamp and detected config — tts.py reads this to know bootstrap was completed

---

### 3. `config.toml` — Full Spec

```toml
[model]
size = "0.6b"           # "0.6b" or "1.7b" (Base model, no builtin speakers)
device = "cpu"          # "cpu" only (no CUDA/MPS)

[voice]
default = "samples/"    # first valid .wav in samples/ folder, or explicit path

[output]
format = "wav"          # "wav" or "mp3" (mp3 requires ffmpeg)
naming = "timestamp"    # "timestamp" or "prompt-prefix"
folder = "output"       # relative to project root
auto_play = true        # play file after generation

[session]
log_history = true      # save text+filename to logs/
show_history = true     # show history panel in TUI
```

Config is loaded at `tts.py` startup and validated against an expected schema (required keys, valid values). Any bad value prints a clear error pointing to the specific key.

**Key difference from original:** No `builtin` voice keyword. The Base model has no builtin speakers — all voices come from reference samples in `samples/`. The `default` setting points to a path; if it's a directory, the first valid `.wav` with a matching `.txt` is used.

---

### 4. `tts.py` — Architecture

The file is the entry point but it orchestrates three internal components:

**Component A: `ModelEngine`**
A class that owns the model. Responsibilities:
- Load model from `./models/{size}-base/` based on `config.model.size` via `Qwen3TTSModel.from_pretrained(...)` with a progress callback (so TUI can show stages)
- `generate(text, voice_sample_path)` method:
  1. Read voice sample path (`.wav`)
  2. Read transcript from matching `.txt` file
  3. Call `model.create_voice_clone_prompt(ref_audio=path, ref_text=transcript)` — first time only, cache result
  4. Call `model.generate_voice_clone(text=text, language="auto", voice_clone_prompt=cached_prompt)`
  5. Return `(wav_array, sample_rate)`
- Cache voice clone prompts keyed by sample path for reuse
- Run generation in a **background thread** — never blocks the TUI thread
- Expose a `cancel()` method that sets a flag the generation loop checks
- After each generation, explicitly delete intermediate tensors
- Track total generations this session for the status panel

**Component B: `SessionLog`**
A simple class that manages the session history. Responsibilities:
- Keep an in-memory list of `{timestamp, text, output_file, voice_used}` entries
- If `log_history = true` in config, append each entry to `logs/YYYY-MM-DD.jsonl` (one JSON object per line, easy to parse later)
- Expose a method to get recent entries for the history panel

**Component C: `AudioPlayer`**
A small utility class. Responsibilities:
- Detect OS (Linux/Mac/Windows) once at init
- Expose a `play(filepath)` method that uses the right system command per OS
- Run playback in a subprocess so it never blocks
- Expose a `stop()` method

---

### 5. TUI Layout — Full Detail

Built with Textual. The app has one screen with a fixed layout:

**Header bar (top, 1 line)**
- App name
- Device indicator: `CPU ⚠` (always, since CPU-only)
- Currently active voice name (filename without extension)
- Generation count this session

**Main area (split horizontally)**

Left panel — History (40% width, scrollable):
- Each entry shows: time, output filename, first ~50 chars of text
- Clicking an entry highlights it (future: could re-play)
- Auto-scrolls to latest on new entry

Right panel — Status (60% width):
- Model loaded: `Qwen3-TTS-12Hz-{size}-Base` (from config)
- Device: `CPU`
- Current voice + validation status (✓ transcript found / ✗ missing transcript)
- CPU speed notice: `CPU mode — generation may be slow (10-30s)`
- Current state: `Ready` / `Generating...` / `Cancelled`
- During generation: a progress indicator (indeterminate spinner since we can't know exact duration)
- Last generated filename
- Key bindings reminder at bottom

**Input area (bottom, fixed)**
- A text input widget, full width
- Placeholder text: `Type text and press Enter to generate`
- `Shift+Enter` inserts a newline for multi-sentence input
- Disabled (grayed out) during generation, re-enabled on completion or cancel

**Footer (bottom, 1 line)**
- `[Enter] Generate  [F2] Voice  [F3] Folder  [F4] Play last  [Esc] Cancel  [q] Quit`

---

### 6. Key Interactions — Detailed Behavior

**Enter (generate)**
1. Read text from input field
2. Validate not empty
3. Validate active voice sample still exists and has transcript
4. Disable input field
5. Update status panel to `Generating...` with spinner
6. Call `ModelEngine.generate()` in background thread
7. On completion: write file via `soundfile.write()`, add to `SessionLog`, update history panel, update status to `Ready`, re-enable input, auto-play if configured
8. Input field is cleared and focused

**Esc (cancel)**
1. If not generating: do nothing
2. If generating: call `ModelEngine.cancel()`
3. Status updates to `Cancelled` briefly then back to `Ready`
4. Input re-enabled immediately
5. No partial file written

**F2 (change voice)**
1. Open a modal overlay panel
2. List all `.wav` files in `samples/` with transcript status (✓/✗)
3. Arrow keys to navigate, Enter to select, Esc to cancel
4. On selection: call `ModelEngine.load_voice(sample_path)` which reads the transcript, builds the clone prompt, caches it; update header bar, close modal

**F3 (open output folder)**
1. Run OS-appropriate command to open `output/` in file manager
2. Linux: `xdg-open`, Mac: `open`, Windows: `explorer`

**F4 (play last)**
1. If no file generated yet: show brief message in status
2. Otherwise: call `AudioPlayer.play()` with last generated file path

**q (quit)**
1. If generating: ask for confirmation via a small modal (`Generation in progress. Quit anyway? [y/n]`)
2. Otherwise: clean exit, flush `SessionLog` if needed

---

### 7. Output File Naming

Two modes from config:

**`timestamp` mode:**
`output/2026-05-25_14-32-01.wav`

**`prompt-prefix` mode:**
Takes first 4 words of input text, lowercases, replaces spaces with dashes, strips non-alphanumeric, appends timestamp to prevent collision:
`output/hello-how-are-you_14-32-01.wav`

---

### 8. Session Log Format

Each line in `logs/2026-05-25.jsonl`:
```
{"time": "14:32:01", "text": "Hello world", "file": "output/2026-05-25_14-32-01.wav", "voice": "john.wav", "device": "cpu", "duration_ms": 3241}
```

Gives you a full searchable record of every generation. Plain text, one line per entry, never overwrites.

---

### 9. Error Handling Strategy

Every possible failure has a defined outcome:

| Failure | Behavior |
|---|---|
| Model files missing | Hard exit at startup with message pointing to bootstrap |
| Config key missing or invalid | Hard exit listing exactly which key and what valid values are |
| Voice sample missing transcript (.txt) | Show error in status panel, prompt user to create one |
| Generation throws exception | Status shows error message, input re-enabled, session continues |
| Voice sample file deleted mid-session | Show error in status panel, disable generate until new voice selected |
| Output folder missing | Create it automatically |
| Disk full on save | Show error in status panel, don't crash |
| Audio playback fails | Silent failure with a status note — not critical |
| OOM during generation | Caught, status shows memory warning |

---

### 10. File Execution Order (What Runs When)

**One-time setup:**
```
python bootstrap.py
  → creates venv
  → installs deps (including qwen-tts from GitHub source)
  → downloads models (tokenizer + {size}-base per config)
  → validates samples + transcripts
  → writes bootstrap.lock
```

**Every session:**
```
python tts.py
  → reads bootstrap.lock (exits if missing)
  → loads config.toml (validates all keys)
  → initializes ModelEngine (loads model with progress)
  → initializes SessionLog
  → initializes AudioPlayer
  → launches Textual TUI
  → enters event loop
```

---

### 11. What Is Explicitly Out of Scope

To keep this buildable and not a forever project:

- No audio recording inside the tool (you provide pre-recorded samples)
- No batch mode (file of texts in → multiple wavs out) — can be added later
- No web UI or API server mode
- No model switching mid-session (change config, restart)
- No waveform visualization
- No undo / regenerate last prompt (could be added trivially later using session log)
- No builtin/default voice — all voices require a reference sample + transcript
- No CUDA/GPU support — CPU-only

---

### 12. Key Technical Decisions Summary

| Decision | Choice | Reason |
|---|---|---|
| Inference package | `qwen-tts` from GitHub source | Official Qwen3-TTS Python package |
| Model type | `Qwen3-TTS-12Hz-{size}-Base` (0.6B or 1.7B) | Voice clone only, no builtin speakers; config selects size |
| Voice method | `create_voice_clone_prompt()` + `generate_voice_clone()` | Official API for voice cloning |
| Device | CPU, `dtype=torch.float32` | No NVIDIA GPU available |
| Transcripts | `.txt` file per `.wav` sample | Required by `create_voice_clone_prompt()` for ICL mode |
| Output writing | `soundfile.write()` | Qwen returns numpy arrays, sf is dependency of qwen-tts |
| Language detection | `language="auto"` | Model auto-detects from text |
| Prompt caching | Cache `voice_clone_prompt` per sample path | Avoids re-encoding reference audio on every generation |

---

