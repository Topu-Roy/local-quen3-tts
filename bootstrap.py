#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys

from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()


def _activation_command() -> str:
    shell = os.environ.get("SHELL", "")
    if "fish" in shell:
        return "source venv/bin/activate.fish"
    if "csh" in shell or "tcsh" in shell:
        return "source venv/bin/activate.csh"
    return "source venv/bin/activate"


def _print_step(step: int, total: int, label: str) -> None:
    print(f"\n[{'='*50}]")
    print(f"  Phase {step}/{total}: {label}")
    print(f"[{'='*50}]\n")


def _confirm_import(module: str, package: str | None = None) -> bool:
    try:
        __import__(module)
        print(f"  ✓ {module} imported successfully")
        return True
    except Exception as e:
        hint = f" ({e})" if str(e) else ""
        print(f"  ✗ Failed to import {module}{hint}")
        return False


def _run_pip(args: list[str]) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install"] + args,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def _get_config() -> dict:
    config_path = PROJECT_DIR / "config.toml"
    default_path = PROJECT_DIR / "config.toml.default"

    if not config_path.exists():
        if default_path.exists():
            shutil.copy2(default_path, config_path)
            print(f"  Created config.toml from default template")
        else:
            print("  ✗ No config.toml or config.toml.default found")
            sys.exit(1)

    try:
        import tomllib
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"  ✗ Failed to parse config.toml: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Phase 1 — Environment checks
# ---------------------------------------------------------------------------

def phase1_env_checks() -> dict:
    _print_step(1, 7, "Environment Checks")

    # Python version
    py_ok = sys.version_info >= (3, 12)
    if not py_ok:
        print(f"  ✗ Python {sys.version_info.major}.{sys.version_info.minor} detected, need 3.12+")
        sys.exit(1)
    print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # Virtual environment
    venv_dir = PROJECT_DIR / "venv"
    in_venv = sys.prefix != sys.base_prefix

    if not venv_dir.exists():
        print("  Creating virtual environment...")
        subprocess.check_call(
            [sys.executable, "-m", "venv", str(venv_dir)],
            stdout=sys.stdout, stderr=sys.stderr,
        )
        if platform.system() == "Windows":
            print("  → Run: .\\venv\\Scripts\\activate")
        else:
            print(f"  → Run: {_activation_command()}")
        print("  → Then re-run: python bootstrap.py")
        sys.exit(0)

    if not in_venv:
        print(f"  ✗ venv/ exists but not activated")
        if platform.system() == "Windows":
            print("  → Run: .\\venv\\Scripts\\activate")
        else:
            print(f"  → Run: {_activation_command()}")
        print("  → Then re-run: python bootstrap.py")
        sys.exit(0)

    print(f"  ✓ Virtual environment active at {sys.prefix}")

    # Config
    config = _get_config()
    print(f"  ✓ Config loaded (model size: {config.get('model', {}).get('size', 'unknown')})")

    return config


# ---------------------------------------------------------------------------
# Phase 2 — Hardware detection
# ---------------------------------------------------------------------------

def _get_ram_gb() -> float:
    system = platform.system()
    try:
        if system == "Windows":
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            return mem.ullTotalPhys / (1024 ** 3)
        elif system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024 ** 3)
        else:
            mem_path = Path("/proc/meminfo")
            if mem_path.exists():
                for line in mem_path.read_text().splitlines():
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def phase2_hardware_detection(config: dict) -> None:
    _print_step(2, 7, "Hardware Detection")

    cpu_count = os.cpu_count() or 0
    ram_gb = _get_ram_gb()

    print(f"  CPU cores: {cpu_count}")
    print(f"  RAM:       {ram_gb:.1f} GB" if ram_gb else "  RAM:       (unknown)")
    print(f"  Platform:  {platform.system()} {platform.machine()}")

    if ram_gb and ram_gb < 8:
        print(f"\n  ⚠ WARNING: Less than 8GB RAM. 1.7B model may be very slow.")
    elif ram_gb:
        model_size = config.get("model", {}).get("size", "0.6b")
        if model_size == "1.7b" and ram_gb < 12:
            print(f"\n  ⚠ WARNING: 1.7B model with {ram_gb:.1f}GB RAM may be slow.")
            print(f"     Consider setting model.size = \"0.6b\" in config.toml")

    print(f"\n  {'='*40}")
    print(f"  CPU mode — generation will be slow (10-60s per sentence)")
    print(f"  {'='*40}\n")


# ---------------------------------------------------------------------------
# Phase 3 — Dependency installation
# ---------------------------------------------------------------------------

def phase3_dependency_installation() -> None:
    _print_step(3, 7, "Dependency Installation")

    # Step 1: Install PyTorch CPU
    print("  Installing PyTorch CPU...")
    try:
        import torch  # noqa: F401
        print("  ✓ PyTorch already installed, skipping")
    except (ImportError, OSError):
        _run_pip([
            "torch", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu",
        ])
        try:
            import torch  # noqa: F401
            print("  ✓ PyTorch installed successfully")
        except OSError:
            print("  ✗ PyTorch installed but native DLLs failed to load.")
            print("     Install Microsoft Visual C++ Redistributable:")
            print("     https://aka.ms/vs/17/release/vc_redist.x64.exe")
            print("     Then re-run bootstrap.py")
            sys.exit(1)

    # Step 2: Install requirements.txt
    req_file = PROJECT_DIR / "requirements.txt"
    if req_file.exists():
        print("  Installing requirements.txt...")
        _run_pip(["-r", str(req_file)])
    else:
        print("  ⚠ No requirements.txt found, skipping")

    # Verify key deps
    for mod, pkg in [("textual", None), ("soundfile", None), ("huggingface_hub", None)]:
        try:
            __import__(mod)
            print(f"  ✓ {mod} available")
        except ImportError:
            print(f"  ✗ {mod} not installed, installing...")
            _run_pip([pkg or mod])
            _confirm_import(mod)

    # Step 3: Clone and install Qwen3-TTS from GitHub
    qwen_dir = PROJECT_DIR / "Qwen3-TTS"
    if qwen_dir.exists() and (qwen_dir / "pyproject.toml").exists():
        print("  Qwen3-TTS source already cloned, updating...")
        try:
            subprocess.check_call(
                ["git", "-C", str(qwen_dir), "pull", "--ff-only"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            print("  ⚠ Could not update, using existing")
    else:
        print("  Cloning Qwen3-TTS from GitHub...")
        if qwen_dir.exists():
            shutil.rmtree(qwen_dir)
        subprocess.check_call(
            ["git", "clone", "https://github.com/QwenLM/Qwen3-TTS.git", str(qwen_dir)],
            stdout=sys.stdout, stderr=sys.stderr,
        )

    print("  Installing qwen-tts from source...")
    _run_pip(["-e", str(qwen_dir)])
    if not _confirm_import("qwen_tts"):
        print("  ✗ qwen-tts import failed. Try: pip install qwen-tts")
        sys.exit(1)
    print("  ✓ qwen-tts installed successfully")


# ---------------------------------------------------------------------------
# Phase 4 — ffmpeg check
# ---------------------------------------------------------------------------

def phase4_ffmpeg_check() -> None:
    _print_step(4, 7, "FFmpeg Check")

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            version_line = result.stdout.splitlines()[0] if result.stdout else ""
            print(f"  ✓ ffmpeg available: {version_line}")
        else:
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  ⚠ ffmpeg not found")
        print("     mp3 output disabled, forcing format = \"wav\" in config.toml")
        _patch_config({"output": {"format": "wav"}})


def _patch_config(updates: dict) -> None:
    config_path = PROJECT_DIR / "config.toml"
    if not config_path.exists():
        return

    try:
        import tomllib
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return

    changed = False
    for section, values in updates.items():
        if section not in data:
            data[section] = {}
        for key, val in values.items():
            if data[section].get(key) != val:
                data[section][key] = val
                changed = True

    if changed:
        lines = []
        for section, values in data.items():
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
        print(f"     config.toml updated to force format=wav")


# ---------------------------------------------------------------------------
# Phase 5 — Model download
# ---------------------------------------------------------------------------

MODEL_MAP = {
    "0.6b": ("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "0.6b-base"),
    "1.7b": ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "1.7b-base"),
}
TOKENIZER_REPO = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
TOKENIZER_DIR = "tokenizer"
MODELS_DIR = PROJECT_DIR / "models"


def _download_model(repo_id: str, local_dir: Path, label: str) -> None:
    from huggingface_hub import snapshot_download

    download_dir = local_dir.with_suffix(local_dir.suffix + ".download") if local_dir.suffix else local_dir.parent / (local_dir.name + ".download")

    if download_dir.exists():
        shutil.rmtree(download_dir)

    print(f"  Downloading {label} ({repo_id})...")
    snapshot_download(
        repo_id,
        local_dir=str(download_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
    )

    # Verify key files
    safetensors = list(download_dir.rglob("*.safetensors"))
    if not safetensors:
        print(f"  ✗ No safetensors found in download, may be incomplete")
        shutil.rmtree(download_dir)
        sys.exit(1)
    print(f"  ✓ {len(safetensors)} safetensors files verified")

    # Atomic move
    if local_dir.exists():
        shutil.rmtree(local_dir)
    download_dir.rename(local_dir)
    print(f"  → Saved to {local_dir}")


def _has_safetensors(path: Path) -> bool:
    return path.exists() and bool(list(path.rglob("*.safetensors")))


def phase5_model_download(config: dict) -> None:
    _print_step(5, 7, "Model Download")

    MODELS_DIR.mkdir(exist_ok=True)
    tokenizer_dir = MODELS_DIR / TOKENIZER_DIR

    # Tokenizer (shared, download once)
    if _has_safetensors(tokenizer_dir):
        print(f"  ✓ Tokenizer already downloaded at {tokenizer_dir}")
    else:
        _download_model(TOKENIZER_REPO, tokenizer_dir, "Tokenizer")

    # Both models (skip if already present)
    for size_key, (repo_id, dir_name) in MODEL_MAP.items():
        model_dir = MODELS_DIR / dir_name
        if _has_safetensors(model_dir):
            print(f"  ✓ {size_key} model already downloaded at {model_dir}")
        else:
            _download_model(repo_id, model_dir, f"{size_key} model")


# ---------------------------------------------------------------------------
# Phase 6 — Sample validation
# ---------------------------------------------------------------------------

def phase6_sample_validation() -> None:
    _print_step(6, 7, "Sample Validation")

    samples_dir = PROJECT_DIR / "samples"
    if not samples_dir.exists():
        samples_dir.mkdir(exist_ok=True)

    # Auto-convert non-wav audio files to wav
    audio_extensions = ("*.webm", "*.mp3", "*.ogg", "*.flac", "*.m4a", "*.mp4", "*.opus")
    converted = 0
    for pattern in audio_extensions:
        for audio_path in samples_dir.glob(pattern):
            wav_path = audio_path.with_suffix(".wav")
            if wav_path.exists():
                continue
            txt_path = audio_path.with_suffix(".txt")
            print(f"  Converting {audio_path.name} to wav...")
            result = subprocess.run(
                ["ffmpeg", "-i", str(audio_path),
                 "-ac", "1", "-ar", "16000",
                 "-y", str(wav_path)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                converted += 1
                has_txt = txt_path.exists() and txt_path.stat().st_size > 0
                if not has_txt:
                    print(f"    ✓ converted (missing transcript .txt!)")
                else:
                    print(f"    ✓ converted")
            else:
                print(f"    ✗ conversion failed: {result.stderr.strip()[-200:]}")

    wav_files = sorted(samples_dir.glob("*.wav"))
    if not wav_files:
        print("  ⚠ No .wav files found in samples/")
        print("     Voice cloning requires at least one sample.")
        print("     Add a .wav file and matching .txt transcript to samples/")
        return

    from scipy.io import wavfile

    valid_count = 0
    for wav_path in wav_files:
        txt_path = wav_path.with_suffix(".txt")
        issues = []

        try:
            sr, data = wavfile.read(str(wav_path))
        except Exception as e:
            print(f"  ✗ {wav_path.name}: unreadable ({e})")
            continue

        if sr != 16000:
            issues.append(f"sample rate {sr}Hz (prefer 16000)")
        channels = data.ndim if data.ndim == 1 else data.shape[1]
        if channels != 1:
            issues.append(f"{channels} channels (prefer mono)")
        duration = data.shape[0] / sr
        if duration < 3:
            issues.append(f"only {duration:.1f}s (min 3s)")
        elif duration > 30:
            issues.append(f"{duration:.1f}s (max 30s)")

        has_txt = txt_path.exists() and txt_path.stat().st_size > 0
        if not has_txt:
            issues.append("missing transcript (.txt)")

        status = "✓" if not issues else "✗"
        reason = f" — {'; '.join(issues)}" if issues else ""
        print(f"  {status} {wav_path.name}{reason}")

        if not issues:
            valid_count += 1

    if valid_count == 0 and wav_files:
        print("\n  ⚠ No valid samples found. Check the warnings above.")
    elif valid_count > 0:
        print(f"\n  ✓ {valid_count}/{len(wav_files)} samples valid")


# ---------------------------------------------------------------------------
# Phase 7 — Final report
# ---------------------------------------------------------------------------

def phase7_final_report(config: dict) -> None:
    _print_step(7, 7, "Final Report")

    samples = list((PROJECT_DIR / "samples").glob("*.wav")) if (PROJECT_DIR / "samples").exists() else []

    downloaded_sizes = [k for k in MODEL_MAP if (MODELS_DIR / MODEL_MAP[k][1]).exists()]

    print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    print(f"  ✓ Virtual environment active")
    print(f"  ✓ Dependencies installed")
    print(f"  ✓ Qwen3-TTS source installed")
    for size_key in downloaded_sizes:
        print(f"  ✓ Model: Qwen3-TTS-12Hz-{size_key}-Base")
    print(f"  ✓ Tokenizer: Qwen3-TTS-Tokenizer-12Hz")
    print(f"  ✓ Samples: {len(samples)} found")

    if not samples:
        print("  ⚠ No voice samples — add .wav + .txt to samples/ before first use")

    # Write bootstrap.lock
    lock = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "models": downloaded_sizes,
        "device": "cpu",
        "samples_count": len(samples),
    }
    lock_path = PROJECT_DIR / "bootstrap.lock"
    with open(lock_path, "w") as f:
        json.dump(lock, f, indent=2)
    print(f"\n  ✓ bootstrap.lock written")

    print(f"\n  {'='*50}")
    print(f"  Bootstrap complete! Run:")
    print(f"    python tts.py")
    print(f"  {'='*50}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n  Qwen3-TTS TUI — Bootstrap v1.0")
    print(f"  Project: {PROJECT_DIR}\n")

    config = phase1_env_checks()
    phase2_hardware_detection(config)
    phase3_dependency_installation()
    phase4_ffmpeg_check()
    phase5_model_download(config)
    phase6_sample_validation()
    phase7_final_report(config)


if __name__ == "__main__":
    main()
