#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()

TARGETS = [
    ("venv/", PROJECT_DIR / "venv"),
    ("Qwen3-TTS/", PROJECT_DIR / "Qwen3-TTS"),
    ("bootstrap.lock", PROJECT_DIR / "bootstrap.lock"),
]


def main() -> None:
    print("This will delete:")
    for label, path in TARGETS:
        if path.exists():
            print(f"  • {label}")
    print("\nKept: models/, config.toml, logs/, output/, samples/")

    response = input("\nContinue? [y/N] ").strip().lower()
    if response != "y":
        print("Aborted.")
        sys.exit(1)

    for label, path in TARGETS:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            print(f"  ✗ {label}")
        elif path.is_file():
            path.unlink(missing_ok=True)
            print(f"  ✗ {label}")
        else:
            print(f"  - {label} (not found)")

    print("\nReset complete. Run: python bootstrap.py")


if __name__ == "__main__":
    main()
