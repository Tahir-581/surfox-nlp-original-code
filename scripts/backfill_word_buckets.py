#!/usr/bin/env python3
"""Run word-bucket backfill from the repo root."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "backend" / "scripts" / "backfill_word_buckets.py"

if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [sys.executable, str(SCRIPT), *sys.argv[1:]],
            cwd=ROOT / "backend",
        )
    )
