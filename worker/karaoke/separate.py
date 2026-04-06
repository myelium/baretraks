"""Separate vocals from audio using Demucs."""

import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

# htdemucs_ft: higher quality, ~3GB RAM
# htdemucs: lighter, ~1.5GB RAM — use when memory is tight
DEMUCS_MODEL = os.getenv("DEMUCS_MODEL", "htdemucs_ft")

# htdemucs_ft is a bag of 4 models
_MODEL_PASSES = {"htdemucs_ft": 4, "htdemucs": 1}


def separate(audio_path: Path, output_dir: Path, device: str = "cpu",
             model: str | None = None,
             progress_callback: Callable[[float], None] | None = None) -> tuple[Path, Path]:
    """
    Run Demucs htdemucs model to separate vocals and instrumental.

    Args:
        progress_callback: Optional callback receiving progress as 0.0–1.0

    Returns:
        (instrumental_path, vocals_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    use_model = model or DEMUCS_MODEL
    total_passes = _MODEL_PASSES.get(use_model, 4)

    cmd = [
        sys.executable, "-m", "demucs",
        "-n", use_model,
        "--two-stems", "vocals",
        "-d", device,
        "--out", str(output_dir),
        str(audio_path),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    completed_passes = 0
    last_pct = 0

    # Read stderr byte by byte to catch \r-delimited progress updates
    buf = b''
    while True:
        ch = proc.stderr.read(1)
        if not ch:
            break
        if ch in (b'\n', b'\r'):
            line = buf.decode('utf-8', errors='replace')
            buf = b''
            if line.strip():
                print(line, flush=True)
                m = re.search(r'(\d+)%\|', line)
                if m:
                    pct = int(m.group(1))
                    if pct == 100 and last_pct < 100:
                        completed_passes += 1
                    last_pct = pct
                    if progress_callback:
                        overall = (completed_passes + pct / 100.0) / total_passes
                        progress_callback(min(overall, 0.99))
        else:
            buf += ch

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Demucs failed (exit {proc.returncode})")

    if progress_callback:
        progress_callback(1.0)

    # Demucs outputs to: output_dir/<model>/<stem_name>/{vocals,no_vocals}.wav
    stem_name = audio_path.stem
    demucs_out = output_dir / use_model / stem_name
    instrumental_path = demucs_out / "no_vocals.wav"
    vocals_path = demucs_out / "vocals.wav"

    if not instrumental_path.exists():
        raise FileNotFoundError(f"Demucs output not found: {instrumental_path}")

    return instrumental_path, vocals_path
