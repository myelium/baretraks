"""Separate vocals from audio using Demucs."""

import subprocess
import sys
from pathlib import Path


def separate(audio_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """
    Run Demucs htdemucs model to separate vocals and instrumental.

    Returns:
        (instrumental_path, vocals_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable, "-m", "demucs",
            "-n", "htdemucs_ft",
            "--two-stems", "vocals",   # only split into vocals + no_vocals
            "-d", "cpu",               # MPS lacks sparse tensor support
            "--out", str(output_dir),
            str(audio_path),
        ],
        check=True,
    )

    # Demucs outputs to: output_dir/htdemucs_ft/<stem_name>/{vocals,no_vocals}.wav
    stem_name = audio_path.stem
    demucs_out = output_dir / "htdemucs_ft" / stem_name
    instrumental_path = demucs_out / "no_vocals.wav"
    vocals_path = demucs_out / "vocals.wav"

    if not instrumental_path.exists():
        raise FileNotFoundError(f"Demucs output not found: {instrumental_path}")

    return instrumental_path, vocals_path
