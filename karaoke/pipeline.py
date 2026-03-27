"""Full karaoke pipeline: YouTube URL → karaoke video."""

import torch
from pathlib import Path

from .download import download
from .separate import separate
from .transcribe import transcribe
from .subtitles import build_ass
from .compose import compose


def run(url: str, output_dir: Path) -> Path:
    """
    Run the full pipeline for a YouTube URL.

    Steps:
        1. Download video + audio
        2. Separate vocals from instrumental
        3. Transcribe vocals with word-level timestamps
        4. Generate ASS karaoke subtitles
        5. Compose final video

    Returns the path to the output karaoke video.
    """
    device = "cpu"  # MPS lacks sparse tensor support needed by Whisper
    print(f"Using device: {device}")

    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Downloading video and audio...")
    video_path, audio_path = download(url, work_dir)

    print("[2/5] Separating vocals from instrumental...")
    instrumental_path, vocals_path = separate(audio_path, work_dir / "demucs")

    print("[3/5] Transcribing lyrics...")
    # Transcribe from original mixed audio — Whisper is trained on full mixes
    # and produces far more reliable timestamps than from the demucs vocals stem,
    # which contains phase artifacts that confuse VAD and alignment.
    segments = transcribe(audio_path, device=device)
    word_count = sum(len(s.words) for s in segments)
    print(f"      Found {len(segments)} segments, {word_count} words")

    print("[4/5] Building subtitle file...")
    subtitles_path = work_dir / "karaoke.ass"
    build_ass(segments, subtitles_path)

    print("[5/5] Composing final video...")
    output_path = output_dir / "karaoke.mp4"
    compose(video_path, instrumental_path, subtitles_path, output_path)

    print(f"\nDone! Output: {output_path}")
    return output_path
