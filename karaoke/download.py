"""Download video and audio from a YouTube URL using yt-dlp."""

import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path


def _run_ytdlp_with_progress(
    args: list[str],
    progress_callback: Callable[[float], None] | None,
) -> None:
    """Run a yt-dlp command, optionally parsing download progress."""
    if progress_callback is None:
        subprocess.run(args, check=True)
        return

    proc = subprocess.Popen(
        args + ["--newline"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        # yt-dlp progress lines look like: [download]  45.2% of 12.34MiB ...
        m = re.search(r"\[download\]\s+([\d.]+)%", line)
        if m:
            progress_callback(float(m.group(1)) / 100.0)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args[0])


def fetch_metadata(url: str) -> dict:
    """Fetch video metadata (title, duration, thumbnail, channel, etc.) without downloading."""
    fields = "%(title)s\n%(duration)s\n%(thumbnail)s\n%(channel)s\n%(upload_date)s\n%(categories)s\n%(tags)s"
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp",
         "--no-playlist", "--print", fields,
         url],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.strip().split("\n")

    def _get(idx: int, default: str = "") -> str:
        return lines[idx] if idx < len(lines) and lines[idx] != "NA" else default

    # Parse categories and tags from Python list repr (e.g. "['Music', 'Pop']")
    def _parse_list(raw: str) -> list[str]:
        raw = raw.strip()
        if not raw or raw == "NA":
            return []
        # yt-dlp prints Python-style lists: ['a', 'b']
        try:
            import ast
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
        return []

    return {
        "title": _get(0, "Unknown"),
        "duration": float(_get(1, "0")) if _get(1, "0").replace(".", "").isdigit() else 0,
        "thumbnail": _get(2) or None,
        "channel": _get(3) or None,
        "upload_date": _get(4) or None,
        "categories": _parse_list(_get(5)),
        "tags": _parse_list(_get(6)),
    }


def download(
    url: str,
    output_dir: Path,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[Path, Path]:
    """
    Download YouTube video and audio separately.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save files into.
        progress_callback: Optional callback receiving progress as 0.0–1.0
            across both downloads (video = 0–0.5, audio = 0.5–1.0).

    Returns:
        (video_path, audio_path) — video is video-only, audio is audio-only wav
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / "video.mp4"
    audio_path = output_dir / "audio.wav"

    yt_dlp = [sys.executable, "-m", "yt_dlp"]

    # Download best video (no audio) — first half of progress
    def _video_progress(pct: float) -> None:
        if progress_callback:
            progress_callback(pct * 0.5)

    _run_ytdlp_with_progress(
        yt_dlp + [
            "-f", "bestvideo[ext=mp4]",
            "-o", str(video_path),
            "--no-playlist",
            url,
        ],
        progress_callback=_video_progress if progress_callback else None,
    )

    # Download best audio and convert to wav — second half of progress
    def _audio_progress(pct: float) -> None:
        if progress_callback:
            progress_callback(0.5 + pct * 0.5)

    _run_ytdlp_with_progress(
        yt_dlp + [
            "-f", "bestaudio",
            "-o", str(audio_path),
            "--extract-audio",
            "--audio-format", "wav",
            "--no-playlist",
            url,
        ],
        progress_callback=_audio_progress if progress_callback else None,
    )

    return video_path, audio_path
