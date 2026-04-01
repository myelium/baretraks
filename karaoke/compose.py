"""Compose final karaoke video using FFmpeg."""

import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _parse_time(time_str: str) -> float:
    """Parse FFmpeg time string 'HH:MM:SS.ms' to seconds."""
    parts = time_str.split(":")
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])


def compose(
    video_path: Path,
    instrumental_path: Path,
    subtitles_path: Path,
    output_path: Path,
    duration: float | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> Path:
    """
    Combine original video + instrumental audio + burned-in ASS subtitles.

    Args:
        video_path: Path to the video-only file.
        instrumental_path: Path to the instrumental audio.
        subtitles_path: Path to the ASS subtitle file.
        output_path: Where to write the final video.
        duration: Total video duration in seconds (for progress calculation).
        progress_callback: Optional callback receiving progress as 0.0–1.0.

    The subtitles filter renders the ASS file directly into the video frames.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # FFmpeg subtitles filter requires absolute path with colons escaped
    abs_subs = subtitles_path.resolve()
    subs_escaped = str(abs_subs).replace("\\", "/").replace(":", "\\:")

    cmd = [
        FFMPEG, "-y",
        "-i", str(video_path.resolve()),
        "-i", str(instrumental_path.resolve()),
        "-map", "0:v",          # video from first input
        "-map", "1:a",          # audio from second input (instrumental)
        "-vf", f"ass={subs_escaped}",
        "-c:v", "libx264",
        "-crf", "18",           # high quality
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ]

    if progress_callback is None or duration is None:
        subprocess.run(cmd, check=True)
        return output_path

    # Run with Popen to parse progress from stderr
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    for line in proc.stderr:
        # FFmpeg outputs lines like: frame= 1234 ... time=00:01:23.45 ...
        m = re.search(r"time=(\d+:\d+:\d+\.\d+)", line)
        if m and duration > 0:
            current = _parse_time(m.group(1))
            progress_callback(min(current / duration, 1.0))
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd[0])

    return output_path
