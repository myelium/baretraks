"""Transcribe vocals with word-level timestamps using faster-whisper."""

import re
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    start: float
    end: float
    words: list[Word]


# Whisper hallucination phrases — only exact/near-exact matches that would
# never appear in real lyrics. Kept minimal to avoid false positives.
_HALLUCINATION_EXACT = {
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "subscribe to my channel",
    "subscribe to the channel",
    "don't forget to subscribe",
    "hit the bell",
    "notification bell",
    "subtitles by",
    "translated by",
    "captions by",
    "amara.org",
}

# Substrings that only appear in hallucinated URLs/credits
_HALLUCINATION_SUBSTRINGS = ["www.", "http", ".com", ".org"]


def _is_hallucination(text: str) -> bool:
    """Check if a segment's text is a known Whisper hallucination."""
    lower = text.lower().strip()
    if not lower:
        return True
    # Exact match against known hallucination phrases
    if lower in _HALLUCINATION_EXACT:
        return True
    # URL-like substrings
    for pat in _HALLUCINATION_SUBSTRINGS:
        if pat in lower:
            return True
    # Detect highly repetitive text (same word repeated 4+ times)
    words = lower.split()
    if len(words) >= 4 and len(set(words)) == 1:
        return True
    return False


def _filter_hallucinations(segments: list[Segment]) -> list[Segment]:
    """Remove segments that contain known Whisper hallucinations."""
    filtered = []
    for seg in segments:
        text = " ".join(w.text for w in seg.words).strip()

        # Skip known hallucination phrases
        if _is_hallucination(text):
            continue

        filtered.append(seg)

    return filtered


def transcribe(audio_path: Path, device: str = "cpu",
               language: str | None = None,
               translate: bool = False) -> list[Segment]:
    """
    Transcribe audio using faster-whisper (CTranslate2) with word-level timestamps.

    faster-whisper is ~4-6x faster than openai-whisper on CPU with the same accuracy.

    Args:
        language: Optional Whisper language code (e.g. "en", "fr"). None = auto-detect.
        translate: If True, translate the audio to English (task="translate").
    """
    model = WhisperModel("large-v3", device=device, compute_type="int8")
    raw_segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        condition_on_previous_text=False,
        language=language if not translate else None,
        task="translate" if translate else "transcribe",
        hallucination_silence_threshold=3.0,
    )

    segments: list[Segment] = []
    for seg in raw_segments:
        words = [
            Word(text=w.word.strip(), start=w.start, end=w.end)
            for w in (seg.words or [])
            if w.word.strip()
        ]
        if words:
            segments.append(Segment(start=seg.start, end=seg.end, words=words))

    return _filter_hallucinations(segments)
