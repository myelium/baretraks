"""Transcribe vocals with word-level timestamps using faster-whisper."""

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


def transcribe(audio_path: Path, device: str = "cpu") -> list[Segment]:
    """
    Transcribe audio using faster-whisper (CTranslate2) with word-level timestamps.

    faster-whisper is ~4-6x faster than openai-whisper on CPU with the same accuracy.
    """
    model = WhisperModel("large-v3", device=device, compute_type="int8")
    raw_segments, _ = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        condition_on_previous_text=False,
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

    return segments
