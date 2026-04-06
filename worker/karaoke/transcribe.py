"""Transcribe vocals with word-level timestamps using faster-whisper + WhisperX alignment."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

# Shift all word timestamps earlier (seconds) to compensate for Whisper's
# tendency to place word onsets slightly late.
LYRICS_OFFSET = -0.3           # when WhisperX alignment succeeds
LYRICS_OFFSET_UNALIGNED = -0.8  # when falling back to raw Whisper timestamps


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


# Whisper hallucination phrases — exact segment-level matches
_HALLUCINATION_EXACT = {
    "thank you for watching",
    "thanks for watching",
    "thank you for listening",
    "thanks for listening",
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
    "thank you",
}

# Substrings that only appear in hallucinated URLs/credits
_HALLUCINATION_SUBSTRINGS = ["www.", "http", ".com", ".org"]

# Single words that Whisper hallucinates (never real lyrics on their own)
_HALLUCINATION_WORDS = {
    "subscribe", "unsubscribe", "subscribed",
}


def _is_hallucination(text: str) -> bool:
    """Check if a segment's text is a known Whisper hallucination."""
    lower = text.lower().strip()
    if not lower:
        return True
    # Exact match against known hallucination phrases
    if lower in _HALLUCINATION_EXACT:
        return True
    # Substring match for phrases that may appear as part of longer text
    for phrase in _HALLUCINATION_EXACT:
        if phrase in lower:
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
    """Remove segments and individual words that are Whisper hallucinations."""
    filtered = []
    for seg in segments:
        text = " ".join(w.text for w in seg.words).strip()

        # Skip entire segment if it's a known hallucination
        if _is_hallucination(text):
            continue

        # Filter individual hallucinated words within otherwise valid segments
        clean_words = [w for w in seg.words if w.text.lower().strip() not in _HALLUCINATION_WORDS]
        if clean_words:
            filtered.append(Segment(start=clean_words[0].start, end=clean_words[-1].end, words=clean_words))

    return filtered


def transcribe(audio_path: Path, device: str = "cpu",
               language: str | None = None,
               translate: bool = False,
               **kwargs) -> tuple[list[Segment], str]:
    """
    Transcribe audio with word-level timestamps using Whisper large-v3-turbo.

    Args:
        language: Optional language code (e.g. "en", "fr"). None = auto-detect.
        translate: If True, translate to English.

    Returns:
        Tuple of (segments, detected_language_code).
    """
    return _transcribe_whisper(audio_path, device=device, language=language, translate=translate)


def _transcribe_whisper(audio_path: Path, device: str = "cpu",
                        language: str | None = None,
                        translate: bool = False) -> tuple[list[Segment], str]:
    """Transcribe using faster-whisper large-v3."""
    import logging
    _logger = logging.getLogger(__name__)

    # faster-whisper (CTranslate2) only supports cpu and cuda, not mps
    whisper_device = "cpu" if device == "mps" else device
    model = WhisperModel("large-v3-turbo", device=whisper_device, compute_type="int8")

    # Detect language with a quick pass if not explicitly set
    if not language and not translate:
        _, det_info = model.transcribe(
            str(audio_path), word_timestamps=False,
            condition_on_previous_text=False,
        )
        # Consume the generator to get info
        for _ in _: pass
        if det_info.language_probability >= 0.5:
            language = det_info.language
            _logger.info("Detected language: %s (%.0f%% confidence)",
                         language, det_info.language_probability * 100)
        else:
            language = "en"
            _logger.info("Low confidence language detection (%.0f%% %s), defaulting to English",
                         det_info.language_probability * 100, det_info.language)

    raw_segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        condition_on_previous_text=False,
        language=language if not translate else None,
        task="translate" if translate else "transcribe",
        hallucination_silence_threshold=2.0,
        no_speech_threshold=0.6,
    )

    detected_lang = language or info.language or "en"

    segments: list[Segment] = []
    for seg in raw_segments:
        words = [
            Word(text=w.word.strip(), start=w.start, end=w.end)
            for w in (seg.words or [])
            if w.word.strip()
        ]
        if words:
            segments.append(Segment(start=seg.start, end=seg.end, words=words))

    segments = _filter_hallucinations(segments)

    # Refine word timestamps using WhisperX forced alignment (wav2vec2)
    segments, aligned = _align_words(audio_path, segments, lang=detected_lang, device=device)

    # Apply global offset to shift lyrics slightly earlier.
    # Use a larger offset when alignment wasn't available (raw Whisper timestamps
    # tend to lag more than aligned ones).
    offset = LYRICS_OFFSET if aligned else LYRICS_OFFSET_UNALIGNED
    if offset != 0:
        segments = _apply_offset(segments, offset)

    return segments, detected_lang


def _align_words(audio_path: Path, segments: list[Segment],
                 lang: str = "en", device: str = "cpu") -> tuple[list[Segment], bool]:
    """Use WhisperX wav2vec2 alignment for tighter word-level timestamps.
    Returns (segments, aligned) where aligned is True if alignment succeeded."""
    try:
        import whisperx

        audio = whisperx.load_audio(str(audio_path.resolve()))

        # Convert our Segment/Word dataclasses to the dict format WhisperX expects
        whisperx_segments = []
        for seg in segments:
            text = " ".join(w.text for w in seg.words)
            whisperx_segments.append({
                "start": seg.start, "end": seg.end, "text": text,
            })

        if not whisperx_segments:
            return segments

        align_model, metadata = whisperx.load_align_model(
            language_code=lang, device=device)
        aligned = whisperx.align(
            whisperx_segments, align_model, metadata, audio, device)

        # Convert back to our Segment/Word dataclasses, anchoring to
        # Whisper's original segment start times (WhisperX sometimes
        # shifts segment boundaries later, but Whisper's onsets are better).
        aligned_segs = aligned.get("segments", [])
        result = []
        for i, aseg in enumerate(aligned_segs):
            words = []
            for w in aseg.get("words", []):
                if "start" in w and "end" in w and w.get("word", "").strip():
                    words.append(Word(
                        text=w["word"].strip(),
                        start=w["start"],
                        end=w["end"],
                    ))
            if not words:
                continue

            # Anchor: shift this segment's words so the first word aligns
            # with the original Whisper segment start time
            if i < len(segments):
                whisper_start = segments[i].start
                whisperx_start = words[0].start
                shift = whisper_start - whisperx_start
                if abs(shift) > 0.05:  # only shift if meaningful
                    words = [
                        Word(text=w.text,
                             start=max(0, w.start + shift),
                             end=max(0, w.end + shift))
                        for w in words
                    ]

            result.append(Segment(
                start=words[0].start, end=words[-1].end, words=words))

        if result:
            log.info("WhisperX alignment: %d segments, %d words",
                     len(result), sum(len(s.words) for s in result))
            return result, True

        log.warning("WhisperX alignment produced no results, using Whisper timestamps")
        return segments, False

    except ImportError:
        log.info("whisperx not installed, using Whisper timestamps")
        return segments, False
    except Exception as e:
        log.warning("WhisperX alignment failed (%s), using Whisper timestamps", e)
        return segments, False


def _apply_offset(segments: list[Segment], offset: float) -> list[Segment]:
    """Shift all timestamps by offset seconds. Clamps to >= 0."""
    result = []
    for seg in segments:
        words = []
        for w in seg.words:
            start = max(0, w.start + offset)
            end = max(start, w.end + offset)
            words.append(Word(text=w.text, start=start, end=end))
        if words:
            result.append(Segment(
                start=words[0].start, end=words[-1].end, words=words))
    return result
