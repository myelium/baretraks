"""Generate ASS subtitle file with karaoke-style word highlighting."""

import math
from pathlib import Path

from .transcribe import Segment, Word

# ASS header with karaoke styling
ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial,60,&H0000FFFF,&H00808080,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,60,60,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

# Max words per subtitle line
WORDS_PER_LINE = 6


def _ass_time(seconds: float) -> str:
    """Convert seconds to ASS timestamp H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _centiseconds(seconds: float) -> int:
    return max(1, int(round(seconds * 100)))


def _build_line(words: list[Word], line_start: float, line_end: float) -> str:
    """
    Build one ASS Dialogue line.

    Uses segment-level line_start/line_end for reliable display timing.
    Word-level timestamps are used only to derive relative proportions for
    the \\k highlight durations within the line.
    """
    line_duration = line_end - line_start

    # Compute each word's relative share of the line duration using its
    # word-level timestamps. If alignment produced suspicious timestamps
    # (all words collapsed to same time), fall back to equal distribution.
    word_spans = []
    for i, w in enumerate(words):
        if i < len(words) - 1:
            span = max(0.0, words[i + 1].start - w.start)
        else:
            span = max(0.0, w.end - w.start)
        word_spans.append(span)

    total_span = sum(word_spans)

    if total_span < 0.01:
        # Alignment collapsed — distribute evenly
        k_durations = [line_duration / len(words)] * len(words)
    else:
        k_durations = [line_duration * (s / total_span) for s in word_spans]

    text_parts = [
        f"{{\\k{_centiseconds(d)}}}{w.text} "
        for w, d in zip(words, k_durations)
    ]
    text = "".join(text_parts).rstrip()

    return (
        f"Dialogue: 0,{_ass_time(line_start)},{_ass_time(line_end)},"
        f"Karaoke,,0,0,0,,{text}"
    )


def build_ass(segments: list[Segment], output_path: Path) -> Path:
    """
    Build an ASS subtitle file from Whisper segments.

    Each segment uses its reliable Whisper-level start/end for display timing.
    Long segments (many words) are split into sub-lines, with the segment
    duration divided proportionally.
    """
    lines = []

    for seg in segments:
        words = seg.words
        n_lines = math.ceil(len(words) / WORDS_PER_LINE)
        seg_duration = seg.end - seg.start

        for i in range(n_lines):
            chunk = words[i * WORDS_PER_LINE : (i + 1) * WORDS_PER_LINE]
            # Divide segment time proportionally by word count
            line_start = seg.start + (i / n_lines) * seg_duration
            line_end = seg.start + ((i + 1) / n_lines) * seg_duration
            lines.append(_build_line(chunk, line_start, line_end))

    output_path.write_text(ASS_HEADER + "\n".join(lines) + "\n")
    return output_path
