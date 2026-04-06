"""Correct Whisper-transcribed lyrics using Claude's knowledge of song lyrics."""

import json

from anthropic import Anthropic


_NON_LYRICS_PHRASES = [
    "subscribe", "like and", "hit the bell", "notification",
    "comment below", "share this", "check out", "follow me",
    "follow us", "click here", "link in", "description below",
    "don't forget", "make sure", "turn on", "smash the",
    "leave a like", "drop a like", "social media",
    "thank you for watching", "thanks for watching",
    "thanks for listening", "thank you for listening",
]


def _strip_non_lyrics(words: list[dict]) -> list[dict]:
    """Remove words that are clearly YouTube promo/outro content, not song lyrics.
    Scans for clusters of non-lyrics words and removes entire clusters."""
    if not words:
        return words

    # Build full text to find promo phrases
    full_text = " ".join(w["text"] for w in words).lower()
    skip_ranges = []

    for phrase in _NON_LYRICS_PHRASES:
        idx = 0
        while True:
            pos = full_text.find(phrase, idx)
            if pos == -1:
                break
            skip_ranges.append((pos, pos + len(phrase)))
            idx = pos + 1

    if not skip_ranges:
        return words

    # Map character positions back to word indices
    skip_indices = set()
    char_pos = 0
    for i, w in enumerate(words):
        word_start = char_pos
        word_end = char_pos + len(w["text"])
        for rs, re_ in skip_ranges:
            if word_start < re_ and word_end > rs:
                skip_indices.add(i)
                break
        char_pos = word_end + 1  # +1 for the space

    return [w for i, w in enumerate(words) if i not in skip_indices]


def correct_lyrics(words: list[dict], title: str | None = None,
                   artist: str | None = None,
                   user_artist: str | None = None) -> dict:
    """
    Use Claude to correct misheard words in Whisper transcription.

    Each word dict has: {"text": str, "start": float, "end": float}

    Claude compares the transcription against its knowledge of the song's
    official lyrics and corrects words that were likely misheard, while
    preserving the exact same number of entries and all timestamps.

    Args:
        title: YouTube video title
        artist: YouTube channel name
        user_artist: Artist name provided by the user in the queue form (fallback)

    Returns dict with:
        "words": corrected word list (same structure, same length)
        "identified_title": song title Claude identified (or None)
        "identified_artist": artist Claude identified (or None)
    """
    if not words:
        return {"words": words, "identified_title": None, "identified_artist": None}

    # Build the plain text transcription for Claude to review
    texts = [w["text"] for w in words]
    transcript = " ".join(texts)

    # Build context — YouTube title often contains song name + artist
    context_parts = []
    if title:
        context_parts.append(f'YouTube video title: "{title}"')
    if user_artist:
        context_parts.append(f"Artist: {user_artist}")
    context_line = "\n".join(context_parts) + "\n" if context_parts else ""

    prompt = (
        f"You are a song lyrics transcription corrector.\n\n"
        f"STEP 1 — Identify the song:\n"
        f"{context_line}"
        f"Using the metadata above, identify the song and artist. "
        f"The YouTube video title usually contains the song name and artist. "
        f"If an Artist is explicitly provided, trust that. "
        f"Then recall the actual official lyrics for this song from your knowledge.\n\n"
        f"STEP 2 — Correct the transcription:\n"
        f"The following lyrics were transcribed from the audio using speech recognition. "
        f"Some words may be incorrect due to mishearing. Compare each word against "
        f"the actual lyrics you recalled in Step 1 and correct any errors.\n\n"
        f"Transcription ({len(texts)} words):\n{transcript}\n\n"
        f"Your response must have exactly two sections.\n\n"
        f"SECTION 1 — First line only:\n"
        f"SONG: {{song title}} | ARTIST: {{artist/singer name}} | LANG: {{ISO 639-1 language code of the lyrics, e.g. en, ur, hi, es}}\n\n"
        f"SECTION 2 — Corrected words:\n"
        f"EXACTLY {len(texts)} lines, one word per line, in the same order as the transcription.\n\n"
        f"Rules:\n"
        f"- The first line must be the SONG/ARTIST identification.\n"
        f"- Then a blank line.\n"
        f"- Then EXACTLY {len(texts)} word lines.\n"
        f"- Only output words that are actually sung. No additions, no creative interpretation.\n"
        f"- If you don't know the song or aren't sure about a correction, "
        f"return the original word unchanged.\n"
        f"- Preserve contractions as-is (don't → don't, not do not).\n"
        f"- Keep the same word boundaries — if Whisper heard one word, output one word.\n"
        f"- If a word is clearly NOT part of the song (e.g. YouTube promos, "
        f"'subscribe', 'like and share', spoken intros/outros, 'thank you' at the end, "
        f"credits, applause), replace it with the marker __SKIP__. "
        f"This still counts toward the {len(texts)} total.\n"
        f"- No commentary, no numbering beyond the format above.\n"
    )

    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    import logging
    _cl_logger = logging.getLogger(__name__)

    raw = response.content[0].text.strip()
    lines = raw.split("\n")
    _cl_logger.info("Claude response first line: %s", lines[0] if lines else "(empty)")

    # Parse SONG/ARTIST/LANG from first line
    identified_title = None
    identified_artist = None
    identified_language = None
    word_lines_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("SONG:"):
            # Parse "SONG: xxx | ARTIST: yyy | LANG: zz"
            parts = stripped.split("|")
            for part in parts:
                part = part.strip()
                if part.startswith("SONG:"):
                    identified_title = part[5:].strip()
                elif part.startswith("ARTIST:"):
                    identified_artist = part[7:].strip()
                elif part.startswith("LANG:"):
                    identified_language = part[5:].strip().lower()
            word_lines_start = i + 1
            break

    _cl_logger.info("Parsed: title=%s, artist=%s, lang=%s",
                    identified_title, identified_artist, identified_language)

    # Skip blank lines after the SONG line
    while word_lines_start < len(lines) and not lines[word_lines_start].strip():
        word_lines_start += 1

    corrected_lines = [l.strip() for l in lines[word_lines_start:] if l.strip()]

    # Apply corrections if word count matches, otherwise keep original
    if len(corrected_lines) == len(texts):
        result = []
        for i, w in enumerate(words):
            if corrected_lines[i] == "__SKIP__":
                continue
            result.append({
                "text": corrected_lines[i],
                "start": w["start"],
                "end": w["end"],
            })
    else:
        result = list(words)

    # Always strip non-lyrics regardless of whether correction succeeded
    result = _strip_non_lyrics(result)

    return {
        "words": result,
        "identified_title": identified_title,
        "identified_artist": identified_artist,
        "identified_language": identified_language,
    }
