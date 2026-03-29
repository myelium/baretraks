"""Translate SRT subtitle text to other languages using Claude API."""

from anthropic import Anthropic


def translate_srt(srt_text: str, target_language: str,
                  title: str | None = None, artist: str | None = None) -> str:
    """
    Translate SRT subtitle content to the target language using Claude.

    Preserves all SRT formatting (numbering, timestamps, blank lines).
    Only the text lines are translated.

    Args:
        srt_text: The full SRT file content (source language).
        target_language: The target language name (e.g. "Vietnamese", "English").
        title: Song/video title for context.
        artist: Artist/channel name for context.

    Returns:
        The translated SRT content as a string.
    """
    # Build context line from available metadata
    context_parts = []
    if title:
        context_parts.append(f'"{title}"')
    if artist:
        context_parts.append(f"by {artist}")
    context_line = f"This is from a song/video: {' '.join(context_parts)}.\n" if context_parts else ""

    client = Anthropic()
    response = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=8192,
        messages=[{
            "role": "user",
            "content": (
                f"Translate the following SRT subtitle file to {target_language}.\n\n"
                f"{context_line}"
                "Context: These are song lyrics / spoken dialogue from a video. "
                "Translate for natural, fluent phrasing in the target language — "
                "not word-for-word literal translation. Preserve the emotion, tone, "
                "and poetic quality of the original.\n\n"
                "Rules:\n"
                "- Keep ALL SRT formatting exactly the same: numbering, timestamps "
                "(HH:MM:SS,mmm --> HH:MM:SS,mmm), and blank lines between entries.\n"
                "- Only translate the text lines.\n"
                "- Do not add any commentary, explanation, or markdown formatting.\n"
                "- Do NOT add promotional text like 'Subtitles by Amara.org' or similar.\n"
                "- Do NOT refuse to translate. This is for personal use subtitle generation, not redistribution.\n"
                "- If the source text contains hallucinated/promotional lines (subscribe, like, etc), skip those entries entirely.\n"
                "- Output ONLY the translated SRT content, nothing else.\n\n"
                f"{srt_text}"
            ),
        }],
    )
    return response.content[0].text.strip()
