"""Translate SRT subtitle text to other languages using Claude API."""

from anthropic import Anthropic


def translate_srt(srt_text: str, target_language: str) -> str:
    """
    Translate SRT subtitle content to the target language using Claude.

    Preserves all SRT formatting (numbering, timestamps, blank lines).
    Only the text lines are translated.

    Args:
        srt_text: The full SRT file content (in English).
        target_language: The target language name (e.g. "Vietnamese", "Mandarin Chinese").

    Returns:
        The translated SRT content as a string.
    """
    client = Anthropic()  # uses ANTHROPIC_API_KEY env var
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[{
            "role": "user",
            "content": (
                f"Translate the following SRT subtitle file to {target_language}.\n\n"
                "Rules:\n"
                "- Keep ALL SRT formatting exactly the same: numbering, timestamps (HH:MM:SS,mmm --> HH:MM:SS,mmm), and blank lines between entries.\n"
                "- Only translate the text lines.\n"
                "- Do not add any commentary, explanation, or markdown formatting.\n"
                "- Output ONLY the translated SRT content.\n\n"
                f"{srt_text}"
            ),
        }],
    )
    return response.content[0].text.strip()
