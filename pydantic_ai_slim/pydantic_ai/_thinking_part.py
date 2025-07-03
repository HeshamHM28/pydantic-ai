from __future__ import annotations as _annotations

from pydantic_ai.messages import TextPart, ThinkingPart

START_THINK_TAG = '<think>'
END_THINK_TAG = '</think>'


def split_content_into_text_and_thinking(content: str) -> list[ThinkingPart | TextPart]:
    """Split a string into text and thinking parts.

    Some models don't return the thinking part as a separate part, but rather as a tag in the content.
    This function splits the content into text and thinking parts.

    We use the `<think>` tag because that's how Groq uses it in the `raw` format, so instead of using `<Thinking>` or
    something else, we just match the tag to make it easier for other models that don't support the `ThinkingPart`.
    """
    parts: list[ThinkingPart | TextPart] = []
    start_len = len(START_THINK_TAG)
    end_len = len(END_THINK_TAG)
    i = 0
    n = len(content)
    while i < n:
        sidx = content.find(START_THINK_TAG, i)
        if sidx == -1:
            if i < n:
                parts.append(TextPart(content=content[i:]))
            break
        if sidx > i:
            parts.append(TextPart(content=content[i:sidx]))
        t_start = sidx + start_len
        eidx = content.find(END_THINK_TAG, t_start)
        if eidx == -1:
            # No closing tag: treat the rest as text (discard opening tag, per original reasoning)
            parts.append(TextPart(content=content[t_start:]))
            break
        parts.append(ThinkingPart(content=content[t_start:eidx]))
        i = eidx + end_len
    return parts
