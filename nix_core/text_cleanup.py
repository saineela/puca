"""Small deterministic whitespace cleanup for generated chat text."""
from __future__ import annotations

import re

_SPACE_RUN = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"[ \t]+([,.;:!?])")
_FENCE = re.compile(r"^\s{0,3}(?:```|~~~)")


def clean_response_text(text: object) -> str:
    """Trim trailing blanks and collapse accidental spaces in prose.

    Line breaks and fenced code blocks are preserved. This is deliberately a
    presentation-only cleanup: it does not alter vocabulary or sentence style.
    """
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    value = value.translate({0x00A0: " ", 0x2007: " ", 0x202F: " "})

    cleaned: list[str] = []
    in_fence = False
    for line in value.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            cleaned.append(line.rstrip())
            continue
        if in_fence:
            cleaned.append(line)
            continue

        leading_length = len(line) - len(line.lstrip(" \t"))
        leading, body = line[:leading_length], line[leading_length:]
        body = _SPACE_RUN.sub(" ", body)
        body = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", body)
        cleaned.append(leading + body.rstrip())

    return "\n".join(cleaned).strip()
