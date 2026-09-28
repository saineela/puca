"""Presentation cleanup and identity guards shared by Luna runtime/builders."""
from __future__ import annotations

import re
from typing import Any, Iterable

_SPACE_RUN = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"[ \t]+([,.;:!?])")
_FENCE = re.compile(r"^\s{0,3}(?:```|~~~)")
_CASPER_REFERENCE = re.compile(r"\bcasper\b", re.IGNORECASE)


def clean_response_text(text: object) -> str:
    """Trim trailing blanks and collapse accidental spacing in prose.

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


def luna_identity_control_text(text: str) -> bool:
    """True when text teaches Luna a Casper cross-identity relationship."""
    return bool(_CASPER_REFERENCE.search(str(text or "")))


def luna_identity_control_allowed(messages: Iterable[dict[str, Any]]) -> bool:
    """Reject legacy cross-identity examples from new Luna training data."""
    return not any(
        luna_identity_control_text(str(message.get("content") or ""))
        for message in messages
    )


def validate_luna_training_rows(rows: Iterable[dict[str, Any]]) -> int:
    """Fail closed if new SFT data teaches cross-person identity text."""
    rejected = sum(
        not luna_identity_control_allowed(row.get("messages", []))
        for row in rows
    )
    if rejected:
        raise ValueError(
            f"Luna SFT contains {rejected} row(s) mentioning a different model/person identity"
        )
    return rejected
