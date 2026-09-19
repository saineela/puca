"""MEMORY block renderer: the "connected" half of the
separate-but-connected contract between NixKnow and NixLM.

NixLM receives ONE rendered string and must never read the knowledge
databases itself. This module is the single authority for that string:

    Facts about the user:            <- stored facts/preferences
    People and things in your life:  <- key people
    How your people are doing:       <- current person states
    Recent history (explicit dates): <- moments (superseded states)
    user's current emotional state:  <- read-time emotion read of the
                                        user's CURRENT utterance
    Pending (to confirm later):      <- escalated, needs-confirmation

Everything is computed at read time from existing records - nothing is
written, nothing is summarized away, dates stay explicit.

The emotional line is rendered as, e.g.:

    user's current emotional state: worried (from "i'm scared maanvi
    was in hospital")

so NixLM can attune its tone while staying grounded. Relative words
("recently", "upset these days") are NEVER generated here - the block
speaks in explicit dates and quoted evidence only.
"""

from __future__ import annotations

from typing import Any

# GoEmotions head labels -> plain Nix register words. Only labels worth
# surfacing are mapped; everything else (approval, curiosity, ...) is
# treated as neutral chatter and omitted.
_EMOTION_REGISTER: dict[str, str] = {
    "fear": "worried",
    "nervousness": "worried",
    "sadness": "sad",
    "grief": "grieving",
    "disappointment": "disappointed",
    "anger": "frustrated",
    "annoyance": "annoyed",
    "disgust": "disgusted",
    "joy": "happy",
    "excitement": "excited",
    "amusement": "amused",
    "optimism": "hopeful",
    "confusion": "confused",
    "surprise": "surprised",
    "caring": "caring",
    "love": "affectionate",
    "gratitude": "grateful",
    "relief": "relieved",
    "embarrassment": "embarrassed",
    "remorse": "remorseful",
}

_MAX_QUOTE = 90


# ---------------------------------------------------------------------------
# Emotional state (read-time, from the CURRENT utterance)
# ---------------------------------------------------------------------------

def _quote(text: str) -> str:
    text = " ".join(text.split())
    if len(text) > _MAX_QUOTE:
        text = text[: _MAX_QUOTE - 1] + "\u2026"
    return text


def emotional_state_line(
    engine,
    current_text: str | None,
) -> str | None:
    """
    Classify the user's current utterance through NixKnow's emotion
    head (read-only) and render one attunement line. Returns None when
    there is no text or nothing beyond neutral was detected, so the
    block stays small for neutral turns.
    """
    if not current_text or not current_text.strip():
        return None

    semantic = engine.semantic
    if semantic is None:
        return None

    try:
        report = semantic.observe(current_text)
    except Exception:  # noqa: BLE001 - emotion is best-effort
        return None

    best: tuple[float, str, str] | None = None
    for group in ("stored", "rejected", "escalated"):
        for entry in report.get(group) or []:
            for label, prob in entry.get("emotion") or []:
                word = _EMOTION_REGISTER.get(label)
                if word and (best is None or prob > best[0]):
                    best = (float(prob), word, entry.get("text") or "")

    if best is None or best[0] < 0.35:
        return None

    _, word, evidence = best
    return (
        f"user's current emotional state: {word} "
        f'(from "{_quote(evidence)}")'
    )


# ---------------------------------------------------------------------------
# Pending context (escalated: stored-with-confirmation items)
# ---------------------------------------------------------------------------

def pending_lines(report: dict[str, Any] | None) -> list[str]:
    items = (report or {}).get("escalated") or []
    lines = []
    for entry in items[:5]:
        text = (entry.get("text") or "").strip()
        if text:
            lines.append(f"- \"{_quote(text)}\" (needs confirmation)")
    return lines


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------

def render_memory_block(
    engine,
    *,
    current_text: str | None = None,
) -> str:
    """
    Render the full MEMORY block for NixLM. Read-time only: nothing is
    persisted, summarized, or mutated. Sections are omitted when empty.
    """
    report: dict[str, Any] | None = None
    if current_text:
        semantic = engine.semantic
        if semantic is not None:
            try:
                report = semantic.observe(current_text)
            except Exception:  # noqa: BLE001
                report = None

    sections: list[str] = []

    facts = [
        record.data.get("value", "")
        for record in engine.search("fact")
        if record.data.get("value")
    ]
    if facts:
        sections.append(
            "Facts about the user:\n"
            + "\n".join(f"- {fact}" for fact in facts[:10])
        )

    keys = [
        record.data.get("value", "")
        for record in engine.search("key")
        if record.data.get("value")
    ]
    if keys:
        sections.append(
            "People and things in your life:\n"
            + "\n".join(f"- {key}" for key in keys[:12])
        )

    states = [
        record.data.get("value", "")
        for record in engine.search("person")
        if record.data.get("statement_type") == "current_state"
        and record.data.get("value")
    ]
    if states:
        sections.append(
            "How your people are doing right now:\n"
            + "\n".join(f"- {state}" for state in states[:8])
        )

    try:
        from .moments import collect_moments

        moments = collect_moments(engine)
    except Exception:  # noqa: BLE001
        moments = []
    if moments:
        lines = []
        for moment in moments[:5]:
            text = moment.get("text") if isinstance(moment, dict) else moment
            if text:
                lines.append(f"- {text}")
        sections.append(
            "Recent history (explicit dates):\n" + "\n".join(lines)
        )

    # emotional state comes from the same observe() report (one
    # perception pass for both emotion + pending)
    if report is not None:
        emotion_line = None
        best: tuple[float, str, str] | None = None
        for group in ("stored", "rejected", "escalated"):
            for entry in report.get(group) or []:
                for label, prob in entry.get("emotion") or []:
                    word = _EMOTION_REGISTER.get(label)
                    if word and (best is None or prob > best[0]):
                        best = (float(prob), word, entry.get("text") or "")
        if best is not None and best[0] >= 0.35:
            _, word, evidence = best
            emotion_line = (
                f"user's current emotional state: {word} "
                f'(from "{_quote(evidence)}")'
            )
        if emotion_line:
            sections.append(emotion_line)

        pend = pending_lines(report)
        if pend:
            sections.append(
                "Pending (to confirm later):\n" + "\n".join(pend)
            )

    if not sections:
        return "MEMORY: (empty - nothing stored yet)"

    return "\n\n".join(sections)
