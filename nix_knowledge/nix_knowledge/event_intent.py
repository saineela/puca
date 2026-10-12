"""Third hybrid layer: event/alert intent detection.

The Qwen selector proposes a function, but a short reminder request can be
misread as a durable fact (for example, "I have to take medicines every 2
days from now remind me"). This gate identifies explicit alert/reminder intent
and extracts only the raw title/time span. Temporal meaning remains owned by
TemporalResolver and the calendar hybrid validator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EventIntent:
    requires_event: bool
    title: str = ""
    temporal_expression: str = ""
    kind: str = "event"
    confidence: float = 0.0
    reason: str = ""


_ADDRESS_RE = re.compile(
    r"^(?:hey|hi|hello)\s+(?:casper|nix)\s*[,!:]?\s*",
    re.IGNORECASE,
)
_ALERT_RE = re.compile(
    r"\b(?:remind\s+me|reminder|alert\s+me|notify\s+me|"
    r"let\s+me\s+know|wake\s+me)\b",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"\b(?:every\s+\d+\s+days?(?:\s+at\s+[^,.;!?]+)?|"
    r"every\s+(?:day|daily|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday)(?:\s+at\s+[^,.;!?]+)?|"
    r"(?:(?:today|tomorrow|tmr|tonight|this\s+weekend|next\s+week)\b"
    r"(?:\s+(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?)?|"
    r"in\s+\d+\s+(?:minutes?|hours?|days?|weeks?)|"
    r"(?:on\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?:\s+at\s+[^,.;!?]+)?|"
    r"at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)|"
    # Absolute calendar dates, spoken with or without "at" before the
    # clock ("October 28th 6:30pm", "Oct 28 at 6:30 p.m."). Measured live:
    # reminder requests phrased with a real date reached no function at all
    # because no branch here matched a month-day expression.
    r"(?:on\s+)?(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?"
    r"(?:\s*,?\s*\d{4})?"
    r"(?:\s+(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?)?))\b",
    re.IGNORECASE,
)


def _clean_title(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" ,:;.!?")
    value = re.sub(r"\b(?:from\s+now|now)\b", "", value, flags=re.IGNORECASE)
    # Speech framing around the alert noun ("that's a reminder for ...",
    # "it's a reminder about ...") is not part of the subject.
    value = re.sub(
        r"^(?:that|it|this)(?:'s|\s+is)\s+(?:a|an)?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:like|to)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" ,:;.!?")
    return value


def detect_event_intent(text: str) -> EventIntent:
    """Detect explicit alert/reminder intent without resolving time.

    This is deliberately conservative: it requires an alert verb and a
    temporal expression. Statements such as "I have medicine every day"
    remain facts unless the user asks Casper to remind/alert/notify them.
    """
    original = " ".join((text or "").split())
    normalized = _ADDRESS_RE.sub("", original).strip()
    alert = _ALERT_RE.search(normalized)
    temporal = _TIME_RE.search(normalized)
    if not alert or not temporal:
        return EventIntent(False)

    # Keep the exact temporal phrase, then clean command words from the
    # remaining content. The temporal hybrid layer validates this expression.
    expression = temporal.group(0).strip(" ,:;.!?")
    before = normalized[: temporal.start()].strip()
    after = normalized[temporal.end() :].strip()
    # Drop prepositions that only connect the alert verb to the time or the
    # time to the subject ("remind me for October 28th ... for zoo ticket
    # booking", "reminder about October 28th"): they are not title words.
    before = re.sub(r"\s*(?:for|about|to|on|at)$", "", before, flags=re.IGNORECASE).strip()
    # Punctuation-tolerant: the temporal span can end mid-sentence
    # ("...6:30 p.m. about zoo ticket booking") and leave a stray dot.
    after = re.sub(r"^[\s,.;:!?]*(?:for|about|to|on)\s+", "", after, flags=re.IGNORECASE).strip()
    title = (before + " " + after).strip()
    title = re.sub(
        r"\b(?:remind\s+me(?:\s+(?:to|about|for))?|alert\s+me|notify\s+me|wake\s+me|"
        r"let\s+me\s+know|reminder|from\s+now)\b",
        " ",
        title,
        flags=re.IGNORECASE,
    )
    title = _clean_title(title)
    if not title:
        return EventIntent(False)

    return EventIntent(
        requires_event=True,
        title=title,
        temporal_expression=expression,
        kind="alert" if re.search(r"\b(?:alert|notify|wake)\b", normalized, re.I) else "reminder",
        confidence=0.98,
        reason="explicit_alert_verb_plus_temporal_expression",
    )
