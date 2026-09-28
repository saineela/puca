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
    r"(?:today|tomorrow|tmr|tonight|this\s+weekend|next\s+week|"
    r"in\s+\d+\s+(?:minutes?|hours?|days?|weeks?)|"
    r"(?:on\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?:\s+at\s+[^,.;!?]+)?|"
    r"at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)))\b",
    re.IGNORECASE,
)


def _clean_title(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" ,:;.!?")
    value = re.sub(r"\b(?:from\s+now|now)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" ,:;.!?")
    value = re.sub(r"^(?:like|to)\s+", "", value, flags=re.IGNORECASE)
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
    before = normalized[: temporal.start()]
    after = normalized[temporal.end() :]
    title = before + " " + after
    title = re.sub(
        r"\b(?:remind\s+me|alert\s+me|notify\s+me|wake\s+me|"
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
