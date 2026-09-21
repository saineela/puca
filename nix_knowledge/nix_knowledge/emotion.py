"""Conservative conversational emotion and care signals.

The neural GoEmotions head remains the broad emotion classifier. This small
lexical layer is deliberately used for high-risk, high-cost-to-miss signals:
explicit distress, crisis language, urgency, and caring relationships. It
never claims to know a person's inner state from a single ambiguous turn.
"""
from __future__ import annotations

import re
from typing import Any


_PATTERNS: tuple[tuple[str, str, float], ...] = (
    ("crisis", r"\b(?:want to die|kill myself|end my life|suicid(?:e|al)|hurt myself)\b", 1.0),
    ("grief", r"\b(?:someone died|he died|she died|lost my father|lost my mother|funeral)\b", 0.95),
    ("panic", r"\b(?:panic attack|can't breathe|cannot breathe|having a panic|terrified)\b", 0.9),
    ("distress", r"\b(?:i feel hopeless|i feel worthless|i can't cope|cannot cope|falling apart|breaking down)\b", 0.85),
    ("sadness", r"\b(?:i(?:'m| am) sad|i feel sad|so lonely|heartbroken|devastated|crying)\b", 0.8),
    ("anger", r"\b(?:i(?:'m| am) furious|i(?:'m| am) enraged|so angry|i hate this)\b", 0.75),
    ("anxiety", r"\b(?:i(?:'m| am) anxious|i(?:'m| am) worried|so nervous|stressed out|overwhelmed)\b", 0.75),
    ("joy", r"\b(?:i(?:'m| am) thrilled|so happy|really excited|best news|wonderful news)\b", 0.7),
)

_RELATION_RE = re.compile(
    r"\bmy\s+(?:older\s+|younger\s+)?(?:sister|brother|mom|mother|dad|father|"
    r"partner|wife|husband|son|daughter|friend|grandma|grandpa)\b", re.I
)


def analyze_emotion(text: str) -> dict[str, Any]:
    """Return conservative emotion, intensity, urgency, and care metadata.

    ``certain`` is false for unmatched language: callers should respond with
    empathy and a question rather than asserting an emotion as fact.
    """
    original = " ".join((text or "").split())
    matches: list[tuple[str, float, str]] = []
    for label, pattern, intensity in _PATTERNS:
        match = re.search(pattern, original, re.I)
        if match:
            matches.append((label, intensity, match.group(0)))

    if not matches:
        return {
            "emotion": "unknown",
            "valence": "unknown",
            "intensity": 0.0,
            "certain": False,
            "urgent": False,
            "crisis": False,
            "care_subject": None,
            "evidence": None,
        }

    label, intensity, evidence = max(matches, key=lambda item: item[1])
    negative = {"crisis", "grief", "panic", "distress", "sadness", "anger", "anxiety"}
    care_subject = _RELATION_RE.search(original)
    return {
        "emotion": label,
        "valence": "negative" if label in negative else "positive",
        "intensity": intensity,
        "certain": intensity >= 0.75,
        "urgent": label in {"crisis", "panic", "distress"},
        "crisis": label == "crisis",
        "care_subject": care_subject.group(0) if care_subject else None,
        "evidence": evidence,
    }
