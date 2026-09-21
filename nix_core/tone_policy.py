"""Conversation policy derived from user tone and current emotional state.

This is a guardrail around the chat model, not a claim that emotion
classification is infallible. High-confidence lexical cues take priority over
a noisy model label, and ambiguous input remains neutral.
"""
from __future__ import annotations

import re
from typing import Any


_TIRED_RE = re.compile(
    r"\b(?:tired|exhausted|drained|worn out|sleepy|long day|can't keep my eyes open)\b",
    re.I,
)
_UNWELL_RE = re.compile(
    r"\b(?:sick|unwell|ill|fever|hurt|in pain|nauseous|migraine|not feeling well)\b",
    re.I,
)
_POSITIVE_RE = re.compile(
    r"\b(?:happy|excited|thrilled|great news|good news|proud|relieved|"
    r"celebrat(?:e|ing)|wonderful|amazing|glad|joyful)\b|!{2,}",
    re.I,
)
_HIGH_DISTRESS_RE = re.compile(
    r"\b(?:hopeless|can't cope|cannot cope|panic attack|want to die|"
    r"hurt myself|end my life)\b",
    re.I,
)


def conversation_policy(
    text: str,
    intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return grounded interaction guidance for one user turn."""
    text = " ".join((text or "").split())
    intent = intent or {}
    label = str(intent.get("emotion") or "").lower()
    valence = str(intent.get("valence") or "").lower()

    if _HIGH_DISTRESS_RE.search(text):
        state, priority = "high_distress", "urgent"
    elif _TIRED_RE.search(text):
        state, priority = "tired", "high"
    elif _UNWELL_RE.search(text) or label in {"sadness", "grief", "fear", "anxiety", "nervousness"} or valence == "bad":
        state, priority = "unwell_or_distressed", "high"
    elif _POSITIVE_RE.search(text) or label in {"joy", "excitement", "gratitude", "relief", "optimism", "love"} or valence == "good":
        state, priority = "positive", "normal"
    else:
        state, priority = "neutral", "normal"

    if state in {"tired", "unwell_or_distressed", "high_distress"}:
        follow_up = (
            "Do not ask nonessential follow-up questions. Answer directly and "
            "briefly. Be gentle; do not make the user manage a long conversation. "
            "Ask one question only if it is necessary for safety or to complete "
            "the requested task."
        )
    elif state == "positive":
        follow_up = (
            "Match the user's positive energy. Celebrate the good news briefly. "
            "A relevant question about someone important in the user's life is "
            "welcome when it naturally follows the topic, but never invent a "
            "person, event, or reason to ask."
        )
    else:
        follow_up = (
            "Be conversational but concise. Ask a follow-up only when it helps "
            "answer the user's request or resolve genuine ambiguity."
        )

    return {
        "state": state,
        "priority": priority,
        "avoid_nonessential_questions": state in {"tired", "unwell_or_distressed", "high_distress"},
        "allow_relevant_positive_follow_up": state == "positive",
        "instruction": follow_up,
    }
