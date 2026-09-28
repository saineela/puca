"""Independent Core routing engine.

This module is intentionally separate from Brain and all model clients.  It
makes a bounded routing decision in-process and returns observability data so
model latency cannot be mislabeled as routing latency.

The engine is multi-label at the boundary: a request may be chat, Knowledge,
or a compound of both.  It never calls Qwen, Casper, a database, or an HTTP
service.  Ambiguous text is returned as ``unknown`` for the caller to handle;
that fallback is explicit rather than hidden inside routing.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, asdict
from typing import Any

from router import CHAT, KNOWLEDGE, UNKNOWN, classify


@dataclass(frozen=True)
class RouteDecision:
    route: str
    confidence: float
    reason: str
    rule: str
    requires_knowledge: bool
    requires_model: bool
    safety: str = "normal"
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# High-precision response-act shortcuts.  These prevent trivial social turns
# from entering the general classifier/model pipeline.
_SOCIAL_RE = re.compile(
    r"^(?:hi|hey|hello|yo)\b.*\b(?:are\s+you\s+alive|you\s+there)\b|"
    r"^(?:hi|hey|hello|yo)[.!?,\s]*$",
    re.I,
)
_EMOTION_RE = re.compile(
    r"\b(?:i(?:'m| am)\s+(?:tired|exhausted|overwhelmed|sad|happy|angry|"
    r"upset|stressed|worried)|i\s+feel\s+)\b",
    re.I,
)
_ACTION_RE = re.compile(
    r"\b(?:remind|reminder|schedule|calendar|appointment|alarm|"
    r"reschedule|cancel|delete\s+(?:my|the)\s+(?:reminder|event)|"
    r"remember|save|store|keep\s+track|make\s+(?:a\s+)?note)\b",
    re.I,
)
_PERSONAL_RE = re.compile(
    r"\b(?:my|our|i\s+(?:like|love|hate|enjoy|prefer)|"
    r"who\s+am\s+i|how\s+(?:is|are)\s+my)\b",
    re.I,
)
_WORLD_CHAT_RE = re.compile(
    r"\b(?:joke|story|poem|weather|news|capital|explain|define|"
    r"how\s+do\s+i|what\s+is|who\s+was|translate|recommend)\b",
    re.I,
)
_INJECTION_RE = re.compile(
    r"\b(?:ignore|disregard)\s+(?:all\s+)?(?:previous|prior)\s+"
    r"(?:instructions?|rules?|prompts?)\b|\bsystem\s*:",
    re.I,
)


class CoreRoutingEngine:
    """Fast, deterministic route decision boundary.

    ``legacy_classifier`` is injected only as a compatibility fallback for
    the repository's large symbolic corpus. It is never a neural model. The
    engine's high-confidence paths return before it is consulted.
    """

    name = "core-routing-engine-v1"

    def __init__(self, legacy_classifier=classify) -> None:
        self._legacy_classifier = legacy_classifier

    def decide(self, text: str, *, allow_ambiguous: bool = True) -> RouteDecision:
        started = time.perf_counter()
        clean = " ".join((text or "").split())
        if not clean:
            return self._decision(UNKNOWN, 0.0, "empty_input", "empty", False, True, started)

        if _INJECTION_RE.search(clean):
            return self._decision(CHAT, 0.99, "prompt_safety_boundary", "injection_guard", False, False, started, "guarded")

        # Social and emotion turns are ordinary chat, never Knowledge/model
        # routing. They may still be handled by a deterministic response act.
        if _SOCIAL_RE.match(clean):
            return self._decision(CHAT, 1.0, "social_response_act", "social_fast_path", False, False, started)
        if _EMOTION_RE.search(clean) and not _ACTION_RE.search(clean):
            return self._decision(CHAT, 0.96, "emotional_response_act", "emotion_fast_path", False, False, started)

        # Strong personal/action signals outrank generic world-question words.
        # This preserves compound personal requests for Knowledge validation.
        if _ACTION_RE.search(clean) or _PERSONAL_RE.search(clean):
            route, features = self._legacy_classifier(clean)
            if route == CHAT and _WORLD_CHAT_RE.search(clean) and not _ACTION_RE.search(clean):
                return self._decision(CHAT, 0.90, "personal_word_is_world_question", features.get("rule") or "world_chat", False, False, started)
            if route == UNKNOWN:
                route = KNOWLEDGE if _ACTION_RE.search(clean) else UNKNOWN
            return self._decision(route, 0.93 if route == KNOWLEDGE else 0.78, "personal_or_action_signal", features.get("rule") or "personal_boundary", route == KNOWLEDGE, route == UNKNOWN, started)

        # Clear general chat avoids any Knowledge model gate.
        if _WORLD_CHAT_RE.search(clean):
            return self._decision(CHAT, 0.90, "world_chat_signal", "world_chat_fast_path", False, False, started)

        if allow_ambiguous:
            route, features = self._legacy_classifier(clean)
            if route != UNKNOWN:
                return self._decision(route, 0.86, "symbolic_classifier", features.get("rule") or "symbolic_route", route == KNOWLEDGE, False, started)

        return self._decision(UNKNOWN, 0.0, "insufficient_route_evidence", "abstain", False, True, started)

    @staticmethod
    def _decision(route: str, confidence: float, reason: str, rule: str,
                  requires_knowledge: bool, requires_model: bool,
                  started: float, safety: str = "normal") -> RouteDecision:
        return RouteDecision(
            route=route,
            confidence=round(confidence, 3),
            reason=reason,
            rule=rule,
            requires_knowledge=requires_knowledge,
            requires_model=requires_model,
            safety=safety,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )


def route_request(text: str) -> RouteDecision:
    """Convenience API used by tests and lightweight integrations."""
    return CoreRoutingEngine().decide(text)
