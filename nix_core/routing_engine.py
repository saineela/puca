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

import hashlib
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
    # D4 ledger: what the decision cost, what escalation would have cost, and
    # what the controlled replay later found. Defaults keep every existing
    # caller (and its tests) source-compatible.
    cost_estimate: float = 0.0
    escalation_cost: float = 0.0
    counterfactual: str = "unmeasured"
    outcome: str = "unlabelled"
    fingerprint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def ledger_entry(self, *, request: str = "") -> dict[str, Any]:
        """The bounded per-request row: route, cost and a fingerprint only."""
        return {
            "route": self.route,
            "confidence": self.confidence,
            "reason": self.reason,
            "rule": self.rule,
            "requires_knowledge": self.requires_knowledge,
            "requires_model": self.requires_model,
            "safety": self.safety,
            "latency_ms": self.latency_ms,
            "cost_estimate": self.cost_estimate,
            "escalation_cost": self.escalation_cost,
            "counterfactual": self.counterfactual,
            "outcome": self.outcome,
            "fingerprint": self.fingerprint or request_fingerprint(request),
        }


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
    r"\b(?:remind|reminder|schedule|scheduled|calendar|appointment|alarm|"
    r"reschedule|rescheduled|cancel|cancell?ed|called\s+off|"
    r"delete\s+(?:my|the)\s+(?:reminder|event|note|fact|memory|"
    r"address|record)|"
    r"put\s+.{0,40}?\s+in\s+for|\badd\s+(?:my|the|a)\b|"
    r"\brenewal\b|\bdue\b|\bbook\b|\blog\s+that\b|"
    r"remember|save|store|keep\s+track|keep\s+in\s+mind|"
    r"make\s+(?:a\s+)?note|note\s+(?:that|down))\b",
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
# A question about the user's own stored data is a Knowledge read whatever the
# symbolic classifier makes of its wording. Measured in the D4 confirmation
# replay: 15 of 38 decidable turns came back `unknown` for exactly this shape
# ("Which city do I live in?", "Who do I rent from?", "What medication should I
# avoid?"), and the personal questions that did route were split between chat
# and knowledge. Questions that talk about the world ("how do I make bread",
# "tell me a joke") stay chat unless they name something of the user's own.
_SELF_QUESTION_RE = re.compile(
    r"^(?:what|whats|which|when|where|who|whos|whose|how|"
    r"do|does|did|am|is|are|can|could|would|will|should)\b"
    r"[^?]*\b(?:i|me|my|mine|we|our|ours|us)\b",
    re.I,
)
# "Tell me my garage code" is a data read; "tell me one fun fact about
# octopuses" is world chat. The bare pronoun is what separates them: an
# instruction only names the user's own data when it takes a possessive, so the
# end-to-end suite's world-chat turn keeps its chat route.
_SELF_INSTRUCTION_RE = re.compile(
    r"^(?:tell|show|list|give|read|check|pull\s+up)\b"
    r"[^?]*\b(?:my|our|mine|ours)\b",
    re.I,
)
_POSSESSIVE_RE = re.compile(r"\b(?:my|our|mine|ours)\b", re.I)
# A deadline the user stored is personal data even without a possessive:
# "when is the renewal due". The D4 c32 write left exactly this read behind;
# routing it to chat let the model answer "March 4, 2027" for a fact stored
# without a year (measured in the live check). Only a short subject counts, so
# a world question ("when is the tax filing deadline") is untouched.
_DUE_QUESTION_RE = re.compile(
    r"^when\s+(?:is|are)\s+(?:the|my|our)\s+"
    r"[\w'\-]+(?:\s+[\w'\-]+){0,2}\s+due\b",
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
        fingerprint = request_fingerprint(clean)
        if not clean:
            decision = self._decision(UNKNOWN, 0.0, "empty_input", "empty", False, True, started)
            return _with_ledger(decision, fingerprint)

        if _INJECTION_RE.search(clean):
            return _with_ledger(self._decision(CHAT, 0.99, "prompt_safety_boundary", "injection_guard", False, False, started, "guarded"), fingerprint)

        # Social and emotion turns are ordinary chat, never Knowledge/model
        # routing. They may still be handled by a deterministic response act.
        if _SOCIAL_RE.match(clean):
            return _with_ledger(self._decision(CHAT, 1.0, "social_response_act", "social_fast_path", False, False, started), fingerprint)
        if _EMOTION_RE.search(clean) and not _ACTION_RE.search(clean):
            return _with_ledger(self._decision(CHAT, 0.96, "emotional_response_act", "emotion_fast_path", False, False, started), fingerprint)

        # Self-questions are reads before the classifier gets a vote: the
        # selector owns them, and the gate must not turn them into chat.
        possessive = _POSSESSIVE_RE.search(clean)
        world_shape = bool(_WORLD_CHAT_RE.search(clean))
        # A world question that merely mentions something of the user's
        # ("what is my laptop supposed to get this hot") stays chat: a data
        # read names a short attribute phrase after the possessive ("my
        # garage code", "my flight"), not a trailing clause.
        tail_ok = not world_shape or bool(
            possessive
            and len(re.findall(r"[a-z0-9']+", clean[possessive.end():].lower())) <= 3
        )
        # "what's the wifi key at home" names the user's own belongings
        # without a pronoun; "how do I get home" is a different question, so
        # only the wh-openers count here.
        at_home = re.search(r"\bat\s+home\b|\bof\s+mine\b", clean, re.I)
        asks_about_home = bool(
            at_home and re.match(r"^(?:what|whats|which|when|where|who|whos)\b", clean, re.I)
        )
        if _DUE_QUESTION_RE.match(clean):
            return _with_ledger(self._decision(KNOWLEDGE, 0.94, "deadline_about_the_user", "due_question", True, False, started), fingerprint)

        if (
            (_SELF_QUESTION_RE.match(clean) and tail_ok)
            or (_SELF_INSTRUCTION_RE.match(clean) and tail_ok)
            or asks_about_home
        ):
            return _with_ledger(self._decision(KNOWLEDGE, 0.94, "question_about_the_user", "self_question", True, False, started), fingerprint)

        # Strong personal/action signals outrank generic world-question words.
        # This preserves compound personal requests for Knowledge validation.
        if _ACTION_RE.search(clean) or _PERSONAL_RE.search(clean):
            route, features = self._legacy_classifier(clean)
            if route == CHAT and _WORLD_CHAT_RE.search(clean) and not _ACTION_RE.search(clean):
                return _with_ledger(self._decision(CHAT, 0.90, "personal_word_is_world_question", features.get("rule") or "world_chat", False, False, started), fingerprint)
            if route == UNKNOWN:
                route = KNOWLEDGE if _ACTION_RE.search(clean) else UNKNOWN
            return _with_ledger(self._decision(route, 0.93 if route == KNOWLEDGE else 0.78, "personal_or_action_signal", features.get("rule") or "personal_boundary", route == KNOWLEDGE, route == UNKNOWN, started), fingerprint)

        # Clear general chat avoids any Knowledge model gate.
        if _WORLD_CHAT_RE.search(clean):
            return _with_ledger(self._decision(CHAT, 0.90, "world_chat_signal", "world_chat_fast_path", False, False, started), fingerprint)

        if allow_ambiguous:
            route, features = self._legacy_classifier(clean)
            if route != UNKNOWN:
                return _with_ledger(self._decision(route, 0.86, "symbolic_classifier", features.get("rule") or "symbolic_route", route == KNOWLEDGE, False, started), fingerprint)

        # A statement with no knowledge cue is conversation, not an unrouted
        # turn: the D4 confirmation replay recorded 15 unrouted decidable cases,
        # several of them ordinary statements ("I watched a documentary last
        # night"). A question with no evidence still abstains, because the
        # model must not guess an answer to it.
        if not clean.endswith("?"):
            return _with_ledger(self._decision(CHAT, 0.70, "statement_without_knowledge_cue", "plain_statement", False, True, started), fingerprint)

        return _with_ledger(self._decision(UNKNOWN, 0.0, "insufficient_route_evidence", "abstain", False, True, started), fingerprint)

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


def request_fingerprint(text: str) -> str:
    """A bounded request fingerprint for the ledger (never the prompt itself)."""
    clean = " ".join((text or "").split())
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16]


def _with_ledger(decision: RouteDecision, fingerprint: str) -> RouteDecision:
    """Stamp a decision with its bounded cost ledger row."""
    fast_cost = round(decision.latency_ms / 1000.0, 6)
    # Escalation is a model call: a small, explicit estimate, not a promise.
    escalation = 0.002 if decision.route == CHAT else 0.004
    return RouteDecision(
        route=decision.route,
        confidence=decision.confidence,
        reason=decision.reason,
        rule=decision.rule,
        requires_knowledge=decision.requires_knowledge,
        requires_model=decision.requires_model,
        safety=decision.safety,
        latency_ms=decision.latency_ms,
        cost_estimate=fast_cost,
        escalation_cost=escalation,
        counterfactual="unmeasured",
        outcome="unlabelled",
        fingerprint=fingerprint,
    )


def route_request(text: str) -> RouteDecision:
    """Convenience API used by tests and lightweight integrations."""
    return CoreRoutingEngine().decide(text)
