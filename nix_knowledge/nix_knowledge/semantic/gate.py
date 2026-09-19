from __future__ import annotations

import re

"""
StorageGate: decides what is ALLOWED to enter the knowledge base.

This is the trust boundary of the whole system. Its decisions are
deterministic and policy-driven so that storage behavior never
depends on a probabilistic judgment alone:

  1. DUPLICATE   cosine >= 0.92 vs an existing chunk -> REJECT
  1b. QUESTION   the candidate is a question (wh-word, aux-fronting,
                 or a trailing "?") -> REJECT: questions are requests
                 for recall, never knowledge (fixed the measured
                 25-30% question leakage / precision-0.32 defect)
  2. FIGURATIVE  sarcasm/irony probability >= 0.5    -> REJECT
                 (flagged, never stored as literal fact; Core is
                  informed so it can respond to the rhetoric)
  3. IMPORTANCE  importance < 0.55 and no extracted fact -> REJECT
  4. UNCERTAIN   head confidence too low on a would-be key fact
                 -> ESCALATE (Core decides, not the gate)

Everything else passes with its full classification payload.
"""

from dataclasses import dataclass, field
from typing import Any

from .extraction import KnowledgeCandidate

# tunable policy constants
# NOTE: sarcasm probabilities live on the CALIBRATED scale (the neural
# head's raw output carries a tweet-domain prior; training calibrates
# the gate threshold, default 0.8 -- see classifier._save_meta).
DUP_THRESHOLD = 0.92
SARCASM_REJECT = 0.8
IMPORTANCE_FLOOR = 0.55
LOW_CONFIDENCE = 0.45

# Deterministic question detection (see rule 1b). Fronted wh-words
# and aux+pronoun inversion are unambiguous; a trailing "?" alone is
# decisive. Aux + CONTENT word ("can speak spanish" after interjection
# stripping) is deliberately NOT matched - only aux + pronoun/there.
_QUESTION_WH_RE = re.compile(
    r"^(who|what|when|where|why|which|whose|whom|how)\b", re.I
)
_QUESTION_AUX_RE = re.compile(
    r"^(do|does|did|is|are|was|were|am|can|could|will|would|should"
    r"|shall|may|might|must|have|has|had)\s+"
    r"(i|you|we|they|he|she|it|there|anyone|anybody|someone)\b",
    re.I,
)

# Casual knowledge bypass: knowledge-bearing categories may store at
# a LOWER importance floor, but only when the classifier is confident
# (blocks chatter that was misclassified) and the statement still has
# some salience.
KNOWLEDGE_CATEGORIES = {
    "personal_fact", "preference", "event", "relationship",
    "skill", "goal",
}
BYPASS_IMPORTANCE = 0.30
BYPASS_MIN_CATEGORY_CONFIDENCE = 0.50


@dataclass
class GateDecision:
    decision: str          # "store" | "reject" | "escalate"
    reason: str
    candidate: KnowledgeCandidate
    detail: dict[str, Any] = field(default_factory=dict)


class StorageGate:
    """Policy layer between extraction and the knowledge base."""

    def __init__(
        self,
        *,
        dup_threshold: float = DUP_THRESHOLD,
        sarcasm_reject: float = SARCASM_REJECT,
        importance_floor: float = IMPORTANCE_FLOOR,
        low_confidence: float = LOW_CONFIDENCE,
        bypass_importance: float = BYPASS_IMPORTANCE,
        bypass_min_category_conf: float = (
            BYPASS_MIN_CATEGORY_CONFIDENCE
        ),
    ):
        self.dup_threshold = dup_threshold
        self.sarcasm_reject = sarcasm_reject
        self.importance_floor = importance_floor
        self.low_confidence = low_confidence
        self.bypass_importance = bypass_importance
        self.bypass_min_category_conf = bypass_min_category_conf

    def evaluate(
        self,
        candidate: KnowledgeCandidate,
        *,
        similarity_score: float | None = None,
    ) -> GateDecision:
        """
        Decide storage for one candidate.

        similarity_score: cosine of the nearest existing chunk, if a
        vector store was available during extraction.
        """
        c = candidate

        # 1. duplicates --------------------------------------------------
        if c.duplicate_of is not None or (
            similarity_score is not None
            and similarity_score >= self.dup_threshold
        ):
            return GateDecision(
                decision="reject",
                reason="duplicate",
                candidate=c,
                detail={
                    "duplicate_of": c.duplicate_of,
                    "similarity": similarity_score,
                },
            )

        # 1b. questions are recall requests, never knowledge ---------
        text = (c.text or "").strip()
        if (
            text.endswith("?")
            or _QUESTION_WH_RE.match(text)
            or _QUESTION_AUX_RE.match(text)
        ):
            return GateDecision(
                decision="reject",
                reason="question_not_knowledge",
                candidate=c,
                detail={
                    "text": text[:120],
                    "note": (
                        "questions request recall; store answers, "
                        "not the question"
                    ),
                },
            )

        # 2. figurative language ------------------------------------------
        if c.sarcasm_flagged or (
            c.sarcasm_probability >= self.sarcasm_reject
        ):
            return GateDecision(
                decision="reject",
                reason="figurative_language",
                candidate=c,
                detail={
                    "sarcasm_probability": c.sarcasm_probability,
                    "note": (
                        "not stored as literal fact; surfaced to Core "
                        "as possible sarcasm/irony"
                    ),
                },
            )

        # 3. importance / relevance ---------------------------------------
        # Store when: a confident knowledge-bearing category carries
        # casual salience, OR there is an extracted fact, OR the
        # importance is high outright.
        # 'general' statements without a fact are chatter by default.
        has_fact = c.fact is not None
        casual_knowledge = (
            c.category in KNOWLEDGE_CATEGORIES
            and c.category_confidence >= self.bypass_min_category_conf
            and c.importance >= self.bypass_importance
        )
        if c.category == "general" and not has_fact:
            return GateDecision(
                decision="reject",
                reason="general_chatter",
                candidate=c,
                detail={"category": c.category,
                        "importance": c.importance},
            )
        if (
            not has_fact
            and not casual_knowledge
            and c.importance < self.importance_floor
        ):
            return GateDecision(
                decision="reject",
                reason="low_importance",
                candidate=c,
                detail={"importance": c.importance,
                        "category": c.category,
                        "category_confidence": c.category_confidence},
            )

        # 4. uncertainty escalation ---------------------------------------
        key_category = c.category in (
            "personal_fact", "relationship", "event"
        )
        if (
            (has_fact or casual_knowledge)
            and key_category
            and c.confidence < self.low_confidence
        ):
            return GateDecision(
                decision="escalate",
                reason="low_confidence_key_fact",
                candidate=c,
                detail={
                    "confidence": c.confidence,
                    "note": "Core should confirm before storage",
                },
            )

        # chatter guard: low-salience, low-confidence statements that
        # slipped past importance are escalated, not stored silently
        if (
            not has_fact
            and not key_category
            and c.importance < self.importance_floor
            and c.confidence < self.low_confidence
        ):
            return GateDecision(
                decision="escalate",
                reason="uncertain_low_salience",
                candidate=c,
                detail={"importance": c.importance,
                        "confidence": c.confidence},
            )

        return GateDecision(
            decision="store",
            reason="passes_policy",
            candidate=c,
            detail={"importance": c.importance},
        )
