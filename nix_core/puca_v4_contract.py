"""Nix PUCA V4 policy contract.

This module deliberately contains no model calls and no database writes.  It is the
small, testable boundary between Core policy and the Casper V6 verbalizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal, Mapping, Sequence

ResponseAct = Literal[
    "acknowledge",
    "answer",
    "clarify",
    "reflect",
    "brief_support",
    "tool_result",
    "boundary",
    "end",
]
Emotion = Literal["neutral", "positive", "distressed", "tired", "uncertain"]
Grounding = Literal["none", "memory", "knowledge_result", "current_state", "event"]


@dataclass(frozen=True)
class PucaV4Envelope:
    """Authoritative instructions that Casper may verbalize, but not override."""

    response_act: ResponseAct
    question_budget: int = 0
    emotion: Emotion = "neutral"
    grounding: Grounding = "none"
    current_state_over_history: bool = True
    must_preserve: tuple[str, ...] = ()
    raw_user_request: str = ""
    authoritative_context: str = ""
    confidence: float = 1.0
    pending_clarification: bool = False
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.question_budget <= 2:
            raise ValueError("question_budget must be between 0 and 2")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.raw_user_request.strip():
            raise ValueError("raw_user_request is required")
        if self.response_act == "clarify" and self.question_budget < 1:
            raise ValueError("clarify requires a question budget")
        if self.response_act == "tool_result" and self.grounding not in {
            "event",
            "knowledge_result",
            "current_state",
        }:
            raise ValueError("tool_result requires structured grounding")

    def as_prompt_block(self) -> str:
        """Render a stable, compact block for the final Casper prompt."""
        preserve = ", ".join(self.must_preserve) or "none"
        return "\n".join(
            (
                "PUCA_POLICY_V4",
                f"response_act={self.response_act}",
                f"question_budget={self.question_budget}",
                f"emotion={self.emotion}",
                f"grounding={self.grounding}",
                f"current_state_over_history={str(self.current_state_over_history).lower()}",
                f"must_preserve={preserve}",
                f"pending_clarification={str(self.pending_clarification).lower()}",
                "Use the policy and authoritative context; do not invent facts, actions, dates, or names.",
            )
        )


def infer_envelope(
    request: str,
    *,
    knowledge_result: Mapping[str, Any] | None = None,
    emotion: Emotion = "neutral",
    authoritative_context: str = "",
) -> PucaV4Envelope:
    """Choose a response act from structured state, never from prose alone."""
    result = dict(knowledge_result or {})
    nested = result.get("result")
    if isinstance(nested, Mapping):
        result = dict(nested)
    operation = str(result.get("operation") or "")
    failed = result.get("ok") is False or operation in {
        "KNOWLEDGE_UNAVAILABLE",
        "ERROR",
    }
    if operation == "NEEDS_CLARIFICATION":
        return build_envelope(
            request,
            response_act="clarify",
            question_budget=1,
            emotion=emotion,
            grounding="current_state" if result.get("candidates") else "knowledge_result",
            must_preserve=tuple(str(item) for item in result.get("candidates", ())),
            authoritative_context=authoritative_context,
            confidence=0.95,
            pending_clarification=True,
        )
    if failed:
        return build_envelope(
            request,
            response_act="boundary",
            emotion=emotion,
            grounding="knowledge_result",
            authoritative_context=authoritative_context,
            confidence=0.9,
        )
    if operation in {"CREATE", "UPDATE", "CANCEL", "STORE_STATE", "SUPERSEDE_STATE"}:
        grounding: Grounding = "event" if result.get("record_type") == "calendar_event" or result.get("actions") else "current_state" if "state" in result else "knowledge_result"
        preserve = []
        for key in ("title", "when", "recurrence", "timezone"):
            if result.get(key):
                preserve.append(key)
        return build_envelope(
            request,
            response_act="tool_result",
            emotion=emotion,
            grounding=grounding,
            must_preserve=preserve,
            authoritative_context=authoritative_context,
        )
    if operation in {"STORE_KEYS", "DUPLICATE"}:
        return build_envelope(
            request,
            response_act="acknowledge",
            emotion=emotion,
            grounding="memory",
            authoritative_context=authoritative_context,
        )
    return build_envelope(
        request,
        response_act="answer",
        emotion=emotion,
        grounding="knowledge_result" if result else "none",
        authoritative_context=authoritative_context,
    )


def verify_reply(
    envelope: PucaV4Envelope,
    reply: str,
    *,
    deterministic: str = "",
    knowledge_result: Mapping[str, Any] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Validate generated prose against V4 policy before it reaches a user."""
    text = " ".join((reply or "").split())
    failures: list[str] = []
    if not text:
        failures.append("empty_reply")
    if re.search(r"<think>|<\|im_|Thinking Process:", text, re.IGNORECASE):
        failures.append("control_trace")
    if text.count("?") > envelope.question_budget:
        failures.append("question_budget_exceeded")
    result = dict(knowledge_result or {})
    nested = result.get("result")
    if isinstance(nested, Mapping):
        result = dict(nested)
    operation = str(result.get("operation") or "")
    if envelope.response_act == "clarify" and "?" not in text:
        failures.append("clarification_without_question")
    if envelope.response_act == "tool_result" and result.get("ok") is False:
        if re.search(r"\b(?:done|scheduled|created|set|cancelled|updated)\b", text, re.IGNORECASE):
            failures.append("failed_operation_claimed_successfully")
    if envelope.grounding == "current_state" and envelope.current_state_over_history:
        state = str(result.get("state") or "").strip()
        if state and state.casefold() not in text.casefold() and deterministic:
            # A paraphrase is acceptable if the model retains at least one
            # meaningful word from the authoritative current state.
            state_words = [word.casefold() for word in re.findall(r"[A-Za-z]+", state) if len(word) > 3]
            if state_words and not any(word in text.casefold() for word in state_words):
                failures.append("current_state_dropped")
    return not failures, tuple(failures)


def build_envelope(
    request: str,
    *,
    response_act: ResponseAct = "answer",
    question_budget: int = 0,
    emotion: Emotion = "neutral",
    grounding: Grounding = "none",
    must_preserve: Sequence[str] = (),
    authoritative_context: str = "",
    confidence: float = 1.0,
    pending_clarification: bool = False,
) -> PucaV4Envelope:
    """Create a validated policy envelope without invoking an LLM."""
    return PucaV4Envelope(
        response_act=response_act,
        question_budget=question_budget,
        emotion=emotion,
        grounding=grounding,
        must_preserve=tuple(must_preserve),
        raw_user_request=request,
        authoritative_context=authoritative_context,
        confidence=confidence,
        pending_clarification=pending_clarification,
    )
