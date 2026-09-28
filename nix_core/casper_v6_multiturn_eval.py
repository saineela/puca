"""Multi-turn V6 evaluation contracts.

The cases are intentionally anonymized and judge state/behavior, not a fixed
internal execution path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class MultiTurnCase:
    case_id: str
    turns: tuple[str, ...]
    required_final: tuple[str, ...]
    forbidden_final: tuple[str, ...] = ()
    max_final_questions: int = 0
    note: str = ""


CASES = (
    MultiTurnCase(
        "positive-to-family-clarification",
        ("I got great news today!", "My sister is doing well."),
        required_final=("which",),
        max_final_questions=1,
        note="positive tone does not remove person ambiguity",
    ),
    MultiTurnCase(
        "tired-follow-up",
        ("I had a terrible day.", "I'm exhausted; keep it short."),
        required_final=("brief",),
        forbidden_final=("what's on your mind", "tell me more"),
        max_final_questions=0,
    ),
    MultiTurnCase(
        "state-supersession",
        ("Person A is sick.", "Person A is feeling better now.", "How is Person A?"),
        required_final=("better",),
        forbidden_final=("sick",),
        note="new state overrides historical state",
    ),
    MultiTurnCase(
        "clarification-continuation",
        ("How is my sister?", "Sister B."),
        required_final=("sister b",),
        max_final_questions=0,
        note="answer continues pending clarification instead of starting a new request",
    ),
    MultiTurnCase(
        "memory-plus-chat",
        ("Remember that I like tea and tell me a joke.", "That was funny."),
        required_final=("tea",),
        max_final_questions=0,
        note="durable memory and casual conversation remain separate",
    ),
)


def check_final(case: MultiTurnCase, reply: str) -> list[str]:
    text = " ".join((reply or "").casefold().split())
    failures = [f"missing: {item}" for item in case.required_final if item.casefold() not in text]
    failures.extend(f"forbidden: {item}" for item in case.forbidden_final if item.casefold() in text)
    questions = (reply or "").count("?")
    if questions > case.max_final_questions:
        failures.append(f"question_count={questions}")
    if not text:
        failures.append("empty_reply")
    return failures


def evaluate(responder: Callable[[Sequence[str]], str]) -> dict[str, object]:
    results = []
    for case in CASES:
        reply = responder(case.turns)
        results.append({"id": case.case_id, "reply": reply, "failures": check_final(case, reply)})
    return {
        "passed": sum(not row["failures"] for row in results),
        "total": len(results),
        "results": results,
    }
