"""Held-out behavioral checks for Casper V6.

This module does not call a model. It defines the cases and deterministic
checks that a model runner must apply to V5 and V6 outputs identically.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    request: str
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    max_questions: int = 0
    note: str = ""


CASES = (
    EvalCase("greeting", "hi", forbidden=("how can i help", "what can i do", "digital companion", "what's on your mind"), note="brief natural opening"),
    EvalCase("alive-check", "hello? are you alive", required=("here",), forbidden=("digital companion", "what's on your mind", "personal companion"), note="ordinary social check-in must not trigger product copy"),
    EvalCase("tired", "I'm exhausted and don't want a long conversation", forbidden=("what's on your mind", "tell me more"), max_questions=0),
    EvalCase("current-state", "How is Maya now?", required=("better",), forbidden=("sick",), note="current state beats history"),
    EvalCase("ambiguous-person", "My sister is doing well", required=("which",), max_questions=1),
    EvalCase("creative-no-memory", "Tell me a story using my name", forbidden=("your name is", "my name is"), max_questions=1),
    EvalCase("recurring-event", "Remind me to take medicine every 2 days", required=("2 days",), forbidden=("i'll remember that",)),
    EvalCase("multi-intent", "Remember that I like tea and tell me a joke", required=("tea",), note="both clauses must survive"),
    EvalCase("identity", "Who created Casper?", required=("Sai Neela",), forbidden=("Microsoft", "Qwen")),
)


def check(case: EvalCase, reply: str) -> list[str]:
    """Return failures; an empty list means this case passes."""
    text = " ".join((reply or "").casefold().split())
    failures = [f"missing: {item}" for item in case.required if item.casefold() not in text]
    failures.extend(f"forbidden: {item}" for item in case.forbidden if item.casefold() in text)
    questions = len(re.findall(r"\?", reply or ""))
    if questions > case.max_questions:
        failures.append(f"question_count={questions}, max={case.max_questions}")
    if not text:
        failures.append("empty reply")
    return failures


def evaluate(responder: Callable[[str], str]) -> dict[str, object]:
    results = []
    for case in CASES:
        reply = responder(case.request)
        failures = check(case, reply)
        results.append({"id": case.case_id, "reply": reply, "failures": failures})
    passed = sum(not row["failures"] for row in results)
    return {"passed": passed, "total": len(results), "results": results}


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: python casper_v6_eval.py response_file.json")
    responses = json.loads(open(sys.argv[1], encoding="utf-8").read())
    print(json.dumps(evaluate(lambda request: responses[request]), indent=2, ensure_ascii=False))
