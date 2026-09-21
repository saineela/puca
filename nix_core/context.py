"""Context budgeting for Nix Core chat requests.

The durable Knowledge service is the long-term memory. This module only
prepares short-term conversational context so a long session cannot crowd out
the current request or the grounded MEMORY block.
"""
from __future__ import annotations

from typing import Any


def select_context(
    turns: list[dict[str, Any]],
    *,
    max_turns: int,
    max_chars: int,
) -> list[dict[str, str]]:
    """Keep the newest valid turns within both count and character budgets."""
    selected: list[dict[str, str]] = []
    used = 0
    for turn in reversed(turns):
        role = turn.get("role")
        content = turn.get("content")
        if role not in {"user", "assistant"} or not content:
            continue
        text = " ".join(str(content).split())
        if not text:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(text) > remaining:
            # Keep the end of an older turn only when it fits; recent
            # turns are considered first and remain complete whenever
            # possible.
            text = text[-remaining:]
        selected.append({"role": role, "content": text})
        used += len(text)
        if len(selected) >= max_turns:
            break
    return list(reversed(selected))
