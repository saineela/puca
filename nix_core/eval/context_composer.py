"""D3 - a deterministic context composer with five section budgets.

The composer is deliberately pure: it takes the candidate sections and returns
the assembled prompt *and* the size of every section for the request log. It
knows nothing about the model, the knowledge service or the clock, so the same
input always composes the same prompt.

Eviction follows the approved decision, not age:

1. policy is never evicted; no other section may spend budget while policy is
   still being dropped;
2. memory is ranked by relevance to the current question (subject/topic match,
   then recency), not creation order;
3. a question about the present prefers the current, non-superseded record;
4. a question about the past deliberately admits superseded records whose
   validity window covers the asked period, ranked above unrelated current ones;
5. ties break on a stable key so composition is reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Mapping, Sequence

SECTIONS = ("policy", "memory", "records", "history", "tools")

#: Initial hypotheses from the proposal; Wave 2 tunes them against D2 and only
#: then enforces them. They are recorded with the model and machine they were
#: measured on in ``BUDGET_PROVENANCE``.
INITIAL_BUDGETS = {
    "policy": 1_600,
    "memory": 1_200,
    "records": 900,
    "history": 1_200,
    "tools": 900,
}

BUDGET_PROVENANCE = {
    "kind": "hypothesis",
    "tuned_on": None,
    "note": (
        "Initial guesses from the proposal. Enforced only after the D2 budget "
        "curve supports a value; re-tune when model, tasks or hardware change."
    ),
}


@dataclass(frozen=True)
class SectionItem:
    """One candidate line (a memory record, a message, a tool result)."""

    text: str
    relevance: float = 0.0
    current: bool = True
    superseded: bool = False
    valid_from: str = ""
    valid_until: str = ""
    key: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def stable_key(self) -> str:
        return self.key or self.text


@dataclass(frozen=True)
class ComposedContext:
    prompt: str
    sizes: dict[str, int]
    dropped: dict[str, int]
    budgets: dict[str, int]

    def report(self) -> dict[str, Any]:
        return {
            "sections": dict(self.sizes),
            "dropped": dict(self.dropped),
            "budgets": dict(self.budgets),
            "provenance": dict(BUDGET_PROVENANCE),
        }


def estimate_tokens(text: str) -> int:
    """A deterministic, conservative token estimate (no tokenizer needed)."""
    words = len(re.findall(r"\S+", text or ""))
    characters = len(text or "")
    return max(0, max(words, (characters + 3) // 4))


def _terms(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", (text or "").casefold())
        if len(word) > 2
    }


def relevance(item: SectionItem, question: str) -> float:
    """Rank by overlap with the current question, then by the item's own score."""
    asked = _terms(question)
    if not asked:
        return float(item.relevance)
    overlap = len(asked & _terms(item.text))
    return overlap * 10.0 + float(item.relevance)


_HISTORICAL_RE = re.compile(
    r"\b(?:was|were|used\s+to|back\s+in|last\s+(?:month|year|week)|"
    r"in\s+(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)|before|previously|histor(?:y|ical))\b",
    re.IGNORECASE,
)


def is_historical_question(question: str) -> bool:
    return bool(_HISTORICAL_RE.search(question or ""))


def _window_covers(item: SectionItem, question: str) -> bool:
    """A cheap, deterministic check that a superseded fact fits an asked period."""
    text = (question or "").casefold()
    months = (
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
    )
    named = [month for month in months if month in text]
    if named:
        blob = f"{item.valid_from} {item.valid_until}".casefold()
        for month in named:
            number = months.index(month) + 1
            if month in blob or f"-{number:02d}-" in blob:
                return True
        return False
    return True


def rank_items(items: Iterable[SectionItem], question: str) -> list[SectionItem]:
    """Deterministic D3 ordering with current-fact precedence."""
    historical = is_historical_question(question)
    ranked = list(items)
    if historical:
        def sort_key(item: SectionItem) -> tuple[Any, ...]:
            return (
                0 if (item.superseded and _window_covers(item, question)) else 1,
                -relevance(item, question),
                item.stable_key(),
            )
    else:
        def sort_key(item: SectionItem) -> tuple[Any, ...]:
            return (
                0 if item.current and not item.superseded else 1,
                -relevance(item, question),
                item.stable_key(),
            )
    ranked.sort(key=sort_key)
    return ranked


def _fit(items: Sequence[SectionItem], budget: int) -> tuple[list[SectionItem], int]:
    kept: list[SectionItem] = []
    used = 0
    for item in items:
        cost = estimate_tokens(item.text)
        if used + cost <= budget:
            kept.append(item)
            used += cost
    return kept, len(items) - len(kept)


def compose(
    *,
    policy: str,
    memory: Iterable[SectionItem] = (),
    records: Iterable[SectionItem] = (),
    history: Iterable[SectionItem] = (),
    tools: Iterable[SectionItem] = (),
    question: str = "",
    budgets: Mapping[str, int] | None = None,
) -> ComposedContext:
    """Compose the five sections within their budgets and report their sizes."""
    active = {**INITIAL_BUDGETS, **dict(budgets or {})}
    for name, value in active.items():
        if name not in SECTIONS:
            raise ValueError(f"unknown context section: {name}")
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"section budget must be a non-negative int: {name}")

    # Policy is not negotiable. It is included whole even when it exceeds its
    # own budget, because dropping half a policy is worse than a long prompt.
    policy_text = " ".join((policy or "").split())
    policy_used = estimate_tokens(policy_text)
    sizes = {name: 0 for name in SECTIONS}
    dropped = {name: 0 for name in SECTIONS}
    sizes["policy"] = policy_used

    candidates = {
        "memory": rank_items(memory, question),
        "records": rank_items(records, question),
        "history": rank_items(history, question),
        "tools": rank_items(tools, question),
    }
    kept: dict[str, list[SectionItem]] = {}
    for name in ("memory", "records", "history", "tools"):
        items, lost = _fit(candidates[name], active[name])
        kept[name] = items
        dropped[name] = lost
        sizes[name] = sum(estimate_tokens(item.text) for item in items)

    lines = [policy_text] if policy_text else []
    for name in ("memory", "records", "history", "tools"):
        if not kept[name]:
            continue
        lines.append(f"\n[{name.upper()}]")
        lines.extend(item.text for item in kept[name])
    return ComposedContext(
        prompt="\n".join(lines),
        sizes=sizes,
        dropped=dropped,
        budgets=active,
    )


def section_sizes(prompt: str, *, policy: str = "") -> dict[str, int]:
    """Report sizes for an already-assembled prompt.

    Used by the logging-only path: the existing prompt stays authoritative, and
    the composer only labels how much of it belongs to each known section.
    """
    text = prompt or ""
    sizes = {name: 0 for name in SECTIONS}
    sizes["policy"] = estimate_tokens(policy or text)
    if not policy:
        return sizes
    remaining = text.replace(policy, "", 1)
    # Everything not policy is reported as memory until a caller supplies a
    # better split; the five keys are always present, which is the log contract.
    sizes["memory"] = estimate_tokens(remaining)
    return sizes
