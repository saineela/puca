from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

#: D8: every trigger declares its urgency, so the proactivity policy can gate it
#: by tier, quiet hours and an interruption budget. ``routine`` is the default:
#: a producer that says nothing gets the least intrusive treatment it can have.
TIERS = ("routine", "important", "critical")


@dataclass(frozen=True)
class DecisionTrigger:
    """An explicit event submitted by a future producer.

    The engine intentionally does not infer or register trigger rules yet.
    Producers must submit a trigger name and payload explicitly, and declare
    the urgency tier the policy will judge them by.
    """

    name: str
    payload: Mapping[str, Any] | None = None
    occurred_at: datetime | None = None
    trigger_id: str | None = None
    tier: str = "routine"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A trigger name is required")
        if self.tier not in TIERS:
            raise ValueError(f"tier must be one of: {', '.join(TIERS)}")


@dataclass(frozen=True)
class CorePromptRequest:
    """Safe handoff from Decision to Core; Core owns final wording."""

    trigger: DecisionTrigger
    user_state: Any
    instruction: str


@dataclass(frozen=True)
class DecisionOutcome:
    """Result of evaluating one explicit trigger."""

    status: str
    reason: str
    prompt_request: CorePromptRequest | None = None
