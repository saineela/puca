from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class DecisionTrigger:
    """An explicit event submitted by a future producer.

    The engine intentionally does not infer or register trigger rules yet.
    Producers must submit a trigger name and payload explicitly.
    """

    name: str
    payload: Mapping[str, Any] | None = None
    occurred_at: datetime | None = None
    trigger_id: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A trigger name is required")


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
