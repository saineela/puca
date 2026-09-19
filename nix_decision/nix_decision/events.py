from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Observation:
    """
    A raw observation entering Nix Decision.

    Observations are facts/signals.
    They are NOT decisions.
    """

    kind: str
    source: str
    timestamp: datetime

    data: dict[str, Any] = field(default_factory=dict)

    confidence: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "Observation confidence must be between 0.0 and 1.0"
            )
