from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Decision:
    """
    What Nix Decision currently believes it should do.

    This is deliberately separate from observations.
    """

    state: str
    action: str
    attention: str

    reason: str

    timestamp: datetime

    def identity(self):
        """
        Used to determine whether the decision actually changed.
        """

        return (
            self.state,
            self.action,
            self.attention,
            self.reason,
        )
