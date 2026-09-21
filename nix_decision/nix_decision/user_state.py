from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Presence(str, Enum):
    """Whether the user is known to be available in the house."""

    HOME = "home"
    AWAY = "away"
    UNKNOWN = "unknown"


class Energy(str, Enum):
    """The currently observed tiredness state."""

    TIRED = "tired"
    NOT_TIRED = "not_tired"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class UserStateSnapshot:
    """Minimal context Decision may pass to Core.

    This is an observation snapshot, not a diagnosis. ``Presence.HOME`` is
    deliberately required before a trigger can request a user-facing Core
    prompt; phone location alone is not treated as proof that the user is
    available.
    """

    presence: Presence = Presence.UNKNOWN
    location: str | None = None
    energy: Energy = Energy.UNKNOWN
    confidence: float = 0.0
    observed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("User-state confidence must be between 0.0 and 1.0")

    @property
    def user_is_home(self) -> bool:
        return self.presence is Presence.HOME

    @property
    def user_is_tired(self) -> bool | None:
        if self.energy is Energy.TIRED:
            return True
        if self.energy is Energy.NOT_TIRED:
            return False
        return None

    def as_dict(self) -> dict[str, str | float | None | bool]:
        return {
            "presence": self.presence.value,
            "in_house": self.user_is_home,
            "location": self.location,
            "energy": self.energy.value,
            "tired": self.user_is_tired,
            "confidence": self.confidence,
            "observed_at": self.observed_at.isoformat() if self.observed_at else None,
        }
