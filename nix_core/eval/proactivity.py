"""D8 - proactivity is off unless a policy says otherwise.

Three separate gates, in order: the master switch, the urgency tier, then the
timing (quiet hours, the owner speaking, and the interruption budget). With the
master switch off, ``evaluate`` returns no notification *and* no side effect -
it never registers a rule, writes a reminder or calls a dispatcher.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Any, Literal

Tier = Literal["critical", "important", "routine"]
DEFAULT_TIERS = ("critical", "important")


@dataclass
class ProactivityPolicy:
    enabled: bool = False
    tiers: tuple[str, ...] = DEFAULT_TIERS
    quiet_from: time = time(22, 0)
    quiet_until: time = time(7, 0)
    daily_budget: int = 3
    sent_today: int = 0
    registry: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        allowed = {"critical", "important", "routine"}
        unknown = set(self.tiers) - allowed
        if unknown:
            raise ValueError(f"unknown proactivity tier(s): {sorted(unknown)}")
        if self.daily_budget < 0:
            raise ValueError("daily_budget must be non-negative")

    def in_quiet_hours(self, at: time) -> bool:
        if self.quiet_from <= self.quiet_until:
            return self.quiet_from <= at < self.quiet_until
        return at >= self.quiet_from or at < self.quiet_until

    def evaluate(
        self,
        *,
        tier: Tier,
        at: time,
        owner_speaking: bool = False,
        urgent: bool = False,
    ) -> dict[str, Any]:
        """Return the decision; a blocked decision carries no side effect."""
        if not self.enabled:
            return {"notify": False, "reason": "policy_off", "side_effects": []}
        if tier not in self.tiers:
            return {"notify": False, "reason": "tier_not_allowed", "side_effects": []}
        if self.in_quiet_hours(at) and not (urgent and tier == "critical"):
            return {"notify": False, "reason": "quiet_hours", "side_effects": []}
        if owner_speaking:
            return {"notify": False, "reason": "owner_speaking", "side_effects": []}
        if self.sent_today >= self.daily_budget:
            return {"notify": False, "reason": "budget_exhausted", "side_effects": []}
        self.sent_today += 1
        self.registry.append({"tier": tier, "at": at.isoformat()})
        return {
            "notify": True,
            "reason": "allowed",
            "side_effects": ["interruption_registered"],
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "tiers": list(self.tiers),
            "quiet_from": self.quiet_from.isoformat(),
            "quiet_until": self.quiet_until.isoformat(),
            "daily_budget": self.daily_budget,
            "sent_today": self.sent_today,
        }


def default_policy() -> ProactivityPolicy:
    return ProactivityPolicy(enabled=False)
