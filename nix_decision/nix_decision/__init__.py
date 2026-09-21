"""Trigger-driven foundation for proactive nix_core interactions."""

from .core_boundary import CoreBoundary
from .engine import DecisionEngine
from .triggers import CorePromptRequest, DecisionOutcome, DecisionTrigger
from .user_state import Energy, Presence, UserStateSnapshot

__all__ = [
    "CoreBoundary",
    "CorePromptRequest",
    "DecisionEngine",
    "DecisionOutcome",
    "DecisionTrigger",
    "Energy",
    "Presence",
    "UserStateSnapshot",
]
