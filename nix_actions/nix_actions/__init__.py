"""Nix Actions Engine.

A deterministic (no-AI) action scheduler and hardware trigger layer for
the Nix ecosystem. Knowledge decides *what* should happen; Actions
decides *when it fires* and *executes it*.

Public API:
    ActionsEngine  - schedule / list / cancel / run-due actions
    get_handlers   - hardware & software handler registry
    main           - CLI entry point
"""

from .engine import ActionsEngine, Action, CaptureEvent, SessionInfo
from .handlers import get_handler, list_handlers

__all__ = [
    "ActionsEngine",
    "Action",
    "CaptureEvent",
    "SessionInfo",
    "get_handler",
    "list_handlers",
]
