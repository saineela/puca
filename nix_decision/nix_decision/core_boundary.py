from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .triggers import CorePromptRequest


class CoreBoundary:
    """Adapter between nix_decision and nix_core.

    The boundary transports structured context only. nix_core remains
    responsible for deciding whether and how to speak to the user.
    """

    def __init__(self, handler: Callable[[dict[str, Any]], Any]) -> None:
        self._handler = handler

    def dispatch(self, request: CorePromptRequest) -> Any:
        return self._handler(
            {
                "source": "nix_decision",
                "trigger": {
                    "name": request.trigger.name,
                    "id": request.trigger.trigger_id,
                    "occurred_at": request.trigger.occurred_at.isoformat()
                    if request.trigger.occurred_at
                    else None,
                    "payload": dict(request.trigger.payload or {}),
                },
                "user_state": request.user_state.as_dict(),
                "instruction": request.instruction,
            }
        )
