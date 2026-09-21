from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .triggers import CorePromptRequest, DecisionOutcome, DecisionTrigger
from .user_state import Presence, UserStateSnapshot

CoreCallback = Callable[[CorePromptRequest], Any]


class DecisionEngine:
    """Foundation for proactive decisions.

    No calling rules are installed by default. Future rules should be added
    explicitly with ``register`` and should return a CorePromptRequest only
    after considering the supplied UserStateSnapshot.
    """

    def __init__(self, core_callback: CoreCallback | None = None) -> None:
        self._rules: dict[str, Callable[[DecisionTrigger, UserStateSnapshot], CorePromptRequest | None]] = {}
        self._core_callback = core_callback

    @property
    def registered_triggers(self) -> tuple[str, ...]:
        return tuple(self._rules)

    def register(
        self,
        trigger_name: str,
        rule: Callable[[DecisionTrigger, UserStateSnapshot], CorePromptRequest | None],
    ) -> None:
        """Register a future rule explicitly; this does not call Core."""
        name = trigger_name.strip()
        if not name:
            raise ValueError("A trigger name is required")
        if name in self._rules:
            raise ValueError(f"Trigger already registered: {name}")
        self._rules[name] = rule

    def evaluate(
        self,
        trigger: DecisionTrigger,
        user_state: UserStateSnapshot,
    ) -> DecisionOutcome:
        """Evaluate one trigger and optionally deliver its result to Core.

        Availability is a hard gate: away or unknown users never receive a
        proactive Core call. An unregistered trigger is a deliberate no-op.
        """
        if user_state.presence is not Presence.HOME:
            return DecisionOutcome("ignored", "user_not_home")
        rule = self._rules.get(trigger.name)
        if rule is None:
            return DecisionOutcome("ignored", "trigger_not_registered")

        request = rule(trigger, user_state)
        if request is None:
            return DecisionOutcome("ignored", "rule_no_action")
        if self._core_callback is not None:
            self._core_callback(request)
        return DecisionOutcome("dispatched", "core_callback_invoked", request)

    def evaluate_many(
        self,
        triggers: Iterable[DecisionTrigger],
        user_state: UserStateSnapshot,
    ) -> list[DecisionOutcome]:
        return [self.evaluate(trigger, user_state) for trigger in triggers]
