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

    def __init__(
        self,
        core_callback: CoreCallback | None = None,
        proactivity_policy=None,
    ) -> None:
        self._rules: dict[str, Callable[[DecisionTrigger, UserStateSnapshot], CorePromptRequest | None]] = {}
        self._core_callback = core_callback
        # D8: off unless a policy object is supplied. With no policy the
        # engine behaves exactly as before (the foundation keeps no default
        # rules), so existing callers cannot become proactive by accident.
        self._proactivity_policy = proactivity_policy

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
        if self._proactivity_policy is not None:
            from datetime import datetime

            decision = self._proactivity_policy.evaluate(
                tier=str(getattr(trigger, "tier", None) or "routine"),
                at=datetime.now().time(),
                owner_speaking=bool(getattr(user_state, "speaking", False)),
            )
            if not decision.get("notify"):
                return DecisionOutcome("ignored", str(decision.get("reason") or "policy"))
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
