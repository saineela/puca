from __future__ import annotations

from nix_decision import (
    CoreBoundary,
    CorePromptRequest,
    DecisionEngine,
    DecisionTrigger,
    Energy,
    Presence,
    UserStateSnapshot,
)


def test_engine_starts_without_calling_rules() -> None:
    engine = DecisionEngine()
    outcome = engine.evaluate(
        DecisionTrigger("future_event"), UserStateSnapshot(presence=Presence.HOME)
    )
    assert engine.registered_triggers == ()
    assert outcome.status == "ignored"
    assert outcome.reason == "trigger_not_registered"


def test_user_must_be_home_before_rule_can_dispatch() -> None:
    calls: list[CorePromptRequest] = []
    engine = DecisionEngine(calls.append)

    def rule(trigger: DecisionTrigger, state: UserStateSnapshot) -> CorePromptRequest:
        return CorePromptRequest(trigger, state, "Ask whether the user needs help.")

    engine.register("future_event", rule)
    outcome = engine.evaluate(
        DecisionTrigger("future_event"), UserStateSnapshot(presence=Presence.AWAY)
    )
    assert outcome.reason == "user_not_home"
    assert calls == []


def test_registered_rule_dispatches_state_to_core() -> None:
    received: list[dict] = []
    engine = DecisionEngine(CoreBoundary(received.append).dispatch)

    def rule(trigger: DecisionTrigger, state: UserStateSnapshot) -> CorePromptRequest:
        return CorePromptRequest(trigger, state, "Carefully check in without tiring the user.")

    engine.register("future_event", rule)
    state = UserStateSnapshot(
        presence=Presence.HOME, location="living_room", energy=Energy.TIRED, confidence=0.9
    )
    outcome = engine.evaluate(DecisionTrigger("future_event", {"kind": "reminder"}), state)

    assert outcome.status == "dispatched"
    assert received[0]["source"] == "nix_decision"
    assert received[0]["user_state"]["in_house"] is True
    assert received[0]["user_state"]["tired"] is True
    assert received[0]["instruction"].startswith("Carefully")


def test_rule_can_decline_to_speak() -> None:
    engine = DecisionEngine()
    engine.register("future_event", lambda trigger, state: None)
    outcome = engine.evaluate(
        DecisionTrigger("future_event"), UserStateSnapshot(presence=Presence.HOME)
    )
    assert outcome.reason == "rule_no_action"


class _OffPolicy:
    """D8: a policy object whose observable behaviour is nothing at all."""

    def __init__(self, reason: str = "disabled") -> None:
        self.reason = reason
        self.evaluations = 0
        self.side_effects: list[str] = []

    def evaluate(self, *, tier: str, at, owner_speaking: bool = False) -> dict:
        self.evaluations += 1
        return {"notify": False, "reason": self.reason, "side_effects": self.side_effects}


def test_proactivity_policy_gate_runs_before_any_rule_and_delivers_nothing_when_off() -> None:
    """With the policy off, a registered rule must not reach Core at all."""
    calls: list[CorePromptRequest] = []
    policy = _OffPolicy("disabled")
    engine = DecisionEngine(calls.append, proactivity_policy=policy)
    engine.register(
        "future_event",
        lambda trigger, state: CorePromptRequest(trigger, state, "Say something."),
    )
    outcome = engine.evaluate(
        DecisionTrigger("future_event", tier="critical"),
        UserStateSnapshot(presence=Presence.HOME),
    )
    assert outcome.status == "ignored"
    assert outcome.reason == "disabled"
    assert calls == [], "nothing is dispatched while the policy is off"
    assert policy.side_effects == []


def test_proactivity_policy_gate_lets_a_permitted_trigger_through() -> None:
    calls: list[CorePromptRequest] = []

    class _OnPolicy(_OffPolicy):
        def evaluate(self, *, tier: str, at, owner_speaking: bool = False) -> dict:
            self.evaluations += 1
            return {"notify": tier == "critical", "reason": "tier_allowed", "side_effects": []}

    policy = _OnPolicy()
    engine = DecisionEngine(calls.append, proactivity_policy=policy)
    engine.register(
        "future_event",
        lambda trigger, state: CorePromptRequest(trigger, state, "Say something."),
    )
    blocked = engine.evaluate(
        DecisionTrigger("future_event", tier="routine"),
        UserStateSnapshot(presence=Presence.HOME),
    )
    assert blocked.status == "ignored"
    allowed = engine.evaluate(
        DecisionTrigger("future_event", tier="critical"),
        UserStateSnapshot(presence=Presence.HOME),
    )
    assert allowed.status == "dispatched"
    assert len(calls) == 1
