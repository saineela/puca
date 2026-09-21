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
