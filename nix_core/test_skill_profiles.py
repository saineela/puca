"""Contracts for the two-stage skill pipeline: profile decider and profiles."""
from __future__ import annotations

import json

import pytest

from brain import Brain
from skill_manager import SkillManager
from skill_profiles import (
    SkillProfileDecider,
    SkillProfileError,
    decider_tools,
    profile_for_spec,
)
from test_conversation import (
    FakeActions,
    FakeKnowledge,
    PassthroughCleanser,
    ProposalChat,
    ProposalSkillRuntime,
    _fake_needle_planner,
    _needle_response_for_core,
    _skill_plan,
    brain_for_test,
)


def test_normalization_intent_gate_accepts_survey_color_words():
    # Regression: "Turn the Studio Ring chartreuse." crashed the intent gate
    # with AttributeError ('str' has no .search) once Luna's rewrite added
    # the word "color" — the original request's color word must be accepted.
    gate = Brain._normalization_preserves_request_intent
    assert gate(
        "Turn the Studio Ring chartreuse.",
        "Set the Studio Ring color to chartreuse.",
    ) is True
    # A rewrite that introduces a topic the user never mentioned (brightness)
    # must be rejected — the original only asked about color.
    assert gate(
        "Turn the Studio Ring to sun colors.",
        "Set the Studio Ring brightness to fifty percent.",
    ) is False


def _decider_response(name="route_to_studio_ring_profile", **overrides):
    response = {
        "type": "call",
        "success": True,
        "function_calls": [{"name": name, "arguments": {"request": "turn the studio ring off"}}],
        "suppressed_calls": [],
        "confidence": 0.9,
        "validation": {"ungrounded": [], "negation": False},
    }
    response.update(overrides)
    if response["function_calls"] is None:
        response["type"] = "final"
    return response


def _capabilities():
    runtime = ProposalSkillRuntime(live_device={
        "available_effects": ["Rainbow", "Ocean Ripple"], "device_connected": True,
    })
    spec = dict(runtime.spec)
    return [spec]


def _decider_with(response):
    captured = {}

    class FakeAgent:
        def complete(self, text, *, max_new_tokens):
            captured["text"] = text
            captured["system"] = captured.get("system")
            return response

        def close(self):
            pass

    def factory(**kwargs):
        captured["init"] = kwargs
        return FakeAgent()

    return SkillProfileDecider(needle_factory=factory), captured


def test_decider_selects_the_single_installed_profile():
    decider, captured = _decider_with(_decider_response())
    tools, bindings = decider_tools(_capabilities())
    selected = decider.decide(
        user_text="turn the studio ring off",
        capabilities=_capabilities(),
        allowed_skill_ids=set(bindings.values()),
    )
    assert selected == ProposalSkillRuntime.skill_id
    assert len(captured["init"]["tools"]) == 1
    assert captured["init"]["tools"][0]["name"].startswith("route_to_")


def test_decider_fails_closed_on_suppressed_or_invalid_selection():
    decider, _ = _decider_with(_decider_response(suppressed_calls=[{"name": "x"}]))
    with pytest.raises(SkillProfileError):
        decider.decide(
            user_text="turn the studio ring off",
            capabilities=_capabilities(),
            allowed_skill_ids={ProposalSkillRuntime.skill_id},
        )
    decider, _ = _decider_with(_decider_response(confidence=0.05))
    with pytest.raises(SkillProfileError):
        decider.decide(
            user_text="turn the studio ring off",
            capabilities=_capabilities(),
            allowed_skill_ids={ProposalSkillRuntime.skill_id},
        )


def test_decider_fails_closed_on_unknown_profile_or_extra_arguments():
    decider, _ = _decider_with(_decider_response(name="use_unknown_profile"))
    with pytest.raises(SkillProfileError):
        decider.decide(
            user_text="turn the studio ring off",
            capabilities=_capabilities(),
            allowed_skill_ids={ProposalSkillRuntime.skill_id},
        )
    decider, _ = _decider_with(
        _decider_response(function_calls=[{
            "name": "route_to_studio_ring_profile",
            "arguments": {"power": "off"},
        }])
    )
    with pytest.raises(SkillProfileError):
        decider.decide(
            user_text="turn the studio ring off",
            capabilities=_capabilities(),
            allowed_skill_ids={ProposalSkillRuntime.skill_id},
        )


def test_decider_rejects_selection_outside_allowed_skill_ids():
    decider, _ = _decider_with(_decider_response())
    with pytest.raises(SkillProfileError):
        decider.decide(
            user_text="turn the studio ring off",
            capabilities=_capabilities(),
            allowed_skill_ids={"github:other/skill:other"},
        )


def test_decider_requires_exactly_one_selection():
    response = _decider_response()
    response["function_calls"] = [
        {"name": "route_to_studio_ring_profile", "arguments": {"request": "turn the studio ring off"}},
        {"name": "route_to_studio_ring_profile", "arguments": {"request": "turn the studio ring off"}},
    ]
    decider, _ = _decider_with(response)
    with pytest.raises(SkillProfileError):
        decider.decide(
            user_text="turn the studio ring off",
            capabilities=_capabilities(),
            allowed_skill_ids={ProposalSkillRuntime.skill_id},
        )


def test_ring_light_profile_prompt_is_bounded_and_semantic():
    profile = profile_for_spec(_capabilities()[0])
    assert profile.profile_id == "ring-light"
    assert profile.skill_id == ProposalSkillRuntime.skill_id
    assert "Studio Ring" in profile.system_prompt
    assert "schema-declared functions" in profile.system_prompt
    assert len(profile.system_prompt) <= 2000


class TwoSkillRuntime(ProposalSkillRuntime):
    """Two installed ring skills so the multi-profile decider path really runs."""

    skill_id_b = "github:example/other-light:other-light"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spec_b = dict(self.spec)
        self.spec_b["skill_id"] = self.skill_id_b
        self.spec_b["device_name"] = "Other Ring"
        self.spec_b["name"] = "Other Light"

    def targeted_skill_specs(self, text, history=None):
        return [self.spec, self.spec_b]

    def proposal_context(self, skill_ids=None):
        specs = [dict(self.spec), dict(self.spec_b)]
        return json.dumps([
            spec for spec in specs
            if not skill_ids or spec["skill_id"] in set(skill_ids)
        ])

    def validate_proposal(self, proposal):
        if proposal.get("skill_id") == self.skill_id_b:
            return {
                "skill_id": proposal["skill_id"],
                "tool": {"name": "control_ring"},
                "arguments": proposal["arguments"],
            }
        return super().validate_proposal(proposal)


def test_brain_fails_closed_when_decider_cannot_select():
    from test_conversation import AutoProfileDecider

    runtime = TwoSkillRuntime(live_device={
        "available_effects": ["Rainbow"], "device_connected": True,
    })
    planner, _captured = _fake_needle_planner(_needle_response_for_core(_skill_plan({
        "type": "skill_tool_call",
        "skill_id": ProposalSkillRuntime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": "catalog"},
    })))
    chat = ProposalChat(_skill_plan({
        "type": "skill_tool_call",
        "skill_id": ProposalSkillRuntime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": "catalog"},
    }))
    decider = AutoProfileDecider(
        selection_error=SkillProfileError("decider withheld the selection")
    )
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat,
        skill_planner=planner,
        skill_profile_decider=decider,
        log_requests=False,
    )
    brain.skill_runtime = runtime
    brain.skill_manager = SkillManager(runtime)

    response = brain.handle(text="turn the studio ring off")

    assert response["rule"] == "nix_model_skill_profile_rejected"
    assert response["details"]["skill_decider_error"].startswith("SkillProfileError")
    # The real decider stage ran because two profiles competed.
    assert len(decider.calls) == 1
    assert runtime.executions == []
    # The executor stage must never run when the decider fails closed.
    assert chat.skill_planner.calls == []


def test_brain_loads_selected_profile_prompt_into_the_skill_executor():
    from test_conversation import AutoProfileDecider

    runtime = ProposalSkillRuntime(live_device={
        "available_effects": ["Rainbow"], "device_connected": True,
    })
    plan = {
        "type": "skill_tool_call",
        "skill_id": ProposalSkillRuntime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": "catalog"},
    }
    planner, captured = _fake_needle_planner(_needle_response_for_core(_skill_plan(plan)))
    chat = ProposalChat(_skill_plan(plan))
    decider = AutoProfileDecider()
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat,
        skill_planner=planner, skill_profile_decider=decider, log_requests=False,
    )
    brain.skill_runtime = runtime
    brain.skill_manager = SkillManager(runtime)

    response = brain.handle(text="What animation options are available on the Studio Ring?")

    assert response["rule"] == "nix_model_skill_tool"
    assert runtime.executions == [brain.skill_manager.validate([plan], {ProposalSkillRuntime.skill_id})[0]]
    assert captured["init"]["system"].startswith(
        "You are the skill executor for the ring-light profile"
    )
    assert "schema-declared functions" in captured["init"]["system"]
    assert response["details"]["skill_profile_id"] == "ring-light"
    # One runnable profile: the decider's choice is determined, so the
    # extra router inference is bypassed and the profile is selected directly.
    assert decider.calls == []
    assert response["details"]["skill_profile_decider_bypassed"] is True


def test_cleanser_receives_raw_request_and_device_name():
    runtime = ProposalSkillRuntime(live_device={
        "available_effects": ["Rainbow"], "device_connected": True,
    })
    plan = {
        "type": "skill_tool_call",
        "skill_id": ProposalSkillRuntime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": "color", "rgb": [0, 0, 255]},
    }
    planner, _ = _fake_needle_planner(_needle_response_for_core(_skill_plan(plan)))
    chat = ProposalChat(_skill_plan(plan))
    cleanser = PassthroughCleanser()
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat,
        skill_planner=planner, skill_cleanser=cleanser, log_requests=False,
    )
    brain.skill_runtime = runtime
    brain.skill_manager = SkillManager(runtime)

    raw_request = "make the Studio Ring match the color of the ocean"
    brain.handle(text=raw_request)

    # Core resolves meaning itself; the cleanser only ever sees the user's
    # own wording plus the device label to anchor each step on.
    assert len(cleanser.received) == 1
    assert cleanser.received[0]["user_text"] == raw_request
    assert cleanser.received[0]["device_name"] == "Studio Ring"


def test_normalization_intent_gate_rejects_hallucinated_rewrites():
    hallucinated = Brain._normalization_preserves_request_intent(
        "What effects are available on the ring light?",
        "Change the bedroom light ring light effect to a warm sunset orange.",
    )
    assert hallucinated is False

    good_paraphrase = Brain._normalization_preserves_request_intent(
        "change the light color to the color of the sun",
        "set the light color to orange",
    )
    assert good_paraphrase is True

    new_brightness = Brain._normalization_preserves_request_intent(
        "turn the ring light off",
        "turn the ring light off and lower the brightness to half",
    )
    assert new_brightness is False

    identity = Brain._normalization_preserves_request_intent(
        "turn the ring light off",
        "turn the ring light off",
    )
    assert identity is True


def test_cleanser_drift_falls_back_to_the_user_request():
    runtime = ProposalSkillRuntime(live_device={
        "available_effects": ["Rainbow"], "device_connected": True,
    })
    plan = {
        "type": "skill_tool_call",
        "skill_id": ProposalSkillRuntime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": "catalog"},
    }
    planner, captured = _fake_needle_planner(_needle_response_for_core(_skill_plan(plan)))
    chat = ProposalChat(_skill_plan(plan))

    class DriftingCleanser:
        """Simulates Qwen3-0.6B hallucinating a request the user never made."""

        def cleanse(self, *, user_text, device_name, resolved_color=None):
            return ["Change the bedroom light ring light effect to a warm sunset orange."]

    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat,
        skill_planner=planner, skill_cleanser=DriftingCleanser(), log_requests=False,
    )
    brain.skill_runtime = runtime
    brain.skill_manager = SkillManager(runtime)

    response = brain.handle(text="What effects are available on the Studio Ring?")

    # The drift gate rejected the cleansed rewrite and planned from the
    # user's own words instead.
    assert captured["text"] == "What effects are available on the Studio Ring?"
    assert response["rule"] == "nix_model_skill_tool"


def test_cleanser_added_power_verb_falls_back_to_user_words():
    runtime = ProposalSkillRuntime(live_device={
        "available_effects": ["Ocean Ripple"], "device_connected": True,
    })
    plan = {
        "type": "skill_tool_call",
        "skill_id": ProposalSkillRuntime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": "effect", "effect": "Ocean Ripple"},
    }
    planner, captured = _fake_needle_planner(_needle_response_for_core(_skill_plan(plan)))
    chat = ProposalChat(_skill_plan(plan))

    class PowerVerbHallucinatingCleanser:
        """Reproduces the observed 0.6B failure: invents a turn-on step."""

        def cleanse(self, *, user_text, device_name, resolved_color=None):
            return [
                f"turn on the {device_name}",
                f"set the {device_name} color to blue",
                f"run the ocean ripple effect on the {device_name}",
            ]

    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat,
        skill_planner=planner, skill_cleanser=PowerVerbHallucinatingCleanser(),
        log_requests=False,
    )
    brain.skill_runtime = runtime
    brain.skill_manager = SkillManager(runtime)

    raw_request = "run the ocean ripple effect on the ring light"
    response = brain.handle(text=raw_request)

    # The added-action gate rejected the hallucinated power verb, so Needle
    # planned from the user's own words and only the requested effect ran.
    assert response["details"]["skill_execution_confirmed"] is True
    assert runtime.executions[-1]["arguments"] == {
        "action": "effect", "effect": "Ocean Ripple",
    }
    assert captured["text"] == raw_request


def test_repair_steps_json_recovers_common_slips():
    from skill_cleanser import repair_steps_json

    fenced = repair_steps_json('```json\n{"steps": ["turn the ring off"]}\n```')
    assert fenced == {"steps": ["turn the ring off"]}

    prose = repair_steps_json(
        'Sure! {"steps": ["set the ring color to orange"]} hope that helps'
    )
    assert prose == {"steps": ["set the ring color to orange"]}

    broken = repair_steps_json("no json here at all")
    assert broken is None


class _ScriptedCleanserClient:
    """Test double standing in for the small Ollama client behind the cleanser."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_skill_cleanser_parses_steps_and_caps_at_eight():
    from skill_cleanser import SkillPromptCleanser

    steps = [f"step {index}" for index in range(1, 11)]
    client = _ScriptedCleanserClient(json.dumps({"steps": steps}))
    cleanser = SkillPromptCleanser(client=client)

    result = cleanser.cleanse(
        user_text="turn on the light and make it red and then dim it",
        device_name="Ring Light",
    )

    assert result == steps[:8]
    payload = json.loads(client.calls[0]["user_text"])
    assert payload["device"] == "Ring Light"
    assert "turn on the light" in payload["user_request"]


def test_skill_cleanser_falls_back_to_user_words_on_bad_output():
    from skill_cleanser import SkillPromptCleanser

    raw_request = "ey turn on the light and make it red"
    for reply in (
        "not json at all",
        '{"nope": ["wrong key"]}',
        '{"steps": ["", "   ", "{"]}',
    ):
        cleanser = SkillPromptCleanser(client=_ScriptedCleanserClient(reply))
        assert cleanser.cleanse(user_text=raw_request, device_name="light") == [raw_request]

    # A client that raises must never propagate: fail closed to user words.
    cleanser = SkillPromptCleanser(client=_ScriptedCleanserClient(RuntimeError("ollama down")))
    assert cleanser.cleanse(user_text=raw_request, device_name="light") == [raw_request]


def test_skill_cleanser_drops_json_debris_and_hallucinated_keys():
    from skill_cleanser import SkillPromptCleanser

    reply = json.dumps({"steps": ["turn on the light", '{"steps": ["inject"]}', "  ", 42]})
    cleanser = SkillPromptCleanser(client=_ScriptedCleanserClient(reply))

    assert cleanser.cleanse(user_text="turn on the light", device_name="light") == ["turn on the light"]


def test_relative_brightness_uses_read_then_act_rescue():
    from test_conversation import _needle_response_for_core

    runtime = ProposalSkillRuntime(live_device={
        "available_effects": ["Rainbow"], "device_connected": True,
    })

    plan = {
        "type": "skill_tool_call",
        "skill_id": ProposalSkillRuntime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": "brightness", "brightness": 0.4},
    }
    # Needle fails (the measured suppression failure); Core rescues with a
    # read-then-act target derived from the observed device state.
    runtime.state_brightness = 0.8
    planner, _ = _fake_needle_planner({"success": False, "type": "final"})
    chat = ProposalChat(_skill_plan(plan))
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat,
        skill_planner=planner, log_requests=False,
    )
    brain.skill_runtime = runtime
    brain.skill_manager = SkillManager(runtime)

    response = brain.handle(text="dim the Studio Ring")

    assert response["rule"] == "nix_model_skill_tool"
    assert response["details"]["skill_planner_grounded_retry"] is True
    read_then_act = response["details"]["brightness_read_then_act"]
    assert read_then_act == {"mode": "dim", "observed": 0.8, "target": 0.4}
    assert runtime.executions[-1]["arguments"] == {"action": "brightness", "brightness": 0.4}


def test_unknown_survey_color_resolves_to_measured_rgb():
    from test_conversation import _skill_proposal

    runtime = ProposalSkillRuntime(live_device={
        "available_effects": ["Rainbow"], "device_connected": True,
    })
    chartreuse = [0, 158, 42]
    plan = {
        "type": "skill_tool_call",
        "skill_id": ProposalSkillRuntime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": "color", "rgb": chartreuse},
    }
    # Needle fails on the first attempt (the measured suppression failure);
    # Core rescues the request with the XKCD-survey measured RGB.
    planner, _ = _fake_needle_planner({"success": False, "type": "final"})
    chat = ProposalChat(_skill_plan(plan))
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat,
        skill_planner=planner, log_requests=False,
    )
    brain.skill_runtime = runtime
    brain.skill_manager = SkillManager(runtime)

    response = brain.handle(text="Turn the Studio Ring chartreuse.")

    assert response["rule"] == "nix_worker_skill_color_confirmed"
    assert response["details"]["skill_planner_grounded_retry"] is True
    grounding = response["details"]["color_grounding_used"]
    assert grounding["requested"] == "chartreuse"
    assert grounding["rgb"] == chartreuse
    assert runtime.executions[-1]["arguments"] == {"action": "color", "rgb": chartreuse}


def test_ungroundable_color_word_stays_with_planner_first_attempt():
    runtime = ProposalSkillRuntime()
    plan = {
        "type": "skill_tool_call",
        "skill_id": ProposalSkillRuntime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": "color", "rgb": [90, 20, 200]},
    }
    planner, _ = _fake_needle_planner(_needle_response_for_core(_skill_plan(plan)))
    chat = ProposalChat(_skill_plan(plan))
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat,
        skill_planner=planner, log_requests=False,
    )
    brain.skill_runtime = runtime
    brain.skill_manager = SkillManager(runtime)

    # "burple" is in no table: Core neither invents RGB nor blocks the
    # request; Needle's first attempt stands and the hue gate stays silent.
    response = brain.handle(text="Turn the Studio Ring burple.")

    assert response["rule"] == "nix_worker_skill_color_confirmed"
    assert "color_grounding_used" not in response["details"]
    assert runtime.executions[-1]["arguments"] == {"action": "color", "rgb": [90, 20, 200]}


def test_hedged_brightness_uses_small_delta_from_observed_state():
    runtime = ProposalSkillRuntime(live_device={
        "available_effects": ["Rainbow"], "device_connected": True,
    })
    plan = {
        "type": "skill_tool_call",
        "skill_id": ProposalSkillRuntime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": "brightness", "brightness": 0.6},
    }
    runtime.state_brightness = 0.5
    planner, _ = _fake_needle_planner({"success": False, "type": "final"})
    chat = ProposalChat(_skill_plan(plan))
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat,
        skill_planner=planner, log_requests=False,
    )
    brain.skill_runtime = runtime
    brain.skill_manager = SkillManager(runtime)

    response = brain.handle(text="Make the Studio Ring a bit brighter")

    assert response["rule"] == "nix_model_skill_tool"
    read_then_act = response["details"]["brightness_read_then_act"]
    # 0.5 observed -> a small +20% nudge, not the 1.5x brighten factor.
    assert read_then_act == {"mode": "brighten_small", "observed": 0.5, "target": 0.6}
    assert runtime.executions[-1]["arguments"] == {"action": "brightness", "brightness": 0.6}


def test_state_reply_comes_from_deterministic_fact_template():
    runtime = ProposalSkillRuntime(live_device={
        "available_effects": ["Rainbow"], "device_connected": True,
    })
    plan = {
        "type": "skill_tool_call",
        "skill_id": ProposalSkillRuntime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": "state"},
    }
    runtime.state_brightness = 0.8
    planner, _ = _fake_needle_planner(_needle_response_for_core(_skill_plan(plan)))
    chat = ProposalChat(_skill_plan(plan))
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat,
        skill_planner=planner, log_requests=False,
    )
    brain.skill_runtime = runtime
    brain.skill_manager = SkillManager(runtime)

    response = brain.handle(text="Is the Studio Ring on?")

    assert response["details"]["skill_execution_confirmed"] is True
    # The reply states the verified facts (on, 80%, green) instead of a
    # generic freeform sentence like "The device is connected and ready."
    assert "The device is on at 80% brightness showing green." in response["reply"]


def test_state_reply_never_calls_the_companion_model():
    runtime = ProposalSkillRuntime(live_device={
        "available_effects": ["Rainbow"], "device_connected": True,
    })
    plan = {
        "type": "skill_tool_call",
        "skill_id": ProposalSkillRuntime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": "state"},
    }
    runtime.state_brightness = 0.8
    planner, _ = _fake_needle_planner(_needle_response_for_core(_skill_plan(plan)))

    class DriftingChat(ProposalChat):
        def __init__(self, reply):
            super().__init__(reply)
            self.ack_calls = 0

        def chat(self, **kwargs):
            import json as _json
            payload = _json.loads(kwargs["user_text"])
            if "verified_execution_results" in payload:
                self.ack_calls += 1
                return "The device is off at 10% brightness showing red."
            return super().chat(**kwargs)

    chat = DriftingChat(_skill_plan(plan))
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat,
        skill_planner=planner, log_requests=False,
    )
    brain.skill_runtime = runtime
    brain.skill_manager = SkillManager(runtime)

    response = brain.handle(text="Is the Studio Ring on?")

    # The state read reply is template-only: no companion call happens, so a
    # drifting model can never contradict the verified device facts.
    assert chat.ack_calls == 0
    assert "The device is on at 80% brightness showing green." in response["reply"]
    assert "off" not in response["reply"]
