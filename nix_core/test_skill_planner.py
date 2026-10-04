"""Needle planner contracts; every test inspects proposals and never dispatches tools."""
from __future__ import annotations

import copy
import os

import pytest

from config import SKILL_PLANNER_MODEL
from skill_planner import SkillPlanner, SkillPlannerError


SKILL_ID = "github:example/ring-light:ring-light"
RGB_SCHEMA = {"type": "array", "minItems": 3, "maxItems": 3,
              "items": {"type": "integer", "minimum": 0, "maximum": 255}}
CAPABILITIES = [{
    "skill_id": SKILL_ID,
    "name": "Ring Light",
    "device_name": "Studio Ring",
    "tools": [{
        "name": "control_ring",
        "description": "Control the ring.",
        "input_schema": {
            "type": "object",
            "oneOf": [
                {"type": "object", "properties": {"action": {"type": "string", "enum": ["catalog", "color_catalog", "state", "on", "off"]}}, "required": ["action"]},
                {"type": "object", "properties": {"action": {"type": "string", "enum": ["color"]}, "rgb": RGB_SCHEMA}, "required": ["action", "rgb"]},
                {"type": "object", "properties": {"action": {"type": "string", "enum": ["brightness"]}, "brightness": {"type": "number", "minimum": 0.05, "maximum": 1.0}}, "required": ["action", "brightness"]},
                {"type": "object", "properties": {"action": {"type": "string", "enum": ["effect"]}, "effect": {"type": "string", "minLength": 1, "maxLength": 64}}, "required": ["action", "effect"]},
            ],
            "properties": {"action": {"type": "string", "enum": ["catalog", "color_catalog", "state", "on", "off", "color", "brightness", "effect"]}, "rgb": RGB_SCHEMA, "brightness": {"type": "number"}, "effect": {"type": "string"}},
            "required": ["action"],
            "additionalProperties": False,
        },
    }],
}]


class FakeNeedle:
    def __init__(self, response, captured):
        self.response = response
        self.captured = captured

    def complete(self, text, *, max_new_tokens):
        self.captured.setdefault("texts", []).append(text)
        self.captured["text"] = text
        self.captured["max_new_tokens"] = max_new_tokens
        response = self.response
        if isinstance(response, list):
            index = min(len(self.captured["texts"]) - 1, len(response) - 1)
            response = response[index]
        return response

    def close(self):
        self.captured["closed"] = True


def _planner(response, captured=None):
    captured = captured if captured is not None else {}

    def factory(**kwargs):
        captured["init"] = kwargs
        return FakeNeedle(response, captured)

    return SkillPlanner(needle_factory=factory), captured


def test_requested_model_identifier_defaults_to_user_supplied_exact_id():
    assert SKILL_PLANNER_MODEL == "Needle3-20L-121M"
    assert SkillPlanner().model == "Needle3-20L-121M"


def _success(name, arguments, *, confidence=0.8):
    return {
        "type": "call",
        "success": True,
        "function_calls": [{"name": name, "arguments": arguments}],
        "suppressed_calls": [],
        "confidence": confidence,
        "validation": {"negation": False, "ungrounded": []},
    }


def test_needle_proposes_schema_bound_call_and_never_runs_tools(monkeypatch):
    tools, bindings = SkillPlanner._needle_tool_definitions(CAPABILITIES)
    color = next(tool for tool in tools if tool["name"] == "set_color")
    planner, captured = _planner(_success("set_color", {"rgb": [12, 34, 56]}))
    monkeypatch.delenv("NEEDLE_TELEMETRY", raising=False)
    calls = planner.plan(
        system_prompt="Only plan color actions for the requested device.",
        user_text="Set RGB(12,34,56)",
        capabilities=CAPABILITIES,
    )
    assert calls == [{
        "type": "skill_tool_call", "skill_id": SKILL_ID, "tool": "control_ring",
        "arguments": {"rgb": [12, 34, 56], "action": "color"},
    }]
    assert captured["init"]["tools"] == [color]  # explicit request safely narrows choices
    assert captured["init"]["system"] == "Only plan color actions for the requested device."
    assert captured["init"]["generation"] == 3 and captured["init"]["stateless"] is True
    assert captured["max_new_tokens"] == 512 and captured["closed"]
    assert os.environ["NEEDLE_TELEMETRY"] == "0"
    assert os.environ["DO_NOT_TRACK"] == "1"
    assert "set_color" in bindings
    assert not hasattr(FakeNeedle, "run")


def test_power_and_read_actions_have_distinct_semantic_tools():
    tools, bindings = SkillPlanner._needle_tool_definitions(CAPABILITIES)
    by_name = {item["name"]: item for item in tools}
    assert {"turn_on", "turn_off", "read_state", "list_effects"} <= set(by_name)
    assert by_name["turn_on"]["parameters"] == {"type": "object", "properties": {}, "required": []}
    assert by_name["turn_off"]["parameters"] == {"type": "object", "properties": {}, "required": []}
    assert by_name["list_effects"]["description"].lower().find("animations") >= 0
    assert bindings["read_state"][2] == {"__call__": "state"}
    assert bindings["list_effects"][2] == {"__call__": "catalog"}
    assert SkillPlanner._requested_tool_groups("Show the Studio Ring state", CAPABILITIES) == {"read"}


def test_read_catalog_intent_has_semantic_single_action_tool():
    tools, bindings = SkillPlanner._needle_tool_definitions(
        CAPABILITIES, {"read"}, {"catalog"}
    )
    assert [item["name"] for item in tools] == ["list_effects"]
    assert bindings["list_effects"][2] == {"__call__": "catalog"}
    planner, captured = _planner(_success("list_effects", {}))
    planner.plan(
        system_prompt="List firmware effects.",
        user_text="List the available animations on the bedroom light",
        capabilities=CAPABILITIES,
    )
    assert [item["name"] for item in captured["init"]["tools"]] == ["list_effects"]


def test_multi_step_power_tools_explicitly_allow_ordered_repeated_calls():
    tools, bindings = SkillPlanner._needle_tool_definitions(CAPABILITIES, {"power"})
    by_name = {item["name"]: item for item in tools}
    assert {"turn_on", "turn_off"} == set(by_name)
    # Compound requests are planned step-by-step: each step runs one calibrated
    # single-call inference, and Core's multi-action gate enforces completeness.
    assert SkillPlanner._split_plan_steps(
        "Turn the bedroom light off, then turn it back on"
    ) == ["Turn the bedroom light off", "turn it back on"]
    assert SkillPlanner._split_plan_steps("Turn the bedroom light off") is None
    assert bindings["turn_off"][2] == {"__call__": "off"}
    assert bindings["turn_on"][2] == {"__call__": "on"}


def test_requested_intents_retain_read_and_ordered_power_groups():
    assert SkillPlanner._requested_tool_groups(
        "List the available animations on the bedroom light", CAPABILITIES
    ) == {"read"}
    assert SkillPlanner._requested_tool_groups(
        "Turn the bedroom light off, then turn it back on", CAPABILITIES
    ) == {"power"}


def test_semantic_read_tool_response_becomes_catalog_proposal():
    planner, _captured = _planner(_success("list_effects", {}))
    calls = planner.plan(
        system_prompt="List firmware effects.",
        user_text="List the available animations on the bedroom light",
        capabilities=CAPABILITIES,
    )
    assert calls == [{
        "type": "skill_tool_call", "skill_id": SKILL_ID, "tool": "control_ring",
        "arguments": {"action": "catalog"},
    }]


def test_ordered_repeated_power_proposals_preserve_user_order():
    # One calibrated single-call inference per step, concatenated in order.
    planner, captured = _planner([
        _success("turn_off", {}),
        _success("turn_on", {}),
    ])
    calls = planner.plan(
        system_prompt="Preserve all requested device actions in order.",
        user_text="Turn the bedroom light off, then turn it back on",
        capabilities=CAPABILITIES,
    )
    assert [item["arguments"]["action"] for item in calls] == ["off", "on"]
    assert captured["texts"] == ["Turn the bedroom light off", "turn it back on"]


def test_missing_ordered_power_step_fails_core_multi_action_count():
    # A step whose Needle run withholds fails the whole plan closed; a partial
    # multi-action plan can never reach execution.
    suppressed = _success("turn_on", {})
    suppressed["suppressed_calls"] = [{"name": "turn_on", "arguments": {}}]
    suppressed["function_calls"] = []
    planner, _captured = _planner([_success("turn_off", {}), suppressed])
    with pytest.raises(SkillPlannerError, match="withheld"):
        planner.plan(
            system_prompt="Preserve all requested device actions in order.",
            user_text="Turn the bedroom light off, then turn it back on",
            capabilities=CAPABILITIES,
        )


@pytest.mark.parametrize(("user_text", "group", "tool_name"), [
    ("Switch light to green", "color", "set_color"),
    ("Turn the Studio Ring off", "power", "turn_off"),
    ("Set brightness to 40 percent", "brightness", "set_brightness"),
    ("Show the Studio Ring state", "read", "read_state"),
])
def test_clear_intent_narrows_tool_choices(user_text, group, tool_name):
    tools, _ = SkillPlanner._needle_tool_definitions(CAPABILITIES, {group})
    assert tool_name in {tool["name"] for tool in tools}
    assert SkillPlanner._requested_tool_groups(user_text, CAPABILITIES) == {group}


def test_color_named_light_power_request_does_not_select_color():
    assert SkillPlanner._requested_tool_groups("Turn the green light off", CAPABILITIES) == {"power"}


def test_negated_request_fails_closed():
    with pytest.raises(SkillPlannerError, match="Negated"):
        SkillPlanner._requested_tool_groups("Do not turn the light off", CAPABILITIES)


def test_unsupported_clear_action_fails_closed():
    with pytest.raises(SkillPlannerError, match="unavailable in the installed schema"):
        SkillPlanner._requested_tool_groups("Set the ring effect to Aurora", CAPABILITIES)


def test_unclassified_request_keeps_full_toolset():
    assert SkillPlanner._requested_tool_groups("do something with the ring", CAPABILITIES) is None
    tools, _ = SkillPlanner._needle_tool_definitions(CAPABILITIES)
    assert {"set_color", "turn_on", "turn_off", "set_brightness", "read_state", "list_effects"} <= {item["name"] for item in tools}


@pytest.mark.parametrize("response", [
    {"type": "call", "success": False},
    {"type": "call", "success": True, "function_calls": [], "suppressed_calls": [], "confidence": 0.9, "validation": {"ungrounded": [], "negation": False}},
    {"type": "call", "success": True, "function_calls": [{"name": "unknown", "arguments": {}}], "suppressed_calls": [], "confidence": 0.9, "validation": {"ungrounded": [], "negation": False}},
    {"type": "call", "success": True, "function_calls": [{"name": "set_color", "arguments": {"rgb": [0, 255, 0]}}], "suppressed_calls": [{"name": "set_power"}], "confidence": 0.9, "validation": {"ungrounded": [], "negation": False}},
    {"type": "call", "success": True, "function_calls": [{"name": "set_color", "arguments": {"rgb": [0, 255, 0]}}], "suppressed_calls": [], "confidence": 0.09, "validation": {"ungrounded": [], "negation": False}},
    {"type": "call", "success": True, "function_calls": [{"name": "set_color", "arguments": {"rgb": [0, 255, 0]}}], "suppressed_calls": [], "confidence": 0.9, "validation": {"ungrounded": ["set_color.rgb"], "negation": False}},
    {"type": "call", "success": True, "function_calls": [{"name": "set_color", "arguments": {"rgb": [0, 255, 0]}}], "suppressed_calls": [], "confidence": 0.9, "validation": {"ungrounded": [], "negation": True}},
])
def test_invalid_suppressed_low_confidence_or_ungrounded_responses_fail_closed(response):
    planner, _ = _planner(response)
    with pytest.raises(SkillPlannerError):
        planner.plan(system_prompt="", user_text="Switch light to green", capabilities=CAPABILITIES)


def test_effect_is_exposed_only_with_fresh_catalog_and_is_enum_limited():
    capabilities = copy.deepcopy(CAPABILITIES)
    tools, _ = SkillPlanner._needle_tool_definitions(capabilities)
    assert "set_effect" not in {item["name"] for item in tools}
    with pytest.raises(SkillPlannerError, match="unavailable in the installed schema"):
        SkillPlanner._requested_tool_groups("Run effect Aurora Pulse on the ring", capabilities)
    capabilities[0]["live_device"] = {"available_effects": ["Aurora Pulse", "Ocean Ripple"]}
    assert SkillPlanner._requested_tool_groups("Run effect Aurora Pulse on the ring", capabilities) == {"effect"}
    tools, _ = SkillPlanner._needle_tool_definitions(capabilities)
    effect = next(item for item in tools if item["name"] == "set_effect")
    assert effect["parameters"]["properties"]["effect"]["enum"] == ["Aurora Pulse", "Ocean Ripple"]


def test_negation_flag_without_negation_cue_allows_read_request():
    # Regression: Needle flagged "what color is the device" as negation,
    # blocking verified state reads. Core honors the flag only when the
    # request actually contains a negation cue or a control verb.
    response = {
        "type": "call",
        "success": True,
        "function_calls": [{"name": "read_state", "arguments": {}}],
        "suppressed_calls": [],
        "confidence": 0.8,
        "validation": {"ungrounded": [], "negation": True},
    }
    planner, _ = _planner(response)
    calls = planner.plan(
        system_prompt="",
        user_text="what color is the ring light right now",
        capabilities=CAPABILITIES,
    )
    assert calls[0]["arguments"]["action"] == "state"


def test_negation_flag_with_control_verb_still_fails_closed():
    response = {
        "type": "call",
        "success": True,
        "function_calls": [{"name": "turn_on", "arguments": {}}],
        "suppressed_calls": [],
        "confidence": 0.8,
        "validation": {"ungrounded": [], "negation": True},
    }
    planner, _ = _planner(response)
    with pytest.raises(SkillPlannerError):
        planner.plan(
            system_prompt="",
            user_text="turn on the ring light",
            capabilities=CAPABILITIES,
        )


def test_missing_validation_block_fails_closed_without_attribute_error():
    # Regression: Needle3 measurably omits the validation block on some
    # question wordings; the planner must fail closed with a planner error,
    # not crash with AttributeError ('NoneType' has no attribute 'get').
    response = {
        "type": "call",
        "success": True,
        "function_calls": [{"name": "read_state", "arguments": {}}],
        "suppressed_calls": [],
        "confidence": 0.8,
        "validation": None,
    }
    planner, _ = _planner(response)
    with pytest.raises(SkillPlannerError, match="omitted valid grounding validation"):
        planner.plan(
            system_prompt="",
            user_text="what color is the ring light right now",
            capabilities=CAPABILITIES,
        )


def test_grounded_read_retry_emits_deterministic_state_read():
    # Regression: the live color question ("What color is the ring light right
    # now?") depended on Needle3 returning a usable validation block. Under
    # the grounded retry a current-state color question emits the state read
    # deterministically, without calling the model.
    planner, captured = _planner({"unexpected": True})
    calls = planner.plan(
        system_prompt="",
        user_text="What color is the ring light right now?",
        capabilities=CAPABILITIES,
        force_grounded=True,
    )
    assert len(calls) == 1
    assert calls[0]["arguments"] == {"action": "state"}
    assert "init" not in captured  # the fake Needle factory was never used


def test_grounded_read_retry_keeps_catalog_wording_on_color_catalog():
    planner, _ = _planner({"unexpected": True})
    calls = planner.plan(
        system_prompt="",
        user_text="What colors are available on the ring light?",
        capabilities=CAPABILITIES,
        force_grounded=True,
    )
    assert len(calls) == 1
    assert calls[0]["arguments"] == {"action": "color_catalog"}
