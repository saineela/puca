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
    assert captured["max_new_tokens"] == 1024 and captured["closed"]
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


MUSIC_SKILL_ID = "github:example/opendeck:opendeck"
MUSIC_CAPABILITIES = [{
    "skill_id": MUSIC_SKILL_ID,
    "name": "OpenDeck",
    "device_name": "OpenDeck",
    "tools": [{
        "name": "opendeck_music",
        "description": "Search and play music, read the library, and fetch lyrics.",
        "input_schema": {
            "type": "object",
            "oneOf": [
                {"type": "object", "properties": {"action": {"type": "string", "enum": ["search"]}, "query": {"type": "string", "minLength": 1, "maxLength": 200}}, "required": ["action", "query"]},
                {"type": "object", "properties": {"action": {"type": "string", "enum": ["library"]}, "kind": {"type": "string", "enum": ["playlists", "likes", "history", "stats"]}}, "required": ["action", "kind"]},
                {"type": "object", "properties": {"action": {"type": "string", "enum": ["play"]}, "query": {"type": "string", "maxLength": 200}, "video_id": {"type": "string", "maxLength": 64}}, "required": ["action"]},
                {"type": "object", "properties": {"action": {"type": "string", "enum": ["queue"]}, "op": {"type": "string", "enum": ["list", "add", "remove", "clear", "next", "play", "move", "front", "next_up", "shuffle", "repeat"]}, "query": {"type": "string", "minLength": 1, "maxLength": 200}, "video_id": {"type": "string", "maxLength": 64}, "index": {"type": "integer", "minimum": 0, "maximum": 99}, "to": {"type": "integer", "minimum": 0, "maximum": 99}, "on": {"type": "boolean"}, "mode": {"type": "string", "enum": ["off", "all", "one"]}}, "required": ["action", "op"]},
                {"type": "object", "properties": {"action": {"type": "string", "enum": ["lyrics"]}, "title": {"type": "string", "maxLength": 300}}, "required": ["action"]},
                {"type": "object", "properties": {"action": {"type": "string", "enum": ["now_playing"]}}, "required": ["action"]},
                {"type": "object", "properties": {"action": {"type": "string", "enum": ["save"]}, "kind": {"type": "string", "enum": ["like", "unlike", "playlist_add", "playlist_create"]}, "video_id": {"type": "string", "maxLength": 64}, "playlist_id": {"type": "string", "maxLength": 100}, "name": {"type": "string", "maxLength": 120}, "title": {"type": "string", "maxLength": 300}, "artist": {"type": "string", "maxLength": 200}, "album": {"type": "string", "maxLength": 300}}, "required": ["action", "kind"]},
            ],
            "properties": {
                "action": {"type": "string", "enum": ["search", "library", "play", "queue", "lyrics", "now_playing", "save"]},
                "kind": {"type": "string", "enum": ["playlists", "likes", "history", "stats", "like", "unlike", "playlist_add", "playlist_create"]},
                "playlist_id": {"type": "string", "maxLength": 100},
                "name": {"type": "string", "maxLength": 120},
                "artist": {"type": "string", "maxLength": 200},
                "album": {"type": "string", "maxLength": 300},
                "query": {"type": "string", "maxLength": 200},
                "video_id": {"type": "string", "maxLength": 64},
                "op": {"type": "string", "enum": ["list", "add", "remove", "clear", "next", "play", "move", "front", "next_up", "shuffle", "repeat"]},
                "index": {"type": "integer", "minimum": 0, "maximum": 99},
                "to": {"type": "integer", "minimum": 0, "maximum": 99},
                "on": {"type": "boolean"},
                "mode": {"type": "string", "enum": ["off", "all", "one"]},
                "title": {"type": "string", "maxLength": 300},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    }],
}]

SPEAKER_CAPABILITIES = [{
    "skill_id": "github:example/ring-light:echo-connect",
    "name": "Echo Connect",
    "device_name": "Nix Dev",
    "tools": [{
        "name": "control_ring",
        "description": "Control the ring and the speaker.",
        "input_schema": {
            "type": "object",
            "oneOf": [
                {"type": "object", "properties": {"action": {"type": "string", "enum": ["state", "mute", "unmute", "stop_speaking", "pause", "resume"]}}, "required": ["action"]},
                {"type": "object", "properties": {"action": {"type": "string", "enum": ["volume"]}, "volume": {"type": "number", "minimum": 0.0, "maximum": 1.0}}, "required": ["action", "volume"]},
                {"type": "object", "properties": {"action": {"type": "string", "enum": ["say"]}, "text": {"type": "string", "minLength": 1, "maxLength": 500}}, "required": ["action", "text"]},
            ],
            "properties": {
                "action": {"type": "string", "enum": ["state", "mute", "unmute", "stop_speaking", "pause", "resume", "volume", "say"]},
                "volume": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "text": {"type": "string", "maxLength": 500},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    }],
}]


def test_music_wording_selects_the_provider_action_it_names():
    assert SkillPlanner._requested_tool_groups("queue up Hotel California", MUSIC_CAPABILITIES) == {"queue"}
    assert SkillPlanner._requested_tool_groups("what is in my queue?", MUSIC_CAPABILITIES) == {"queue"}
    assert SkillPlanner._requested_tool_groups("show my liked songs", MUSIC_CAPABILITIES) == {"library"}
    assert SkillPlanner._requested_tool_groups("show me the lyrics", MUSIC_CAPABILITIES) == {"lyrics"}
    assert SkillPlanner._requested_tool_groups("play music: Bohemian Rhapsody", MUSIC_CAPABILITIES) == {"play"}
    # A word the vocabulary does not own leaves the choice to the model rather
    # than inventing an action the schema does not declare.
    assert SkillPlanner._requested_tool_groups("good morning Luna", MUSIC_CAPABILITIES) is None
    # A history question is a listing, not a play.
    assert SkillPlanner._requested_tool_groups("what did I listen to yesterday", MUSIC_CAPABILITIES) == {"library"}
    # Opening one named playlist is the planner's job (it needs the name), even
    # though the wording selects the library action.
    assert SkillPlanner._requested_tool_groups("open my Ultimate playlist", MUSIC_CAPABILITIES) == {"library"}
    assert SkillPlanner._grounded_music_call(MUSIC_CAPABILITIES, {"library"}, "open my Ultimate playlist") is None


def test_queue_wording_is_planned_by_core_without_calling_the_model():
    # Measured live: "queue up <song>" was planned as a search, and "what's in my
    # queue" and "clear the queue" were withheld for confidence. The track and the
    # operation come from the user's own words, so Core owns the whole call.
    expected = [
        ("queue up Hotel California", {"action": "queue", "op": "add", "query": "hotel california"}),
        ("add Bohemian Rhapsody to the queue", {"action": "queue", "op": "add", "query": "bohemian rhapsody"}),
        # "the song" points at which thing is meant, not the title, so it comes
        # off the track before the provider is asked to search for it.
        ("add the song believer to the queue", {"action": "queue", "op": "add", "query": "believer"}),
        ("add the song Believer to my queue", {"action": "queue", "op": "add", "query": "believer"}),
        ("queue up the track Believer", {"action": "queue", "op": "add", "query": "believer"}),
        ("OpenDeck and add the song believer to the queue", {"action": "queue", "op": "add", "query": "believer"}),
        ("what's in my queue?", {"action": "queue", "op": "list"}),
        ("play the next queued track", {"action": "queue", "op": "next"}),
        ("clear the queue", {"action": "queue", "op": "clear"}),
    ]
    for text, arguments in expected:
        planner, captured = _planner({"unexpected": True})
        calls = planner.plan(system_prompt="", user_text=text, capabilities=MUSIC_CAPABILITIES)
        assert calls == [{
            "type": "skill_tool_call", "skill_id": MUSIC_SKILL_ID,
            "tool": "opendeck_music", "arguments": arguments,
        }], text
        assert "init" not in captured, text


def test_play_wording_is_planned_by_core_without_calling_the_model():
    # Measured live across one session: "play believer" played, "play starboy"
    # failed, "play the song starboy" played, "play maps by maroon 5" failed —
    # the same request planned by the model twice with different answers, and
    # "I didn't send a device command because the dedicated skill planner did
    # not produce a valid, safe plan" every time it lost. The track is the
    # user's own words after the verb, so Core owns the whole call.
    expected = [
        ("play believer", {"action": "play", "query": "believer"}),
        ("Play believer", {"action": "play", "query": "believer"}),
        ("play starboy", {"action": "play", "query": "starboy"}),
        ("play starboy by the weekend", {"action": "play", "query": "starboy by the weekend"}),
        ("play maps by maroon 5", {"action": "play", "query": "maps by maroon 5"}),
        ("play the song circles", {"action": "play", "query": "circles"}),
        ("play the song maps by maroon five", {"action": "play", "query": "maps by maroon five"}),
        ("play mad love by Mable", {"action": "play", "query": "mad love by Mable"}),
        # "the son" is what speech-to-text makes of "the song".
        ("play the son starboy", {"action": "play", "query": "starboy"}),
        # Addressing the assistant by name before the command.
        ("Hey Luna Play believer", {"action": "play", "query": "believer"}),
        ("hey luna, play circles", {"action": "play", "query": "circles"}),
        ("luna, play maps by maroon five", {"action": "play", "query": "maps by maroon five"}),
        ("put on believer", {"action": "play", "query": "believer"}),
    ]
    for text, arguments in expected:
        planner, captured = _planner({"unexpected": True})
        calls = planner.plan(system_prompt="", user_text=text, capabilities=MUSIC_CAPABILITIES)
        assert calls == [{
            "type": "skill_tool_call", "skill_id": MUSIC_SKILL_ID,
            "tool": "opendeck_music", "arguments": arguments,
        }], text
        assert "init" not in captured, text
    # No track named (or only "this song"): which track to play is a real
    # choice, so it stays with the planner rather than being guessed at.
    for text in ("play music", "play something", "play this song", "play it"):
        assert SkillPlanner._grounded_music_call(
            MUSIC_CAPABILITIES, {"play"}, text
        ) is None, text
    # Queue wording still owns its own calls.
    assert SkillPlanner._grounded_music_call(
        MUSIC_CAPABILITIES, {"queue"}, "play the next queued track"
    )["arguments"] == {"action": "queue", "op": "next"}


def test_now_playing_and_search_wording_are_planned_by_core():
    # "what's playing" is a read of the provider's state, and "search for X" is a
    # catalog search; both are Core-owned so the model cannot plan them as a
    # play (measured live: a now-playing question failed planning outright).
    for text, arguments in [
        ("what's playing?", {"action": "now_playing"}),
        ("what is playing right now", {"action": "now_playing"}),
        ("what song is playing", {"action": "now_playing"}),
        ("what song is this", {"action": "now_playing"}),
        ("what am I listening to", {"action": "now_playing"}),
        ("search for Bohemian Rhapsody", {"action": "search", "query": "bohemian rhapsody"}),
        ("find Miles Davis on OpenDeck", {"action": "search", "query": "miles davis"}),
    ]:
        planner, captured = _planner({"unexpected": True})
        calls = planner.plan(system_prompt="", user_text=text, capabilities=MUSIC_CAPABILITIES)
        assert calls[0]["arguments"] == arguments, text
        assert "init" not in captured, text
    assert SkillPlanner._requested_tool_groups("what's playing?", MUSIC_CAPABILITIES) == {"now_playing"}
    assert SkillPlanner._requested_tool_groups("search for Bohemian Rhapsody", MUSIC_CAPABILITIES) == {"search"}


def test_queue_reorder_and_priority_wording_are_planned_by_core():
    # "play <song> next/first" prioritises that track; "move track N to M" is an
    # explicit reorder. Both are Core-owned so the model cannot mis-plan them.
    for text, arguments in [
        ("play Bohemian Rhapsody next", {"action": "queue", "op": "next_up", "query": "bohemian rhapsody"}),
        ("play Hotel California first", {"action": "queue", "op": "front", "query": "hotel california"}),
        ("play the song believer next", {"action": "queue", "op": "next_up", "query": "believer"}),
        ("play the song Believer first", {"action": "queue", "op": "front", "query": "believer"}),
        ("put on the track Believer next", {"action": "queue", "op": "next_up", "query": "believer"}),
        ("move track 3 to position 1", {"action": "queue", "op": "move", "index": 2, "to": 0}),
        ("move Hotel California to the front", {"action": "queue", "op": "front", "query": "hotel california"}),
        # "play this next" names no track: prioritise whatever is playing.
        ("play this next", {"action": "queue", "op": "next_up"}),
    ]:
        planner, captured = _planner({"unexpected": True})
        calls = planner.plan(system_prompt="", user_text=text, capabilities=MUSIC_CAPABILITIES)
        assert calls[0]["arguments"] == arguments, text
        assert "init" not in captured, text
    # "play the next queued track" is still the plain advance, not a priority.
    assert SkillPlanner._requested_tool_groups(
        "play the next queued track", MUSIC_CAPABILITIES
    ) == {"queue"}


def test_shuffle_and_repeat_wording_are_planned_by_core():
    # Shuffle and repeat are queue preferences the provider answers by op, so the
    # whole call is Core-owned: without this, "shuffle the queue" was read as an
    # add of the literal words "shuffle the queue".
    for text, arguments in [
        ("shuffle the queue", {"action": "queue", "op": "shuffle", "on": True}),
        ("shuffle my music", {"action": "queue", "op": "shuffle", "on": True}),
        ("turn shuffle off", {"action": "queue", "op": "shuffle", "on": False}),
        ("stop shuffling", {"action": "queue", "op": "shuffle", "on": False}),
        ("repeat this song", {"action": "queue", "op": "repeat", "mode": "one"}),
        ("loop this track", {"action": "queue", "op": "repeat", "mode": "one"}),
        ("repeat the queue", {"action": "queue", "op": "repeat", "mode": "all"}),
        # "stop repeating" turns the preference off, so the negation guard has to
        # read it as an action rather than as a negated request.
        ("stop repeating", {"action": "queue", "op": "repeat", "mode": "off"}),
        # No mode named: the provider cycles, which is what one word means.
        ("repeat", {"action": "queue", "op": "repeat"}),
    ]:
        planner, captured = _planner({"unexpected": True})
        calls = planner.plan(system_prompt="", user_text=text, capabilities=MUSIC_CAPABILITIES)
        assert calls == [{
            "type": "skill_tool_call", "skill_id": MUSIC_SKILL_ID,
            "tool": "opendeck_music", "arguments": arguments,
        }], text
        assert "init" not in captured, text
    assert SkillPlanner._requested_tool_groups("shuffle the queue", MUSIC_CAPABILITIES) == {"queue"}
    # A request that names a track is not a preference: those words are not all
    # preference words, so the call is left to the planner instead of being
    # guessed at as a queue op.
    assert SkillPlanner._requested_tool_groups("repeat believer", MUSIC_CAPABILITIES) is None
    assert SkillPlanner._grounded_music_call(
        MUSIC_CAPABILITIES, {"queue"}, "repeat the song that is playing"
    ) is None
    # "queue up shuffle" names a track that happens to be called Shuffle.
    queued = SkillPlanner._grounded_music_call(MUSIC_CAPABILITIES, {"queue"}, "queue up shuffle")
    assert queued["arguments"] == {"action": "queue", "op": "add", "query": "shuffle"}


def test_library_and_lyrics_wording_are_planned_by_core():
    for text, arguments in [
        ("show my liked songs", {"action": "library", "kind": "likes"}),
        ("show me my playlists", {"action": "library", "kind": "playlists"}),
        ("what have I been listening to in my history", {"action": "library", "kind": "history"}),
        ("what did I listen to yesterday", {"action": "library", "kind": "history"}),
        ("show me the lyrics", {"action": "lyrics"}),
    ]:
        planner, captured = _planner({"unexpected": True})
        calls = planner.plan(system_prompt="", user_text=text, capabilities=MUSIC_CAPABILITIES)
        assert calls[0]["arguments"] == arguments, text
        assert "init" not in captured, text
    # Lyrics for a named track carry title/artist the model fills, so this one
    # still goes through the planner.
    planner, captured = _planner(_success("do_lyrics", {"title": "Yesterday"}))
    planner.plan(system_prompt="", user_text="show me the lyrics for Yesterday", capabilities=MUSIC_CAPABILITIES)
    assert "init" in captured


def test_speaker_wording_selects_the_speaker_action_it_names():
    assert SkillPlanner._requested_tool_groups("set the volume to 30 percent", SPEAKER_CAPABILITIES) == {"volume"}
    assert SkillPlanner._requested_tool_groups("mute", SPEAKER_CAPABILITIES) == {"mute"}
    assert SkillPlanner._requested_tool_groups("unmute the speaker", SPEAKER_CAPABILITIES) == {"unmute"}
    assert SkillPlanner._requested_tool_groups("say good morning", SPEAKER_CAPABILITIES) == {"say"}
    # "stop speaking" is the action, not a negated plan.
    assert SkillPlanner._requested_tool_groups("stop speaking", SPEAKER_CAPABILITIES) == {"stop_speaking"}
    # A genuine negation still fails closed.
    with pytest.raises(SkillPlannerError, match="Negated"):
        SkillPlanner._requested_tool_groups("don't turn the ring off", SPEAKER_CAPABILITIES)


def test_grounded_speaker_retry_emits_zero_argument_and_speech_calls():
    planner, captured = _planner({"unexpected": True})
    calls = planner.plan(
        system_prompt="", user_text="stop speaking",
        capabilities=SPEAKER_CAPABILITIES, force_grounded=True,
    )
    assert calls[0]["arguments"] == {"action": "stop_speaking"}
    assert "init" not in captured
    planner, captured = _planner({"unexpected": True})
    calls = planner.plan(
        system_prompt="", user_text="say good morning everyone",
        capabilities=SPEAKER_CAPABILITIES, force_grounded=True,
    )
    # The words are the user's own, taken verbatim after the verb.
    assert calls[0]["arguments"] == {"action": "say", "text": "good morning everyone"}
    assert "init" not in captured


def test_grounded_volume_retry_emits_the_core_owned_level():
    planner, captured = _planner({"unexpected": True})
    calls = planner.plan(
        system_prompt="", user_text="set the volume to 30 percent",
        capabilities=SPEAKER_CAPABILITIES, force_grounded=True,
        volume_grounding={"volume": 0.3},
    )
    assert calls[0]["arguments"] == {"action": "volume", "volume": 0.3}
    assert "init" not in captured
    # A level stated without the word "percent" ("set volume to 55") grounds
    # the same Core-owned fraction; measured live, it previously failed
    # planning outright because only the percent sign was recognised.
    planner, captured = _planner({"unexpected": True})
    calls = planner.plan(
        system_prompt="", user_text="set volume to 55",
        capabilities=SPEAKER_CAPABILITIES, force_grounded=True,
        volume_grounding={"volume": 0.55},
    )
    assert calls[0]["arguments"] == {"action": "volume", "volume": 0.55}
    assert "init" not in captured


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


def test_library_writes_are_planned_by_core_without_calling_the_model():
    # A like or a playlist add changes the user's own account, so the phrase is
    # read exactly: the track is the one the provider last handed to the speaker
    # ("this song" names no id at all) and the playlist is the user's own words.
    expected = [
        ("like this song", {"action": "save", "kind": "like"}),
        ("like this track please", {"action": "save", "kind": "like"}),
        ("can you like this song", {"action": "save", "kind": "like"}),
        ("add this song to my liked music", {"action": "save", "kind": "like"}),
        ("add this to my liked songs", {"action": "save", "kind": "like"}),
        ("save this song to my favourites", {"action": "save", "kind": "like"}),
        ("unlike this song", {"action": "save", "kind": "unlike"}),
        ("remove this song from my liked music", {"action": "save", "kind": "unlike"}),
        ("add this song to my Late Night playlist", {"action": "save", "kind": "playlist_add", "name": "Late Night"}),
        ("add this to my Hindi Songs playlist", {"action": "save", "kind": "playlist_add", "name": "Hindi Songs"}),
        ("put this track in my Focus playlist", {"action": "save", "kind": "playlist_add", "name": "Focus"}),
        ("create a playlist called Road Trip", {"action": "save", "kind": "playlist_create", "name": "Road Trip"}),
        ("make me a playlist named Sunday Morning", {"action": "save", "kind": "playlist_create", "name": "Sunday Morning"}),
        # The skill prompt cleanser can hand the planner its own addressing back
        # ("OpenDeck and add ...") and can say "your" for the user's own library:
        # neither is part of the command, so both still ground to the same call.
        ("add this song to your Hindi Songs playlist", {"action": "save", "kind": "playlist_add", "name": "Hindi Songs"}),
        ("OpenDeck and add this song to my Focus playlist", {"action": "save", "kind": "playlist_add", "name": "Focus"}),
        ("openDeck, add this song to my Focus playlist", {"action": "save", "kind": "playlist_add", "name": "Focus"}),
        ("OpenDeck and like this song", {"action": "save", "kind": "like"}),
        ("remove this track from your liked music", {"action": "save", "kind": "unlike"}),
    ]
    for text, arguments in expected:
        assert SkillPlanner._requested_tool_groups(text, MUSIC_CAPABILITIES) == {"save"}, text
        assert SkillPlanner._grounded_music_call(MUSIC_CAPABILITIES, {"save"}, text) == {
            "type": "skill_tool_call",
            "skill_id": MUSIC_SKILL_ID,
            "tool": "opendeck_music",
            "arguments": arguments,
        }, text


def test_library_questions_and_queue_wording_are_not_read_as_writes():
    # A read of the liked songs is a listing, not a write.
    assert SkillPlanner._requested_tool_groups("show my liked songs", MUSIC_CAPABILITIES) == {"library"}
    assert SkillPlanner._requested_tool_groups("what's in my liked music", MUSIC_CAPABILITIES) == {"library"}
    # Queueing keeps its own wording even though it says "add ... to".
    assert SkillPlanner._requested_tool_groups("add Bohemian Rhapsody to the queue", MUSIC_CAPABILITIES) == {"queue"}
    # An opinion is not a command, and neither is a non-music list.
    assert "save" not in (SkillPlanner._requested_tool_groups("I like this song", MUSIC_CAPABILITIES) or set())
    assert "save" not in (SkillPlanner._requested_tool_groups("add this to my shopping list", MUSIC_CAPABILITIES) or set())
    # A song other than the one playing is the planner's job: it has to search.
    assert "save" not in (SkillPlanner._requested_tool_groups("add Believer to my Late Night playlist", MUSIC_CAPABILITIES) or set())
