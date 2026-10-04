"""Conversation-level Core contracts that do not require live services."""
from __future__ import annotations

import json
from random import Random

import pytest

from brain import (
    Brain,
    _confirmed_rgb_offer,
    _explicit_color_followup_skill,
    _explicit_combined_color_intent,
    _explicit_color_request_skill,
    _explicit_named_color_intent,
    _recent_assistant_rgb,
    _rgb_matches_named_color,
    _retry_failed_device_request,
    _unapplied_device_request,
    format_knowledge_result,
)
from skill_runtime import SkillRuntimeError
from skill_manager import SkillManager
from skill_planner import SkillPlanner


class FakeKnowledge:
    def __init__(self, payload):
        self.payload = payload
        self.processed = []

    def process(self, text):
        self.processed.append(text)
        return self.payload

    def classify(self, text):
        return "knowledge"

    def intent(self, text):
        return {"ok": False}

    def memory_block(self, current_text=None):
        return ""

    def digest(self):
        return ""

    def lookup_key_person(self, name):
        return []

    def learn_keys(self, text):
        return []


class FakeActions:
    def __init__(self):
        self.turns = []

    def log_turn(self, **kwargs):
        self.turns.append(kwargs)

    def context(self, limit=20):
        return []


class NoChat:
    model = "test"

    def chat(self, **kwargs):
        raise AssertionError("clarification must not be rewritten by the chat model")


def test_identity_fact_is_formatted_deterministically():
    payload = {
        "result": {
            "ok": True,
            "operation": "READ",
            "query": "name",
            "facts": [{"value": "user's name is Sai"}],
        }
    }
    assert format_knowledge_result(payload) == "Your name is Sai."


def test_clarification_result_is_spoken_verbatim():
    payload = {
        "result": {
            "ok": False,
            "operation": "NEEDS_CLARIFICATION",
            "question": "Which sister do you mean: jane or maanvi?",
            "candidates": ["jane", "maanvi"],
        }
    }
    assert format_knowledge_result(payload) == "Which sister do you mean: jane or maanvi?"


def test_brain_does_not_guess_ambiguous_person():
    knowledge = FakeKnowledge({
        "result": {
            "ok": False,
            "operation": "NEEDS_CLARIFICATION",
            "question": "Which sister do you mean: jane or maanvi?",
            "candidates": ["jane", "maanvi"],
        }
    })
    brain = brain_for_test(
        knowledge=knowledge,
        actions=FakeActions(),
        ollama=NoChat(),
        log_requests=False,
    )

    response = brain.handle(text="How is my sister doing?")
    assert response["route"] == "knowledge"
    assert response["reply"] == "Which sister do you mean: jane or maanvi?"
    assert knowledge.processed == ["How is my sister doing?"]


def test_clarification_answer_continues_original_state_request():
    class ClarifyingKnowledge(FakeKnowledge):
        def __init__(self):
            super().__init__({})
            self.responses = [
                {
                    "result": {
                        "ok": False,
                        "operation": "NEEDS_CLARIFICATION",
                        "role": "sister",
                        "question": "Which sister do you mean: jane or maanvi?",
                        "candidates": ["jane", "maanvi"],
                    }
                },
                {
                    "result": {
                        "ok": True,
                        "operation": "SUPERSEDE_STATE",
                        "about": "maanvi",
                        "state": "alright",
                        "valence": "good",
                        "superseded": [3],
                    }
                },
            ]

        def process(self, text):
            self.processed.append(text)
            return self.responses.pop(0)

    knowledge = ClarifyingKnowledge()
    brain = brain_for_test(
        knowledge=knowledge,
        actions=FakeActions(),
        ollama=NoChat(),
        log_requests=False,
    )
    first = brain.handle(
        text="my sister is doing alright now",
        conversation_id="voice-session-1",
    )
    assert "Which sister" in first["reply"]

    second = brain.handle(
        text="Maanvi",
        conversation_id="voice-session-1",
    )
    assert second["details"]["clarification_continuation"]["answer"] == "maanvi"
    assert knowledge.processed == [
        "my sister is doing alright now",
        "my sister maanvi is doing alright now",
    ]
    assert second["reply"] == "Updated: maanvi is now alright."


def test_turn_logging_keeps_dashboard_and_api_conversation_metadata():
    actions = FakeActions()
    brain = brain_for_test(
        knowledge=FakeKnowledge({"result": {"ok": True}}),
        actions=actions,
        ollama=NoChat(),
        log_requests=False,
    )

    brain.handle(
        text="Who are you?",
        location="openai-api",
        conversation_id="api-thread-42",
    )

    assert [turn["role"] for turn in actions.turns] == ["user", "assistant"]
    assert all(turn["refs"]["location"] == "openai-api" for turn in actions.turns)
    assert all(
        turn["refs"]["conversation_id"] == "api-thread-42"
        for turn in actions.turns
    )


def test_brain_loads_only_source_and_conversation_scoped_history():
    class ScopedActions(FakeActions):
        def __init__(self):
            super().__init__()
            self.context_calls = []

        def context_for(self, *, location, conversation_id):
            self.context_calls.append((location, conversation_id))
            return [{"role": "user", "content": "same thread history"}]

    actions = ScopedActions()
    brain = brain_for_test(
        knowledge=FakeKnowledge({"result": {"ok": True}}),
        actions=actions,
        ollama=NoChat(),
        log_requests=False,
    )

    brain.handle(
        text="Who are you?",
        location="dashboard",
        conversation_id="dashboard-thread-1",
    )

    assert actions.context_calls == [("dashboard", "dashboard-thread-1")]


def test_creator_and_identity_questions_never_fall_through_to_model():
    knowledge = FakeKnowledge({"result": {"ok": True}})
    brain = brain_for_test(
        knowledge=knowledge,
        actions=FakeActions(),
        ollama=NoChat(),
        log_requests=False,
    )
    expectations = {
        "Who created you Casper?": "NIX PUCA was created by Sai Neela",
        "Who are you?": "I'm Luna",
        "Who is Casper?": "I'm Luna",
    }
    for question, expected in expectations.items():
        response = brain.handle(text=question)
        assert expected in response["reply"]
        assert response["details"]["model_called"] is False


class ProposalPlanner:
    model = "mock-small-instruct-planner"

    def __init__(self, owner):
        self.owner = owner
        self.calls = []

    def plan(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.owner.replies.pop(0) if self.owner.replies else self.owner.reply
        reply = reply.lstrip()
        if reply.startswith("NIX_SKILL_PLAN:"):
            reply = reply[len("NIX_SKILL_PLAN:"):].strip()
        payload = json.loads(reply)
        if not isinstance(payload, dict) or payload.get("type") != "skill_tool_plan":
            raise ValueError("invalid mocked plan")
        return payload["calls"]


class ProposalChat:
    model = "mock-luna-model"

    def __init__(self, reply):
        self.reply = reply
        self.replies = None
        self.calls = []
        self.normalized_request = None
        self.skill_planner = ProposalPlanner(self)

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.loads(kwargs["user_text"])
        if "current_user_request" in payload:
            normalized = self.normalized_request or payload["current_user_request"]
            return json.dumps({"normalized_request": normalized})
        payload = json.loads(kwargs["user_text"])
        assert set(payload) == {"verified_execution_results"}
        results = payload["verified_execution_results"]
        result = results[0]["result"] if results else {}
        effects = result.get("available_effects") if isinstance(result, dict) else None
        if effects:
            return "Firmware-reported effects: " + ", ".join(effects) + "."
        state = result.get("state") if isinstance(result, dict) else None
        if isinstance(state, dict):
            return f"The ring is {'on' if state.get('on') else 'off'}; RGB {state.get('rgb')}."
        return "The requested device action was confirmed."


def brain_for_test(**kwargs):
    ollama = kwargs.get("ollama")
    planner = kwargs.pop("skill_planner", getattr(ollama, "skill_planner", None))
    return Brain(skill_planner=planner, **kwargs)


class ProposalSkillRuntime:
    skill_id = "github:example/ring-light:ring-light"

    def __init__(self, *, runnable=True, proposal_error=None, live_device=None):
        self.live_device = live_device
        self.spec = {
            "skill_id": self.skill_id,
            "name": "Ring Light",
            "device_name": "Studio Ring",
            "description": "Mock ring skill",
            "summary": "Mock tool summary",
            "triggers": ["ring", "ring light"],
            "configured": runnable,
            "trusted": runnable,
            "runnable": runnable,
            "tools": [{
                "name": "control_ring",
                "description": "Mock tool",
                "input_schema": {
                    "type": "object",
                    "oneOf": [
                        {"type": "object", "properties": {"action": {"type": "string", "enum": ["on", "off", "state", "catalog"]}}, "required": ["action"]},
                        {"type": "object", "properties": {"action": {"type": "string", "enum": ["color"]}, "rgb": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "integer", "minimum": 0, "maximum": 255}}}, "required": ["action", "rgb"]},
                        {"type": "object", "properties": {"action": {"type": "string", "enum": ["brightness"]}, "brightness": {"type": "number", "minimum": 0.05, "maximum": 1.0}}, "required": ["action", "brightness"]},
                        {"type": "object", "properties": {"action": {"type": "string", "enum": ["effect"]}, "effect": {"type": "string", "minLength": 1, "maxLength": 64}}, "required": ["action", "effect"]},
                    ],
                    "properties": {"action": {"type": "string", "enum": ["on", "off", "state", "catalog", "color", "brightness", "effect"]}, "rgb": {"type": "array"}, "brightness": {"type": "number"}, "effect": {"type": "string"}},
                    "required": ["action"],
                    "additionalProperties": False,
                },
            }],
        }
        self.spec["live_device"] = live_device
        self.proposal_error = proposal_error
        self.execute_error = None
        self.match_result = None
        self.execute_result = None
        self.execute_results = []
        self.matches = []
        self.executions = []
        self.validations = []

    def installed_skill_specs(self):
        return [self.spec]

    def targeted_skill_specs(self, text, history=None):
        current = text.casefold()
        if any(term in current for term in ("ring", "studio ring", self.spec["device_name"].casefold())):
            return [self.spec]
        if any(word in current.split() for word in ("again", "retry")) and history:
            if any(self.spec["device_name"].casefold() in str(turn.get("content") or "").casefold() for turn in history if isinstance(turn, dict)):
                return [self.spec]
        if any(word in current.split() for word in ("it", "that", "they", "them")):
            for turn in reversed(history or []):
                content = str(turn.get("content") or "").casefold()
                if "ring" in content or "studio ring" in content:
                    return [self.spec]
        return []

    def trigger_candidate(self, _text):
        return None

    def status(self, _skill_id):
        return {"runnable": self.spec["runnable"], "configured": self.spec["configured"], "trusted": self.spec["trusted"]}


    @staticmethod
    def _bounded_device_data(value):
        return value

    def proposal_context(self, skill_ids=None):
        spec = dict(self.spec)
        spec.pop("live_device", None)
        return json.dumps([spec] if self.skill_id in (skill_ids or []) else [])

    def match(self, text):
        self.matches.append(text)
        return self.match_result

    def validate_proposal(self, proposal):
        self.validations.append(proposal)
        if self.proposal_error:
            raise SkillRuntimeError(self.proposal_error)
        if proposal.get("skill_id") != self.skill_id:
            raise SkillRuntimeError("not supplied")
        if proposal.get("tool") != "control_ring":
            raise SkillRuntimeError("undeclared tool")
        arguments = proposal.get("arguments")
        if not isinstance(arguments, dict):
            raise SkillRuntimeError("invalid schema")
        if set(arguments) == {"action"}:
            pass
        elif (
            set(arguments) == {"action", "rgb"}
            and arguments.get("action") == "color"
            and isinstance(arguments.get("rgb"), list)
            and len(arguments["rgb"]) == 3
            and all(type(channel) is int and 0 <= channel <= 255 for channel in arguments["rgb"])
        ):
            pass
        elif (
            set(arguments) == {"action", "brightness"}
            and arguments.get("action") == "brightness"
            and type(arguments.get("brightness")) in (int, float)
            and 0.05 <= arguments["brightness"] <= 1.0
        ):
            pass
        elif (
            set(arguments) == {"action", "effect"}
            and arguments.get("action") == "effect"
            and self.live_device
            and arguments.get("effect") in self.live_device.get("available_effects", [])
        ):
            pass
        else:
            raise SkillRuntimeError("invalid schema")
        return {"skill_id": self.skill_id, "tool": {"name": "control_ring"}, "arguments": proposal["arguments"]}

    def execute(self, matched):
        self.executions.append(matched)
        if self.execute_error:
            raise SkillRuntimeError(self.execute_error)
        action = matched["arguments"]["action"]
        if self.execute_results:
            return self.execute_results.pop(0)
        if self.execute_result is not None:
            return self.execute_result
        result = {
            "message": f"confirmed {action}",
            "available_effects": ["Rainbow", "Ocean Ripple"],
        }
        if action == "color":
            rgb = matched["arguments"]["rgb"]
            result["state"] = {"on": True, "rgb": rgb, "brightness": 1.0, "effect": "None"}
        elif action == "on":
            result["state"] = {"on": True, "rgb": [0, 0, 0], "brightness": 1.0, "effect": "None"}
        elif action == "off":
            result["state"] = {"on": False, "rgb": [0, 0, 0], "brightness": 1.0, "effect": "None"}
        elif action == "brightness":
            result["state"] = {"on": True, "rgb": [0, 0, 0], "brightness": matched["arguments"]["brightness"], "effect": "None"}
        elif action == "effect":
            result["state"] = {"on": True, "rgb": [0, 0, 0], "brightness": 1.0, "effect": matched["arguments"]["effect"]}
        return {"skill_id": self.skill_id, "tool": "control_ring", "result": result}


def _call(action="catalog", *, skill_id=ProposalSkillRuntime.skill_id, arguments=None):
    return {
        "type": "skill_tool_call",
        "skill_id": skill_id,
        "tool": "control_ring",
        "arguments": arguments if arguments is not None else {"action": action},
    }


def _skill_plan(*calls):
    return "NIX_SKILL_PLAN:" + json.dumps({"type": "skill_tool_plan", "calls": list(calls)})


def _skill_proposal(action="catalog"):
    return _skill_plan(_call(action))


def _proposal_from_plan(encoded):
    return json.loads(encoded.split(":", 1)[1])["calls"][0]


def _needle_response_for_core(plan, *, confidence=0.9, suppressed=None, validation=None):
    """Wrap a Core-shaped proposal as Needle's compact function-call response."""
    calls = json.loads(plan.split(":", 1)[1])["calls"]
    converted = []
    for call in calls:
        arguments = dict(call["arguments"])
        action = arguments.pop("action", None)
        if call.get("tool") != "control_ring":
            name = "set_unknown"
        elif action == "color":
            name = "set_color"
        elif action in {"on", "off"}:
            name = "turn_on" if action == "on" else "turn_off"
        
        elif action == "brightness":
            name = "set_brightness"
        elif action == "effect":
            name = "set_effect"
        elif action in {"catalog", "color_catalog", "state"}:
            name = {"catalog": "list_effects", "color_catalog": "list_colors", "state": "read_state"}[action]
        else:
            raise AssertionError(f"test proposal uses unsupported action: {action}")
        converted.append({"name": name, "arguments": arguments})
    return {
        "type": "call",
        "success": True,
        "function_calls": converted,
        "suppressed_calls": suppressed or [],
        "confidence": confidence,
        "validation": validation or {"ungrounded": [], "negation": False},
    }


def _fake_needle_planner(response):
    captured = {}

    class FakeAgent:
        def complete(self, text, *, max_new_tokens):
            captured["text"] = text
            captured["max_new_tokens"] = max_new_tokens
            return response

        def close(self):
            captured["closed"] = True

    def factory(**kwargs):
        captured["init"] = kwargs
        return FakeAgent()

    return SkillPlanner(needle_factory=factory), captured


def _run_simulated_user_request(request, plan, *, runtime=None):
    """Exercise normalization, Needle adapter, Core validation, fake dispatch and readback."""
    runtime = runtime or ProposalSkillRuntime(live_device={
        "available_effects": ["Rainbow", "Ocean Ripple"], "device_connected": True,
    })
    planner, captured = _fake_needle_planner(
        _needle_response_for_core(plan),
    )
    chat = ProposalChat(plan)
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat,
        skill_planner=planner, log_requests=False,
    )
    brain.skill_runtime = runtime
    brain.skill_manager = SkillManager(runtime)
    response = brain.handle(text=request)
    return response, runtime, captured, chat


def test_brain_executes_only_a_valid_model_proposal_and_sends_targeted_untrusted_data():
    skill_runtime = ProposalSkillRuntime()
    chat = ProposalChat(_skill_proposal("catalog"))
    brain = brain_for_test(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
    brain.skill_runtime = skill_runtime

    result = brain.handle(text="What animation options are available on the Studio Ring?")

    assert result["rule"] == "nix_model_skill_tool"
    assert result["details"]["skill_planner_called"] is True
    assert result["details"]["conversation_model_called"] is True
    assert result["reply"].startswith("Firmware-reported effects:")
    assert "Rainbow" in result["reply"]
    assert len(skill_runtime.executions) == 1
    planner_call = chat.skill_planner.calls[0]
    assert planner_call["user_text"] == "What animation options are available on the Studio Ring?"
    assert "live_device" not in planner_call["capabilities"][0]
    assert "schema-declared functions" in planner_call["system_prompt"]
    assert len(chat.calls) == 2
    normalization_call = json.loads(chat.calls[0]["user_text"])
    assert set(normalization_call) == {"current_user_request", "selected_device_label"}
    assert normalization_call["current_user_request"] == "What animation options are available on the Studio Ring?"
    luna_input = json.loads(chat.calls[1]["user_text"])
    assert set(luna_input) == {"verified_execution_results"}
    assert "user_request" not in luna_input
    assert "nix_skill_capabilities_untrusted_data" not in chat.calls[1]["user_text"]
    assert chat.calls[1]["max_new_tokens"] == 96
    assert "I turn" not in chat.calls[1]["user_text"]
    assert chat is not brain.skill_planner


def test_luna_enriches_sky_color_before_needle_receives_normalized_skill_request():
    runtime = ProposalSkillRuntime()
    proposal = _proposal_from_plan(_skill_proposal("color"))
    proposal["arguments"] = {"action": "color", "rgb": [135, 206, 235]}
    chat = ProposalChat(_skill_plan(proposal))
    chat.normalized_request = "Set the Studio Ring color to sky blue"
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat,
        log_requests=False,
    )
    brain.skill_runtime = runtime

    response = brain.handle(
        text="Change the color of the Ring Light to match the color of the sky"
    )

    assert response["details"]["skill_execution_confirmed"] is True
    assert runtime.executions[-1]["arguments"] == {
        "action": "color", "rgb": [135, 206, 235],
    }
    assert len(chat.calls) == 2  # Luna normalization, then verified-result phrasing.
    luna_request = json.loads(chat.calls[0]["user_text"])
    assert luna_request["current_user_request"] == (
        "Change the color of the Ring Light to match the color of the sky"
    )
    assert "nix_skill_capabilities_untrusted_data" not in chat.calls[0]["user_text"]
    planner_request = chat.skill_planner.calls[0]["user_text"]
    assert planner_request == "Set the Studio Ring color to sky blue"
    assert "match the color of the sky" not in planner_request
    assert "user_request" not in chat.calls[1]["user_text"]


def test_live_device_catalog_is_attached_to_this_turn_capability_data():
    live = {"available_effects": ["Aurora Pulse", "Ocean Ripple"], "device_connected": True}
    runtime = ProposalSkillRuntime(live_device=live)
    chat = ProposalChat(_skill_proposal("catalog"))
    brain = brain_for_test(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
    brain.skill_runtime = runtime

    result = brain.handle(text="List the Ring Light animations")

    planner_call = chat.skill_planner.calls[0]
    supplied = planner_call["capabilities"][0]
    assert supplied["live_device"] == live
    assert planner_call["user_text"] == "List the Ring Light animations"
    assert result["reply"] == "Firmware-reported effects: Rainbow, Ocean Ripple."
    assert len(chat.calls) == 2
    assert json.loads(chat.calls[0]["user_text"])["current_user_request"] == "List the Ring Light animations"
    assert set(json.loads(chat.calls[1]["user_text"])) == {"verified_execution_results"}


def test_brain_rejects_model_proposals_without_worker_execution():
    invalid_replies = (
        _skill_proposal("on") + " done",
        "I will do it. " + _skill_proposal("on"),
        'NIX_SKILL_CALL:{"type":"skill_tool_call","skill_id":"github:example/ring-light:ring-light","tool":"control_ring","arguments":{"action":"on"}}',
    )
    for reply in invalid_replies:
        skill_runtime = ProposalSkillRuntime()
        planner = ProposalPlanner(ProposalChat(reply))
        brain = brain_for_test(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=ProposalChat("unused"), log_requests=False)
        brain.skill_planner = planner
        brain.skill_runtime = skill_runtime
        response = brain.handle(text="Turn the Studio Ring on")
        assert response["rule"] == "nix_model_skill_tool_rejected"
        assert skill_runtime.executions == []

    for proposal_error in ("not trusted", "undeclared tool", "invalid schema"):
        skill_runtime = ProposalSkillRuntime(proposal_error=proposal_error)
        chat = ProposalChat(_skill_proposal("on"))
        brain = brain_for_test(
            knowledge=FakeKnowledge({}), actions=FakeActions(),
            ollama=chat, log_requests=False,
        )
        brain.skill_runtime = skill_runtime
        response = brain.handle(text="Turn the Studio Ring on")
        assert response["rule"] == "nix_model_skill_tool_rejected"
        assert len(chat.calls) == 1  # Luna's normalizer; no result phrasing on a rejected plan.
        assert skill_runtime.executions == []


def test_brain_handles_seeded_random_device_phrasings_and_proposal_framing():
    rng = Random(0x4E4958)
    requests = [
        "turn the Studio Ring on",
        "switch the Studio Ring on please",
        "power the Studio Ring on now",
        "please enable the Studio Ring",
        "make the Studio Ring glow",
        "could you light up the Studio Ring",
    ]
    rng.shuffle(requests)
    whitespace = ["", " ", "\n", "\t\n"]
    for request_text in requests:
        proposal = json.loads(_skill_proposal("on").split(":", 1)[1])
        encoded = json.dumps(proposal, indent=rng.choice([None, 1, 2, 4]))
        reply = (
            rng.choice(whitespace)
            + "NIX_SKILL_PLAN:"
            + rng.choice(whitespace)
            + encoded
            + rng.choice(whitespace)
        )
        runtime = ProposalSkillRuntime()
        chat = ProposalChat(reply)
        brain = brain_for_test(
            knowledge=FakeKnowledge({}), actions=FakeActions(),
            ollama=chat, log_requests=False,
        )
        brain.skill_runtime = runtime
        result = brain.handle(text=request_text)
        assert result["rule"] == "nix_model_skill_tool"
        assert len(runtime.executions) == 1
        assert len(chat.skill_planner.calls) == 1
        assert len(chat.calls) == 2  # normalization plus verified-outcome phrasing

    valid = _skill_proposal("on")
    legacy_call = 'NIX_SKILL_CALL:{"type":"skill_tool_call","skill_id":"github:example/ring-light:ring-light","tool":"control_ring","arguments":{"action":"on"}}'
    invalid_frames = [
        lambda call: "Okay, " + call,
        lambda call: call + "\nI switched it on.",
        lambda call: "```json\n" + call + "\n```",
        lambda call: call + "\n" + call,
        lambda call: call.replace("{", "{,", 1),
        lambda _call: legacy_call,
    ]
    rng.shuffle(invalid_frames)
    for frame in invalid_frames:
        reply = frame(valid)
        runtime = ProposalSkillRuntime()
        brain = brain_for_test(
            knowledge=FakeKnowledge({}), actions=FakeActions(),
            ollama=ProposalChat(reply), log_requests=False,
        )
        brain.skill_runtime = runtime
        result = brain.handle(text="Turn the Studio Ring on")
        assert result["rule"] == "nix_model_skill_tool_rejected"
        assert runtime.executions == []
        assert len(brain.ollama.calls) == 1  # normalization only


def test_untrusted_targeted_skill_is_never_sent_to_model_or_worker():
    skill_runtime = ProposalSkillRuntime(runnable=False)
    chat = ProposalChat(_skill_proposal("on"))
    brain = brain_for_test(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
    brain.skill_runtime = skill_runtime

    response = brain.handle(text="Turn the Studio Ring on")

    assert response["rule"] == "trusted_skill_unavailable"
    assert response["details"]["model_called"] is False
    assert chat.calls == []
    assert skill_runtime.executions == []


def test_followup_color_uses_only_latest_explicit_assistant_rgb_and_named_target():
    history = [
        {"role": "assistant", "content": "A nice option is RGB(40, 40, 100), which feels soft and calming."},
    ]
    assert _recent_assistant_rgb(history) == [40, 40, 100]
    assert _explicit_color_followup_skill(
        "can you set that to my bedroom light please",
        [{"skill_id": "ring", "device_name": "bedroom light", "runnable": True}],
    ) == "ring"
    assert _explicit_color_followup_skill(
        "can you set that please",
        [{"skill_id": "ring", "device_name": "bedroom light", "runnable": True}],
    ) is None
    assert _explicit_color_followup_skill(
        "change it to green for the Studio Ring",
        [{"skill_id": "ring", "device_name": "Studio Ring", "runnable": True}],
    ) == "ring"
    assert _explicit_color_followup_skill(
        "can you set that to my bedroom light please",
        [
            {"skill_id": "ring", "device_name": "bedroom light", "runnable": True},
            {"skill_id": "lamp", "device_name": "living room light", "runnable": True},
        ],
    ) == "ring"
    assert _recent_assistant_rgb([
        {"role": "assistant", "content": "Try RGB(40, 40, 100)."},
        {"role": "user", "content": "Could you apply that?"},
        {"role": "assistant", "content": "Maybe RGB(1, 2, 3) or RGB(4, 5, 6)."},
    ]) is None
    assert _recent_assistant_rgb([
        {"role": "assistant", "content": "RGB(40, 40, 300) is out of range."},
    ]) is None
    alias = [{"skill_id": "ring", "device_name": "bedroom light", "runnable": True, "name": "Ring Light", "triggers": ["ring light"]}]
    assert _explicit_combined_color_intent("turn on bedroom light and switch its color to green", alias) == ("ring", "green")
    assert _explicit_named_color_intent("can you change ring light color to green", alias) == ("ring", "green")
    assert _explicit_named_color_intent("change the ring light to the color of the sun", alias) is None
    assert _rgb_matches_named_color([0, 128, 0], "green")
    assert not _rgb_matches_named_color([0, 0, 128], "green")
    assert _confirmed_rgb_offer(
        "yes please",
        [
            {"role": "user", "content": "do you mind the bedroom light to something cool for sleeping"},
            {"role": "assistant", "content": "Yes, I'll change it to a cool tone, RGB(60, 60, 180)."},
        ],
        alias,
    ) == ("ring", [60, 60, 180], "cool")
    assert _confirmed_rgb_offer(
        "yes please",
        [
            {"role": "user", "content": "do you mind the bedroom light to something cool for sleeping"},
            {"role": "assistant", "content": "RGB(60, 60, 180) is an idea."},
        ],
        alias,
    ) is None
    assert _confirmed_rgb_offer(
        "yes please",
        [
            {"role": "user", "content": "change ring light color to blue"},
            {"role": "assistant", "content": "I'll change it to blue, RGB(0, 128, 0)."},
        ],
        alias,
    ) is None


def test_brain_requires_current_color_to_match_immediately_preceding_offer():
    skill_runtime = ProposalSkillRuntime()
    skill_runtime.spec["device_name"] = "bedroom light"
    wrong = _proposal_from_plan(_skill_proposal("color"))
    wrong["arguments"] = {"action": "color", "rgb": [0, 0, 128]}
    matched_proposal = _proposal_from_plan(_skill_proposal("color"))
    matched_proposal["arguments"] = {"action": "color", "rgb": [40, 40, 100]}
    chat = ProposalChat(_skill_plan(wrong))
    chat.replies = [_skill_plan(wrong), _skill_plan(matched_proposal)]
    brain = brain_for_test(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
    brain.skill_runtime = skill_runtime

    mismatch = brain.handle(
        text="can you apply that as green to the bedroom light",
        session_context=[{"role": "assistant", "content": "A cool blue option is RGB(0, 0, 128)."}],
    )

    assert mismatch["rule"] == "nix_model_skill_tool_rejected"
    assert skill_runtime.executions == []
    # Contradictory follow-up is rejected before Luna normalization or planning.
    assert brain.ollama.calls == []
    assert brain.skill_planner.calls == []

    # The contradictory attempt was rejected before consuming the plan fixture.
    chat.replies = [_skill_plan(matched_proposal)]
    matched = brain.handle(
        text="can you set that to the bedroom light",
        session_context=[
            {"role": "user", "content": "set the bedroom light to blue"},
            {"role": "assistant", "content": "A cool blue would work: RGB(40, 40, 100)."},
        ],
    )
    assert matched["rule"] == "nix_worker_skill_color_confirmed"
    assert matched["details"]["skill_execution_confirmed"] is True
    assert skill_runtime.executions[-1]["arguments"] == {"action": "color", "rgb": [40, 40, 100]}


def test_brain_resolves_an_elliptical_color_update_from_previous_requester_turn():
    skill_runtime = ProposalSkillRuntime()
    blue = _proposal_from_plan(_skill_proposal("color"))
    blue["arguments"] = {"action": "color", "rgb": [0, 0, 128]}
    chat = ProposalChat(_skill_plan(blue))
    brain = brain_for_test(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
    brain.skill_runtime = skill_runtime

    response = brain.handle(
        text="change it to blue",
        session_context=[
            {"role": "user", "content": "Set the Studio Ring to red"},
            {"role": "assistant", "content": "I can help with that."},
        ],
    )

    sent = chat.skill_planner.calls[0]["user_text"]
    assert sent == "change it to blue for the Studio Ring"
    assert response["details"]["skill_execution_confirmed"] is True
    assert skill_runtime.executions[-1]["arguments"] == {"action": "color", "rgb": [0, 0, 128]}



def test_brain_executes_unambiguous_rgb_followup_from_immediately_previous_assistant_turn():
    skill_runtime = ProposalSkillRuntime()
    skill_runtime.spec["device_name"] = "bedroom light"
    proposal = _proposal_from_plan(_skill_proposal("color"))
    proposal["arguments"] = {"action": "color", "rgb": [40, 40, 100]}
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=ProposalChat(_skill_plan(proposal)), log_requests=False,
    )
    brain.skill_runtime = skill_runtime

    response = brain.handle(
        text="can you set that to my bedroom light please",
        session_context=[
            {"role": "user", "content": "set the bedroom light to blue"},
            {"role": "assistant", "content": "A soft blue would work: RGB(40, 40, 100)."},
        ],
    )

    assert response["rule"] == "nix_worker_skill_color_confirmed"
    assert response["details"]["skill_execution_confirmed"] is True
    assert response["details"]["conversation_model_called"] is True
    assert len(skill_runtime.validations) == 1
    assert skill_runtime.validations[0] == {
        "type": "skill_tool_call",
        "skill_id": skill_runtime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": "color", "rgb": [40, 40, 100]},
    }
    assert skill_runtime.executions[0]["arguments"] == {"action": "color", "rgb": [40, 40, 100]}
    assert len(brain.ollama.calls) == 2
    assert "RGB(40, 40, 100)" in json.loads(brain.ollama.calls[0]["user_text"])["current_user_request"]
    assert set(json.loads(brain.ollama.calls[1]["user_text"])) == {"verified_execution_results"}
    planned = brain.skill_planner.calls[0]["user_text"]
    assert "RGB(40, 40, 100)" in planned


def test_explicit_color_requests_get_validated_structured_recovery_proposals():
    color_requests = (
        "turn on the bedroom light and switch its color to green",
        "can you change ring light color to green",
    )
    for request in color_requests:
        skill_runtime = ProposalSkillRuntime()
        skill_runtime.spec["device_name"] = "bedroom light"
        rgb = [0, 128, 0]
        recovery_proposal = _proposal_from_plan(_skill_proposal("color"))
        recovery_proposal["arguments"] = {"action": "color", "rgb": rgb}
        wrong_reply = ProposalChat(_skill_plan(recovery_proposal))
        brain = brain_for_test(
            knowledge=FakeKnowledge({}), actions=FakeActions(),
            ollama=wrong_reply, log_requests=False,
        )
        brain.skill_runtime = skill_runtime
        result = brain.handle(text=request)

        assert result["rule"] == "nix_worker_skill_color_confirmed"
        assert result["details"]["skill_execution_confirmed"] is True
        assert skill_runtime.executions[0]["arguments"] == {"action": "color", "rgb": rgb}


def test_explicit_color_recovery_rejects_wrong_color_and_untrusted_worker():
    skill_runtime = ProposalSkillRuntime()
    skill_runtime.spec["device_name"] = "bedroom light"
    brain = brain_for_test(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=ProposalChat("plain text"), log_requests=False)
    brain.skill_runtime = skill_runtime
    wrong = _proposal_from_plan(_skill_proposal("color"))
    wrong["arguments"] = {"action": "color", "rgb": [0, 0, 128]}
    brain.skill_planner = ProposalPlanner(ProposalChat(_skill_plan(wrong)))
    result = brain.handle(text="can you change ring light color to green")
    assert result["rule"] == "nix_model_skill_tool_rejected"
    assert skill_runtime.executions == []

    unavailable = ProposalSkillRuntime(runnable=False)
    unavailable.spec["device_name"] = "bedroom light"
    denied = brain_for_test(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=ProposalChat("no command"), log_requests=False)
    denied.skill_runtime = unavailable
    result = denied.handle(text="can you change ring light color to green")
    assert result["rule"] == "trusted_skill_unavailable"
    assert unavailable.executions == []


def test_ambiguous_or_unbacked_color_followup_never_executes():
    for history in (
        [],
        [{"role": "assistant", "content": "Try RGB(40, 40, 100) or RGB(20, 30, 50)."}],
        [
            {"role": "assistant", "content": "Use RGB(40, 40, 100)."},
            {"role": "user", "content": "Thanks."},
            {"role": "assistant", "content": "You're welcome."},
        ],
    ):
        skill_runtime = ProposalSkillRuntime()
        skill_runtime.spec["device_name"] = "bedroom light"
        brain = brain_for_test(
            knowledge=FakeKnowledge({}), actions=FakeActions(),
            ollama=ProposalChat("I didn't send a device command."), log_requests=False,
        )
        brain.skill_runtime = skill_runtime
        response = brain.handle(
            text="can you set that to my bedroom light please",
            session_context=history,
        )
        assert response["rule"] == "nix_model_skill_tool_rejected"
        assert skill_runtime.executions == []
        assert len(brain.ollama.calls) == 1  # normalization only; unbacked color is rejected.
        assert brain.skill_planner.calls


def test_missing_skill_plan_never_falls_back_to_worker_for_direct_alias_command():
    skill_runtime = ProposalSkillRuntime()
    skill_runtime.spec["device_name"] = "bedroom light"
    skill_runtime.match_result = {
        "skill_id": skill_runtime.skill_id,
        "tool": {"name": "control_ring"},
        "arguments": {"action": "off"},
    }
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=ProposalChat("I couldn't help with that."), log_requests=False,
    )
    brain.skill_runtime = skill_runtime

    response = brain.handle(text="can you turn off the bedroom light")

    assert response["rule"] == "nix_model_skill_tool_rejected"
    assert skill_runtime.matches == []
    assert skill_runtime.executions == []
    assert response["details"]["skill_execution_confirmed"] is False
    assert response["details"]["skill_planner_called"] is True
    assert response["details"]["conversation_model_called"] is False
    assert len(brain.ollama.calls) == 1  # request normalization only


def test_missing_model_skill_call_does_not_execute_when_worker_cannot_match():
    skill_runtime = ProposalSkillRuntime()
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=ProposalChat("I couldn't help with that."), log_requests=False,
    )
    brain.skill_runtime = skill_runtime

    response = brain.handle(text="Turn the Studio Ring off")

    assert response["rule"] == "nix_model_skill_tool_rejected"
    assert skill_runtime.matches == []
    assert skill_runtime.executions == []


def test_ordinary_affirmative_does_not_execute_a_stale_or_unrelated_color_offer():
    runtime = ProposalSkillRuntime()
    runtime.spec["device_name"] = "bedroom light"
    chat = ProposalChat("Nothing to do.")
    brain = brain_for_test(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
    brain.skill_runtime = runtime
    result = brain.handle(
        text="yes please",
        session_context=[
            {"role": "user", "content": "do you mind the bedroom light to something cool"},
            {"role": "assistant", "content": "I can change it to RGB(60, 60, 180)."},
            {"role": "user", "content": "Thanks"},
            {"role": "assistant", "content": "You're welcome."},
        ],
    )
    assert result["rule"] != "nix_worker_skill_color_followup"
    assert runtime.executions == []


def test_regular_model_reply_is_preserved_when_it_is_not_a_command():

    skill_runtime = ProposalSkillRuntime()
    chat = ProposalChat("The ring supports several animations.")
    brain = brain_for_test(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
    brain.skill_runtime = skill_runtime

    response = brain.handle(text="What animation options are available on the Ring Light?")

    assert response["reply"].startswith("I didn't send a device command")
    assert len(chat.calls) == 1  # Luna normalizes the wording before structured planning.
    assert len(chat.skill_planner.calls) == 1
    assert response["rule"] == "nix_model_skill_tool_rejected"
    assert skill_runtime.executions == []


def test_elliptical_followup_uses_new_color_and_does_not_reuse_previous_color():
    skill_runtime = ProposalSkillRuntime()
    blue = _proposal_from_plan(_skill_proposal("color"))
    blue["arguments"] = {"action": "color", "rgb": [0, 0, 128]}
    chat = ProposalChat(_skill_plan(blue))
    brain = brain_for_test(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
    brain.skill_runtime = skill_runtime

    response = brain.handle(
        text="change it to blue",
        session_context=[{"role": "user", "content": "Set the Studio Ring to red"}],
    )

    sent = chat.skill_planner.calls[0]["user_text"]
    assert "blue" in sent
    assert "Studio Ring" in sent
    assert "red" not in sent
    assert response["details"]["skill_execution_confirmed"] is True
    assert skill_runtime.executions[-1]["arguments"] == {"action": "color", "rgb": [0, 0, 128]}



def test_natural_color_phrasing_family_targets_one_rgb_color_action():
    requests = (
        "change the ring light to the color of the sun",
        "make the ring light look like sunlight",
        "set Ring Light to a warm sunset tone",
        "paint the Studio Ring a moonlight shade",
        "switch the ring color to ocean blue",
        "make the ring glow like fire",
    )
    for request in requests:
        runtime = ProposalSkillRuntime()
        rgb = [255, 180, 45]
        proposal = {
            "type": "skill_tool_call",
            "skill_id": runtime.skill_id,
            "tool": "control_ring",
            "arguments": {"action": "color", "rgb": rgb},
        }
        chat = ProposalChat(_skill_plan(proposal))
        brain = brain_for_test(
            knowledge=FakeKnowledge({}), actions=FakeActions(),
            ollama=chat, log_requests=False,
        )
        brain.skill_runtime = runtime
        response = brain.handle(text=request)
        assert runtime.executions, (request, response)
        assert runtime.executions[-1]["arguments"]["action"] == "color", request
        assert response["details"].get("skill_execution_confirmed") is True, request


def test_color_of_sun_request_uses_rgb_recovery_and_matching_readback():
    runtime = ProposalSkillRuntime()
    warm_sun_rgb = [255, 190, 40]
    proposal = {
        "type": "skill_tool_call",
        "skill_id": runtime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": "color", "rgb": warm_sun_rgb},
    }
    chat = ProposalChat(_skill_plan(proposal))
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=chat, log_requests=False,
    )
    brain.skill_runtime = runtime

    response = brain.handle(text="Change the color of the Ring Light to the color of the sun.")

    assert response["rule"] == "nix_worker_skill_color_confirmed"
    assert response["details"]["skill_execution_confirmed"] is True
    assert runtime.executions[-1]["arguments"] == {"action": "color", "rgb": warm_sun_rgb}
    assert runtime.executions[-1]["arguments"]["action"] != "color_catalog"



def test_descriptive_color_request_requires_color_action_not_power_only():
    runtime = ProposalSkillRuntime()
    chat = ProposalChat(_skill_proposal("on"))
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=chat, log_requests=False,
    )
    brain.skill_runtime = runtime

    request = "Make the Studio Ring glow like the sun and turn it on."
    assert _explicit_color_request_skill(request, [runtime.spec]) == runtime.skill_id
    response = brain.handle(text=request)

    assert response["details"].get("skill_execution_confirmed") is not True
    assert runtime.executions == []
    assert "didn't send" in response["reply"]


def test_compound_device_skill_request_bypasses_clause_split_and_stays_atomic():
    runtime = ProposalSkillRuntime()
    rgb = [255, 190, 40]
    proposal = {
        "type": "skill_tool_call",
        "skill_id": runtime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": "color", "rgb": rgb},
    }
    chat = ProposalChat(_skill_plan(proposal))
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=chat, log_requests=False,
    )
    brain.skill_runtime = runtime

    response = brain.handle(
        text="Change the Ring Light to the color of the sun. After that, turn it on.",
    )

    assert response["rule"] == "nix_worker_skill_color_confirmed"
    assert response["details"]["skill_execution_confirmed"] is True
    assert len(runtime.executions) == 1
    assert runtime.executions[0]["arguments"] == {"action": "color", "rgb": rgb}
    assert len(chat.calls) == 2
    assert len(chat.skill_planner.calls) == 1


def test_compound_device_requests_with_extra_statements_keep_one_safe_device_action():
    requests = (
        "Please turn the Studio Ring on, make it the color of the sun, and tell me when it is done.",
        "Make the Ring Light look like sunlight. After that, turn it on.",
        "Change the Studio Ring to a warm sunset color, then enable it; don't start an animation.",
        "Set the ring to the color of fire and switch it on. What RGB did you use?",
    )
    for request in requests:
        runtime = ProposalSkillRuntime()
        rgb = [255, 150, 40]
        proposal = {
            "type": "skill_tool_call",
            "skill_id": runtime.skill_id,
            "tool": "control_ring",
            "arguments": {"action": "color", "rgb": rgb},
        }
        chat = ProposalChat(_skill_plan(proposal))
        brain = brain_for_test(
            knowledge=FakeKnowledge({}), actions=FakeActions(),
            ollama=chat, log_requests=False,
        )
        brain.skill_runtime = runtime
        response = brain.handle(text=request)
        assert runtime.executions, request
        assert len(runtime.executions) == 1, request
        assert runtime.executions[0]["arguments"]["action"] == "color", request
        assert runtime.executions[0]["arguments"]["rgb"] == rgb, request
        assert response["details"].get("skill_execution_confirmed") is True, request


def test_combined_device_color_requests_cover_order_and_natural_wording_variants():
    requests = (
        "turn on the Studio Ring and make it yellow",
        "make the Studio Ring yellow, then turn it on",
        "please switch the Studio Ring color to yellow and power it on",
        "could you set the color of the Studio Ring to yellow, then enable it?",
        "switch on the Studio Ring; change its colour to yellow",
        "make the Studio Ring yellow and turn it on please",
        "can you change the Studio Ring to yellow and switch it on",
    )
    for request in requests:
        runtime = ProposalSkillRuntime()
        recovery = {
            "type": "skill_tool_call",
            "skill_id": runtime.skill_id,
            "tool": "control_ring",
            "arguments": {"action": "color", "rgb": [255, 255, 0]},
        }
        chat = ProposalChat(_skill_plan(recovery))
        brain = brain_for_test(
            knowledge=FakeKnowledge({}), actions=FakeActions(),
            ollama=chat, log_requests=False,
        )
        brain.skill_runtime = runtime
        response = brain.handle(text=request)
        assert len(chat.calls) == 2, request
        assert len(chat.skill_planner.calls) == 1, request
        assert response["rule"] == "nix_worker_skill_color_confirmed", (request, response)
        assert response["details"]["skill_execution_confirmed"] is True, request
        assert runtime.executions[-1]["arguments"] == {
            "action": "color", "rgb": [255, 255, 0],
        }, request
        assert response["details"]["skill_execution_plan"]["outcomes"][0]["result"]["state"]["on"] is True


def test_turn_on_light_and_set_green_uses_one_color_action_and_confirms_power():
    runtime = ProposalSkillRuntime()
    runtime.spec["device_name"] = "light"
    proposal = _proposal_from_plan(_skill_proposal("color"))
    proposal["arguments"] = {"action": "color", "rgb": [0, 255, 0]}
    chat = ProposalChat(_skill_plan(proposal))
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat,
        log_requests=False,
    )
    brain.skill_runtime = runtime

    response = brain.handle(text="Turn on the light and set the color to green")

    assert response["details"]["skill_execution_confirmed"] is True
    assert len(runtime.executions) == 1
    assert runtime.executions[0]["arguments"] == {
        "action": "color", "rgb": [0, 255, 0],
    }
    assert response["details"]["skill_execution_plan"]["outcomes"][0]["result"]["state"]["on"] is True
    normalization = json.loads(chat.calls[0]["user_text"])
    assert normalization["current_user_request"] == "Turn on the light and set the color to green"
    planner_request = chat.skill_planner.calls[0]["user_text"]
    assert planner_request == "Turn on the light and set the color to green"

    # Color already powers the Ring Light on; a redundant on+color plan is rejected atomically.
    runtime = ProposalSkillRuntime()
    runtime.spec["device_name"] = "light"
    redundant = _call("on")
    green = _call("color", arguments={"action": "color", "rgb": [0, 255, 0]})
    redundant_chat = ProposalChat(_skill_plan(redundant, green))
    redundant_brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=redundant_chat,
        log_requests=False,
    )
    redundant_brain.skill_runtime = runtime

    rejected = redundant_brain.handle(text="Turn on the light and set the color to green")

    assert rejected["details"]["skill_execution_confirmed"] is False
    assert runtime.executions == []
    assert "didn't send" in rejected["reply"]
    assert "valid, safe plan" in rejected["reply"]


def test_compound_color_prose_then_structured_recovery_is_supported():
    runtime = ProposalSkillRuntime()
    recovery = {
        "type": "skill_tool_call",
        "skill_id": runtime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": "color", "rgb": [255, 255, 0]},
    }
    chat = ProposalChat(_skill_plan(recovery))
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=chat, log_requests=False,
    )
    brain.skill_runtime = runtime

    response = brain.handle(text="turn on the Studio Ring and change the color to yellow")

    assert response["rule"] == "nix_worker_skill_color_confirmed"
    assert response["details"]["skill_execution_confirmed"] is True
    assert len(chat.calls) == 2
    assert len(chat.skill_planner.calls) == 1
    assert runtime.executions[-1]["arguments"] == {"action": "color", "rgb": [255, 255, 0]}


def test_compound_color_accepts_direct_proposals_across_action_orderings():
    requests = (
        "turn on the Studio Ring and make it yellow",
        "make the Studio Ring yellow, then turn it on",
        "please switch the Studio Ring color to yellow and power it on",
        "make the Studio Ring yellow and turn it on please",
    )
    for request in requests:
        runtime = ProposalSkillRuntime()
        proposal = _proposal_from_plan(_skill_proposal("color"))
        proposal["arguments"] = {"action": "color", "rgb": [255, 255, 0]}
        chat = ProposalChat(_skill_plan(proposal))
        brain = brain_for_test(
            knowledge=FakeKnowledge({}), actions=FakeActions(),
            ollama=chat, log_requests=False,
        )
        brain.skill_runtime = runtime
        response = brain.handle(text=request)
        assert response["details"]["skill_execution_confirmed"] is True, request
        assert runtime.executions[-1]["arguments"] == {"action": "color", "rgb": [255, 255, 0]}


def test_compound_color_does_not_report_success_on_missing_malformed_or_wrong_proposal():
    proposals = (
        "Sure, the Studio Ring is yellow now.",
        "NIX_SKILL_PLAN:{broken json",
        "NIX_SKILL_PLAN:" + json.dumps({
            "type": "skill_tool_call",
            "skill_id": ProposalSkillRuntime.skill_id,
            "tool": "control_ring",
            "arguments": {"action": "color", "rgb": [0, 0, 128]},
        }),
    )
    for reply in proposals:
        runtime = ProposalSkillRuntime()
        brain = brain_for_test(
            knowledge=FakeKnowledge({}), actions=FakeActions(),
            ollama=ProposalChat(reply), log_requests=False,
        )
        brain.skill_runtime = runtime
        response = brain.handle(text="turn on the Studio Ring and make it yellow")
        assert response["details"].get("skill_execution_confirmed") is not True
        assert runtime.executions == []
        assert "didn't send" in response["reply"] or "couldn't confirm" in response["reply"]


def test_color_action_retry_after_declined_readback_can_complete_once():
    history = [
        {"role": "user", "content": "turn on the Studio Ring and make it yellow"},
        {"role": "assistant", "content": "The skill returned a result, but I can't say it changed."},
    ]
    proposal = _proposal_from_plan(_skill_proposal("color"))
    proposal["arguments"] = {"action": "color", "rgb": [255, 255, 0]}
    runtime = ProposalSkillRuntime()
    chat = ProposalChat(_skill_plan(proposal))
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=chat, log_requests=False,
    )
    brain.skill_runtime = runtime

    response = brain.handle(text="try again", session_context=history)

    assert response["details"]["skill_execution_confirmed"] is True
    assert len(runtime.executions) == 1
    assert runtime.executions[0]["arguments"] == {"action": "color", "rgb": [255, 255, 0]}


def test_retry_variants_require_prior_failure_and_never_treat_model_prose_as_execution():
    specs = [ProposalSkillRuntime().spec]
    failed_history = [
        {"role": "user", "content": "turn on the Studio Ring and make it yellow"},
        {"role": "assistant", "content": "I didn't send a device command because I couldn't validate it."},
    ]
    retry_phrases = (
        "try again", "please retry", "can you try that again?",
        "give it another try",        "do that again", "try one more time", "all right, please retry the command once more",

    )
    for phrase in retry_phrases:
        assert _retry_failed_device_request(phrase, failed_history, specs) == failed_history[0]["content"]
        runtime = ProposalSkillRuntime()
        brain = brain_for_test(
            knowledge=FakeKnowledge({}), actions=FakeActions(),
            ollama=ProposalChat("Okay, it's on and yellow now!"), log_requests=False,
        )
        brain.skill_runtime = runtime
        response = brain.handle(text=phrase, session_context=failed_history)
        assert response["details"].get("skill_execution_confirmed") is not True
        assert runtime.executions == []
        assert "didn't send" in response["reply"] or "couldn't" in response["reply"]

    valid = _proposal_from_plan(_skill_proposal("color"))
    valid["arguments"] = {"action": "color", "rgb": [255, 255, 0]}
    successful_retry = ProposalSkillRuntime()
    successful_retry_brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=ProposalChat(_skill_plan(valid)), log_requests=False,
    )
    successful_retry_brain.skill_runtime = successful_retry
    successful = successful_retry_brain.handle(
        text="give it another try", session_context=failed_history,
    )
    assert successful["details"]["skill_execution_confirmed"] is True
    assert successful_retry.executions[-1]["arguments"] == {"action": "color", "rgb": [255, 255, 0]}

    assert _retry_failed_device_request(
        "try again",
        [
            {"role": "user", "content": "turn on the Studio Ring"},
            {"role": "assistant", "content": "The ring is now on."},
        ],
        specs,
    ) is None
    assert _retry_failed_device_request("try again", [], specs) is None


def test_unapplied_feedback_retries_only_a_targeted_device_action():
    specs = [ProposalSkillRuntime().spec]
    history = [
        {"role": "user", "content": "turn on the Studio Ring and make it yellow"},
        {"role": "assistant", "content": "The state could not be confirmed."},
    ]
    assert _unapplied_device_request("that didn't apply", history, specs) == history[0]["content"]
    assert _unapplied_device_request(
        "that didn't apply",
        [{"role": "user", "content": "tell me about the Studio Ring"}, history[1]],
        specs,
    ) is None


def test_worker_success_message_without_matching_device_readback_is_not_confirmation():
    runtime = ProposalSkillRuntime()
    chat = ProposalChat(_skill_proposal("on"))
    runtime.execute_result = {
        "skill_id": runtime.skill_id,
        "tool": "control_ring",
        "result": {
            "message": "The Dot confirmed the ring is on.",
            "state": {"on": False, "rgb": [0, 0, 0], "brightness": 1.0, "effect": "None"},
        },
    }
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=chat, log_requests=False,
    )
    brain.skill_runtime = runtime

    response = brain.handle(text="Turn the Studio Ring on")

    assert response["rule"] == "nix_model_skill_tool_unconfirmed"
    assert response["details"]["skill_execution_confirmed"] is False
    assert len(chat.calls) == 1  # normalized before planning; not asked to phrase an unconfirmed result.
    assert "can't say it changed" in response["reply"]


def test_brain_does_not_claim_device_success_when_worker_rejects_proposal():
    skill_runtime = ProposalSkillRuntime()
    skill_runtime.execute_error = "device did not confirm"
    chat = ProposalChat(_skill_proposal("on"))
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=chat, log_requests=False,
    )
    brain.skill_runtime = skill_runtime
    response = brain.handle(text="Turn the Studio Ring on")
    assert "couldn't confirm" in response["reply"] or "can't say it changed" in response["reply"]
    assert response["details"]["model_called"] is False
    assert response["details"]["conversation_model_called"] is False
    assert len(chat.calls) == 1  # request normalization only; execution failed before result phrasing.
    assert len(skill_runtime.executions) == 1


def test_transcript_style_retry_retargets_bedroom_light_and_never_trusts_prose():
    runtime = ProposalSkillRuntime()
    runtime.spec["device_name"] = "bedroom light"
    runtime.spec["name"] = "Ring Light Package"
    prose = ProposalChat("Okay, I'll try that again. I'm turning on your bedroom light.")
    prose.replies = [
        "I didn't send a device command because the model didn't return a valid skill call.",
        "Okay, I'll try that again. I'm turning on your bedroom light.",
        _skill_proposal("on"),
    ]
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=prose, log_requests=False,
    )
    brain.skill_runtime = runtime

    first = brain.handle(text="can you turn on my  bedroom light")
    assert first["details"].get("skill_execution_confirmed") is not True
    assert runtime.executions == []

    failed_history = [
        {"role": "user", "content": "can you turn on my  bedroom light"},
        {"role": "assistant", "content": first["reply"]},
    ]
    second = brain.handle(text="try again", session_context=failed_history)
    assert second["details"].get("skill_execution_confirmed") is not True
    assert runtime.executions == []
    assert "didn't send" in second["reply"] or "couldn't" in second["reply"]

    compound_history = failed_history + [
        {"role": "user", "content": "try again"},
        {"role": "assistant", "content": "Okay, I'll try that again. I'm turning on your bedroom light."},
    ]
    third = brain.handle(
        text="nope, it didnt turn on try again",
        session_context=compound_history,
    )
    assert third["details"].get("skill_execution_confirmed") is True, third
    assert len(runtime.executions) == 1
    assert runtime.executions[0]["arguments"] == {"action": "on"}
    assert len(prose.calls) == 4  # three normalizations plus the verified result acknowledgement.
    assert len(prose.skill_planner.calls) == 3


def test_failed_compound_retry_requires_prior_matching_device_action():
    runtime = ProposalSkillRuntime()
    runtime.spec["device_name"] = "bedroom light"
    runtime.spec["name"] = "Ring Light Package"
    chat = ProposalChat(_skill_proposal("on"))
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=chat, log_requests=False,
    )
    brain.skill_runtime = runtime
    unrelated_history = [
        {"role": "user", "content": "Please explain the bedroom light."},
        {"role": "assistant", "content": "It is a ring light."},
    ]

    response = brain.handle(
        text="nope, it didnt turn on try again",
        session_context=unrelated_history,
    )

    assert response["details"].get("skill_execution_confirmed") is not True
    assert runtime.executions == []


def test_skill_manager_validates_entire_bounded_plan_before_execution_and_stops_on_failure():
    runtime = ProposalSkillRuntime()
    brain = brain_for_test(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=ProposalChat(""), log_requests=False)
    brain.skill_runtime = runtime
    brain.skill_manager = SkillManager(runtime)

    invalid_then_valid = _skill_plan(
        _call("on"),
        {"type": "skill_tool_call", "skill_id": runtime.skill_id, "tool": "not_declared", "arguments": {"action": "off"}},
    )
    calls = brain.skill_manager.parse(invalid_then_valid)
    with pytest.raises(SkillRuntimeError):
        brain.skill_manager.validate(calls, {runtime.skill_id})
    assert runtime.executions == []

    first = _call("on")
    second = _call("off")
    third = _call("catalog")
    runtime.execute_error = "mock dispatch failure"
    parsed = brain.skill_manager.parse(_skill_plan(first, second, third))
    validated = brain.skill_manager.validate(parsed, {runtime.skill_id})
    stopped = brain.skill_manager.execute(validated, confirm=lambda _matched, _outcome: True)
    assert stopped["confirmed"] is False
    assert stopped["failed_index"] == 0
    assert stopped["completed_count"] == 0
    assert stopped["attempted_count"] == 1
    assert runtime.executions == [validated[0]]


def test_structured_multi_action_plan_is_validated_before_dispatch_then_runs_in_order():
    runtime = ProposalSkillRuntime()
    second = _call("off")
    plan = _skill_plan(_call("on"), second)
    chat = ProposalChat(plan)
    brain = brain_for_test(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
    brain.skill_runtime = runtime
    brain.skill_manager = SkillManager(runtime)

    response = brain.handle(text="Turn the Studio Ring on, then turn it off")

    assert response["rule"] == "nix_model_skill_plan"
    assert response["details"]["skill_execution_confirmed"] is True
    assert [item["arguments"]["action"] for item in runtime.executions] == ["on", "off"]


def test_invalid_later_plan_action_prevents_all_dispatch():
    runtime = ProposalSkillRuntime()
    invalid_call = _call("off")
    invalid_call["tool"] = "undeclared"
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=ProposalChat(_skill_plan(_call("on"), invalid_call)), log_requests=False,
    )
    brain.skill_runtime = runtime
    brain.skill_manager = SkillManager(runtime)

    response = brain.handle(text="Turn the Studio Ring on and off")

    assert response["details"].get("skill_execution_confirmed") is not True
    assert runtime.executions == []
    assert "complete plan" in response["reply"] or "didn't send" in response["reply"]


def test_missing_plan_for_multi_action_request_never_falls_back_to_one_worker_match():
    runtime = ProposalSkillRuntime()
    runtime.spec["device_name"] = "Studio Ring"
    runtime.match_result = {
        "skill_id": runtime.skill_id,
        "tool": {"name": "control_ring"},
        "arguments": {"action": "on"},
    }
    brain = brain_for_test(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=ProposalChat("I turned it on."), log_requests=False,
    )
    brain.skill_runtime = runtime
    brain.skill_manager = SkillManager(runtime)

    response = brain.handle(text="Turn the Studio Ring on then turn it off")

    assert response["rule"] == "nix_model_skill_plan_rejected"
    assert runtime.matches == []
    assert runtime.executions == []


@pytest.mark.parametrize(("user_request", "actions"), [
    ("Turn the Studio Ring on and set brightness to 60 percent", [
        {"action": "on"}, {"action": "brightness", "brightness": 0.6},
    ]),
    ("Set the Studio Ring brightness to 40 percent and then turn the Studio Ring off", [
        {"action": "brightness", "brightness": 0.4}, {"action": "off"},
    ]),
    ("Set the Studio Ring brightness to 55 percent, then run Aurora Pulse", [
        {"action": "brightness", "brightness": 0.55}, {"action": "effect", "effect": "Aurora Pulse"},
    ]),
    ("Turn the Studio Ring off, then turn it on", [
        {"action": "off"}, {"action": "on"},
    ]),
])
def test_simulated_multi_task_user_requests_complete_needle_core_worker_flow(user_request, actions):
    runtime = ProposalSkillRuntime(live_device={
        "available_effects": ["Aurora Pulse", "Ocean Ripple"],
        "device_connected": True,
    })
    calls = [_call(arguments=arguments) for arguments in actions]
    plan = _skill_plan(*calls)
    response, runtime, captured, chat = _run_simulated_user_request(user_request, plan, runtime=runtime)

    assert response["details"]["skill_planner_called"] is True
    assert response["details"]["skill_plan_validated"] is True, response
    assert response["details"]["skill_execution_confirmed"] is True
    assert response["rule"] in {"nix_model_skill_plan", "nix_worker_skill_color_confirmed"}
    assert captured["text"] == user_request
    assert captured["closed"] is True
    assert len(runtime.validations) == len(actions)
    assert [item["arguments"] for item in runtime.validations] == actions
    assert [item["arguments"] for item in runtime.executions] == actions
    assert response["details"]["skill_execution_plan"]["attempted_count"] == len(actions)
    assert response["details"]["skill_execution_plan"]["completed_count"] == len(actions)
    assert chat.skill_planner.calls == []  # this harness sends directly through the injected Needle adapter
    assert len(chat.calls) == 2  # normalization plus verified-result wording


def test_multi_task_invalid_later_needle_proposal_blocks_all_simulated_dispatch():
    runtime = ProposalSkillRuntime()
    valid = _call("on")
    invalid = _call("off")
    invalid["tool"] = "undeclared"
    response, runtime, _captured, _chat = _run_simulated_user_request(
        "Turn the Studio Ring on and then off",
        _skill_plan(valid, invalid),
        runtime=runtime,
    )
    assert response["details"]["skill_execution_confirmed"] is False
    assert response["details"]["skill_plan_validated"] is False
    assert runtime.validations == []
    assert runtime.executions == []
    assert "didn't send" in response["reply"] or "complete plan" in response["reply"]


def test_multi_task_readback_mismatch_stops_before_later_simulated_actions():
    runtime = ProposalSkillRuntime()
    first = {
        "skill_id": runtime.skill_id,
        "tool": "control_ring",
        "result": {"state": {"on": False, "rgb": [0, 0, 0], "brightness": 1.0, "effect": "None"}},
    }
    runtime.execute_results = [first]
    response, runtime, _captured, _chat = _run_simulated_user_request(
        "Turn the Studio Ring on, then turn it off",
        _skill_plan(_call("on"), _call("off")),
        runtime=runtime,
    )
    assert response["details"]["skill_execution_confirmed"] is False
    assert response["details"]["skill_execution_plan"]["failed_index"] == 0
    assert response["details"]["skill_execution_plan"]["attempted_count"] == 1
    assert [item["arguments"]["action"] for item in runtime.executions] == ["on"]


def test_skill_plan_step_limit_and_duplicate_keys_are_rejected():
    runtime = ProposalSkillRuntime()
    brain = brain_for_test(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=ProposalChat(""), log_requests=False)
    brain.skill_runtime = runtime
    brain.skill_manager = SkillManager(runtime)
    with pytest.raises(SkillRuntimeError):
        brain.skill_manager.parse(_skill_plan(*[_call("on") for _ in range(9)]))
    with pytest.raises(ValueError):
        brain.skill_manager.parse('NIX_SKILL_PLAN:{"type":"skill_tool_plan","type":"skill_tool_plan","calls":[]}')


def test_brain_formats_selected_person_state_without_chat_guessing():
    knowledge = FakeKnowledge({
        "result": {
            "ok": True,
            "operation": "STORE_STATE",
            "about": "maanvi",
            "state": "sick",
            "valence": "bad",
        }
    })
    brain = brain_for_test(
        knowledge=knowledge,
        actions=FakeActions(),
        ollama=NoChat(),
        log_requests=False,
    )

    response = brain.handle(text="Maanvi is sick")
    assert response["reply"] == "Noted: maanvi is sick."
