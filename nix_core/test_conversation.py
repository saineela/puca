"""Conversation-level Core contracts that do not require live services."""
from __future__ import annotations

import json
from random import Random

from brain import (
    Brain,
    _confirmed_rgb_offer,
    _explicit_color_followup_skill,
    _explicit_combined_color_intent,
    _explicit_named_color_intent,
    _recent_assistant_rgb,
    _rgb_matches_named_color,
    format_knowledge_result,
)
from skill_runtime import SkillRuntimeError


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
    brain = Brain(
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
    brain = Brain(
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
    brain = Brain(
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
    brain = Brain(
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
    brain = Brain(
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


class ProposalChat:
    model = "mock-skill-model"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


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
                "input_schema": {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"], "additionalProperties": False},
            }],
        }
        self.spec["live_device"] = live_device
        self.proposal_error = proposal_error
        self.execute_error = None
        self.match_result = None
        self.matches = []
        self.executions = []
        self.validations = []

    def targeted_skill_specs(self, text, history=None):
        current = text.casefold()
        if any(term in current for term in ("ring", "studio ring", self.spec["device_name"].casefold())):
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
        else:
            raise SkillRuntimeError("invalid schema")
        return {"skill_id": self.skill_id, "tool": {"name": "control_ring"}, "arguments": proposal["arguments"]}

    def execute(self, matched):
        self.executions.append(matched)
        if self.execute_error:
            raise SkillRuntimeError(self.execute_error)
        action = matched["arguments"]["action"]
        result = {
            "message": f"confirmed {action}",
            "available_effects": ["Rainbow", "Ocean Ripple"],
        }
        if action == "color":
            rgb = matched["arguments"]["rgb"]
            result["state"] = {"on": True, "rgb": rgb, "effect": "None"}
        return {"skill_id": self.skill_id, "tool": "control_ring", "result": result}


def _skill_proposal(action="catalog"):
    return "NIX_SKILL_CALL:" + json.dumps({
        "type": "skill_tool_call",
        "skill_id": ProposalSkillRuntime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": action},
    })


def test_brain_executes_only_a_valid_model_proposal_and_sends_targeted_untrusted_data():
    skill_runtime = ProposalSkillRuntime()
    chat = ProposalChat(_skill_proposal("catalog"))
    brain = Brain(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
    brain.skill_runtime = skill_runtime

    result = brain.handle(text="What animation options are available on the Studio Ring?")

    assert result["rule"] == "nix_model_skill_tool"
    assert result["details"]["model_called"] is True
    assert result["details"]["deterministic"] is False
    assert result["reply"].startswith("Firmware-reported effects:")
    assert "Rainbow" in result["reply"]
    assert len(skill_runtime.executions) == 1
    sent = json.loads(chat.calls[0]["user_text"])
    assert sent["user_request"].startswith("What animation")
    assert sent["nix_skill_capabilities_untrusted_data"][0]["device_name"] == "Studio Ring"
    assert "live_device" not in sent["nix_skill_capabilities_untrusted_data"][0]
    assert chat.calls[0]["system_prompt"].count("DEVICE/SKILL COMMANDS:") == 1
    assert "live_device.available_effects" in chat.calls[0]["system_prompt"]
    assert "NIX skill-tool request format" not in chat.calls[0]["system_prompt"]
    assert "NIX-AVAILABLE SKILL DATA" not in chat.calls[0]["system_prompt"]


def test_live_device_catalog_is_attached_to_this_turn_capability_data():
    live = {"available_effects": ["Aurora Pulse", "Ocean Ripple"], "device_connected": True}
    runtime = ProposalSkillRuntime(live_device=live)
    chat = ProposalChat(_skill_proposal("catalog"))
    brain = Brain(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
    brain.skill_runtime = runtime

    result = brain.handle(text="List the Ring Light animations")

    sent = json.loads(chat.calls[0]["user_text"])
    supplied = sent["nix_skill_capabilities_untrusted_data"][0]
    assert supplied["live_device"] == live
    assert sent["user_request"] == "List the Ring Light animations"
    assert result["reply"] == "Firmware-reported effects: Rainbow, Ocean Ripple."


def test_brain_rejects_model_proposals_without_worker_execution():
    invalid_replies = (
        _skill_proposal("on") + " done",
        "I will do it. " + _skill_proposal("on"),
        "NIX_SKILL_CALL:{\"type\":\"skill_tool_call\",\"type\":\"skill_tool_call\",\"skill_id\":\"github:example/ring-light:ring-light\",\"tool\":\"control_ring\",\"arguments\":{\"action\":\"on\"}}",
    )
    for reply in invalid_replies:
        skill_runtime = ProposalSkillRuntime()
        chat = ProposalChat(reply)
        brain = Brain(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
        brain.skill_runtime = skill_runtime
        response = brain.handle(text="Turn the Studio Ring on")
        assert response["rule"] == "nix_model_skill_tool_rejected"
        assert skill_runtime.executions == []

    for proposal_error in ("not trusted", "undeclared tool", "invalid schema"):
        skill_runtime = ProposalSkillRuntime(proposal_error=proposal_error)
        brain = Brain(
            knowledge=FakeKnowledge({}), actions=FakeActions(),
            ollama=ProposalChat(_skill_proposal("on")), log_requests=False,
        )
        brain.skill_runtime = skill_runtime
        response = brain.handle(text="Turn the Studio Ring on")
        assert response["rule"] == "nix_model_skill_tool_rejected"
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
            + "NIX_SKILL_CALL:"
            + rng.choice(whitespace)
            + encoded
            + rng.choice(whitespace)
        )
        runtime = ProposalSkillRuntime()
        brain = Brain(
            knowledge=FakeKnowledge({}), actions=FakeActions(),
            ollama=ProposalChat(reply), log_requests=False,
        )
        brain.skill_runtime = runtime
        result = brain.handle(text=request_text)
        assert result["rule"] == "nix_model_skill_tool"
        assert len(runtime.executions) == 1

    valid = _skill_proposal("on")
    invalid_frames = [
        lambda call: "Okay, " + call,
        lambda call: call + "\nI switched it on.",
        lambda call: "```json\n" + call + "\n```",
        lambda call: call + "\n" + call,
        lambda call: call.replace("{", "{,", 1),
    ]
    rng.shuffle(invalid_frames)
    for frame in invalid_frames:
        reply = frame(valid)
        runtime = ProposalSkillRuntime()
        brain = Brain(
            knowledge=FakeKnowledge({}), actions=FakeActions(),
            ollama=ProposalChat(reply), log_requests=False,
        )
        brain.skill_runtime = runtime
        result = brain.handle(text="Turn the Studio Ring on")
        assert result["rule"] == "nix_model_skill_tool_rejected"
        assert runtime.executions == []


def test_untrusted_targeted_skill_is_never_sent_to_model_or_worker():
    skill_runtime = ProposalSkillRuntime(runnable=False)
    chat = ProposalChat(_skill_proposal("on"))
    brain = Brain(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
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
    chat = ProposalChat("I can change it to blue, RGB(0, 0, 128).")
    brain = Brain(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
    brain.skill_runtime = skill_runtime

    mismatch = brain.handle(
        text="can you apply that as green to the bedroom light",
        session_context=[{"role": "assistant", "content": "A cool blue option is RGB(0, 0, 128)."}],
    )

    assert mismatch["rule"] != "nix_worker_skill_color_followup"
    assert skill_runtime.executions == []

    matched = brain.handle(
        text="can you set that to the bedroom light",
        session_context=[{"role": "assistant", "content": "A cool blue would work: RGB(40, 40, 100)."}],
    )
    assert matched["rule"] == "nix_worker_skill_color_followup"
    assert matched["details"]["skill_execution_confirmed"] is True
    assert skill_runtime.executions[-1]["arguments"] == {"action": "color", "rgb": [40, 40, 100]}


def test_brain_resolves_an_elliptical_color_update_from_previous_requester_turn():
    skill_runtime = ProposalSkillRuntime()
    chat = ProposalChat(_skill_proposal("color"))
    brain = Brain(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
    brain.skill_runtime = skill_runtime

    response = brain.handle(
        text="change it to blue",
        session_context=[
            {"role": "user", "content": "Set the Studio Ring to red"},
            {"role": "assistant", "content": "I can help with that."},
        ],
    )

    sent = json.loads(chat.calls[0]["user_text"])
    assert sent["user_request"] == "change it to blue for the Studio Ring"
    assert response["rule"] == "nix_model_skill_tool_missing"
    assert skill_runtime.executions == []


def test_brain_executes_unambiguous_rgb_followup_from_immediately_previous_assistant_turn():
    skill_runtime = ProposalSkillRuntime()
    skill_runtime.spec["device_name"] = "bedroom light"
    brain = Brain(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=ProposalChat("I couldn't help with that."), log_requests=False,
    )
    brain.skill_runtime = skill_runtime

    response = brain.handle(
        text="can you set that to my bedroom light please",
        session_context=[{
            "role": "assistant",
            "content": "A soft blue would work: RGB(40, 40, 100).",
        }],
    )

    assert response["rule"] == "nix_worker_skill_color_followup"
    assert response["details"]["skill_execution_confirmed"] is True
    assert skill_runtime.validations == [{
        "type": "skill_tool_call",
        "skill_id": skill_runtime.skill_id,
        "tool": "control_ring",
        "arguments": {"action": "color", "rgb": [40, 40, 100]},
    }]
    assert skill_runtime.executions[0]["arguments"] == {"action": "color", "rgb": [40, 40, 100]}
    assert response["details"]["color_source"] == "immediately_preceding_assistant_turn"


def test_explicit_color_requests_get_validated_structured_recovery_proposals():
    color_requests = (
        "turn on the bedroom light and switch its color to green",
        "can you change ring light color to green",
    )
    for request in color_requests:
        skill_runtime = ProposalSkillRuntime()
        skill_runtime.spec["device_name"] = "bedroom light"
        wrong_reply = ProposalChat("The Dot does not publish a finite named-color list.")
        brain = Brain(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=wrong_reply, log_requests=False)
        brain.skill_runtime = skill_runtime
        rgb = [0, 128, 0]
        recovery_proposal = json.loads(_skill_proposal("color").split(":", 1)[1])
        recovery_proposal["arguments"] = {"action": "color", "rgb": rgb}
        wrong_reply.replies = [wrong_reply.reply, "NIX_SKILL_CALL:" + json.dumps(recovery_proposal)]
        wrong_reply.chat = lambda **kwargs: wrong_reply.replies.pop(0)

        result = brain.handle(text=request)

        assert result["rule"] == "nix_worker_skill_color_confirmed"
        assert result["details"]["skill_execution_confirmed"] is True
        assert skill_runtime.executions[0]["arguments"] == {"action": "color", "rgb": rgb}


def test_explicit_color_recovery_rejects_wrong_color_and_untrusted_worker():
    skill_runtime = ProposalSkillRuntime()
    skill_runtime.spec["device_name"] = "bedroom light"
    brain = Brain(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=ProposalChat("plain text"), log_requests=False)
    brain.skill_runtime = skill_runtime
    wrong = json.loads(_skill_proposal("color").split(":", 1)[1])
    wrong["arguments"] = {"action": "color", "rgb": [0, 0, 128]}
    brain.ollama.chat = lambda **kwargs: "NIX_SKILL_CALL:" + json.dumps(wrong)
    result = brain.handle(text="can you change ring light color to green")
    assert result["rule"] == "nix_model_skill_tool_missing"
    assert skill_runtime.executions == []

    unavailable = ProposalSkillRuntime(runnable=False)
    unavailable.spec["device_name"] = "bedroom light"
    denied = Brain(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=ProposalChat("no command"), log_requests=False)
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
        brain = Brain(
            knowledge=FakeKnowledge({}), actions=FakeActions(),
            ollama=ProposalChat("I didn't send a device command."), log_requests=False,
        )
        brain.skill_runtime = skill_runtime
        response = brain.handle(
            text="can you set that to my bedroom light please",
            session_context=history,
        )
        assert response["rule"] == "nix_model_skill_tool_missing"
        assert skill_runtime.executions == []


def test_missing_model_skill_call_falls_back_to_worker_for_direct_alias_command():
    skill_runtime = ProposalSkillRuntime()
    skill_runtime.spec["device_name"] = "bedroom light"
    skill_runtime.match_result = {
        "skill_id": skill_runtime.skill_id,
        "tool": {"name": "control_ring"},
        "arguments": {"action": "off"},
    }
    brain = Brain(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=ProposalChat("I couldn't help with that."), log_requests=False,
    )
    brain.skill_runtime = skill_runtime

    response = brain.handle(text="can you turn off the bedroom light")

    assert response["rule"] == "nix_worker_skill_match"
    assert skill_runtime.matches == ["can you turn off the bedroom light"]
    assert len(skill_runtime.executions) == 1
    assert skill_runtime.executions[0]["arguments"] == {"action": "off"}
    assert response["details"]["skill_execution_confirmed"] is True


def test_missing_model_skill_call_does_not_execute_when_worker_cannot_match():
    skill_runtime = ProposalSkillRuntime()
    brain = Brain(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=ProposalChat("I couldn't help with that."), log_requests=False,
    )
    brain.skill_runtime = skill_runtime

    response = brain.handle(text="Turn the Studio Ring off")

    assert response["rule"] == "nix_model_skill_tool_missing"
    assert skill_runtime.matches == ["Turn the Studio Ring off"]
    assert skill_runtime.executions == []


def test_ordinary_affirmative_does_not_execute_a_stale_or_unrelated_color_offer():
    runtime = ProposalSkillRuntime()
    runtime.spec["device_name"] = "bedroom light"
    chat = ProposalChat("Nothing to do.")
    brain = Brain(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
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
    brain = Brain(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
    brain.skill_runtime = skill_runtime

    response = brain.handle(text="What animation options are available on the Ring Light?")

    assert response["reply"] == "The ring supports several animations."
    assert response["rule"] != "nix_model_skill_tool"
    assert skill_runtime.executions == []


def test_elliptical_followup_uses_new_color_and_does_not_reuse_previous_color():
    skill_runtime = ProposalSkillRuntime()
    chat = ProposalChat(_skill_proposal("color"))
    brain = Brain(knowledge=FakeKnowledge({}), actions=FakeActions(), ollama=chat, log_requests=False)
    brain.skill_runtime = skill_runtime

    response = brain.handle(
        text="change it to blue",
        session_context=[{"role": "user", "content": "Set the Studio Ring to red"}],
    )

    sent = json.loads(chat.calls[0]["user_text"])
    assert "blue" in sent["user_request"]
    assert "Studio Ring" in sent["user_request"]
    assert "red" not in sent["user_request"]
    assert response["rule"] == "nix_model_skill_tool_missing"
    assert skill_runtime.executions == []

    blue = json.loads(_skill_proposal("color").split(":", 1)[1])
    blue["arguments"] = {"action": "color", "rgb": [0, 0, 128]}
    chat.reply = "NIX_SKILL_CALL:" + json.dumps(blue)
    response = brain.handle(
        text="change it to blue",
        session_context=[{"role": "user", "content": "Set the Studio Ring to red"}],
    )
    assert response["rule"] == "nix_worker_skill_color_confirmed"
    assert response["details"]["skill_execution_confirmed"] is True
    assert skill_runtime.executions[-1]["arguments"] == {"action": "color", "rgb": [0, 0, 128]}


def test_brain_does_not_claim_device_success_when_worker_rejects_proposal():
    skill_runtime = ProposalSkillRuntime()
    skill_runtime.execute_error = "device did not confirm"
    brain = Brain(
        knowledge=FakeKnowledge({}), actions=FakeActions(),
        ollama=ProposalChat(_skill_proposal("on")), log_requests=False,
    )
    brain.skill_runtime = skill_runtime
    response = brain.handle(text="Turn the Studio Ring on")
    assert "couldn't confirm" in response["reply"]
    assert response["details"]["model_called"] is True
    assert len(skill_runtime.executions) == 1


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
    brain = Brain(
        knowledge=knowledge,
        actions=FakeActions(),
        ollama=NoChat(),
        log_requests=False,
    )

    response = brain.handle(text="Maanvi is sick")
    assert response["reply"] == "Noted: maanvi is sick."
