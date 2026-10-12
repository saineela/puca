from __future__ import annotations

from brain import Brain


class Knowledge:
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


class Actions:
    def log_turn(self, **kwargs):
        pass

    def context(self, limit=20):
        return []


class Composer:
    model = "casper-test"

    def __init__(self, reply="Casper's final reply."):
        self.reply = reply
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _brain(payload, composer):
    return Brain(
        knowledge=Knowledge(payload),
        actions=Actions(),
        ollama=composer,
        log_requests=False,
    )


def test_knowledge_result_is_always_composed_by_core_with_all_grounding_inputs():
    payload = {
        "result": {
            "ok": True,
            "operation": "STORE_STATE",
            "about": "Jane",
            "state": "feeling better",
            "valence": "good",
        },
        "function": {"name": "create_state", "arguments": {"statement": "Jane is better"}},
    }
    composer = Composer("Casper confirms Jane is feeling better.")
    brain = _brain(payload, composer)

    response = brain.handle(
        text="Jane is feeling better now",
        conversation_id="voice-42",
        session_context=[
            {"role": "user", "content": "Jane was sick yesterday"},
            {"role": "assistant", "content": "I remember."},
        ],
    )

    assert response["reply"] == "Casper confirms Jane is feeling better."
    assert len(composer.calls) == 1
    call = composer.calls[0]
    assert call["think"] is False
    assert call["history"] == [
        {"role": "user", "content": "Jane was sick yesterday"},
        {"role": "assistant", "content": "I remember."},
    ]
    prompt = call["user_text"]
    assert "RAW USER REQUEST" in prompt
    assert "Jane is feeling better now" in prompt
    assert "RECENT CONVERSATION CONTEXT" in prompt
    assert "Jane was sick yesterday" in prompt
    assert "AUTHORITATIVE KNOWLEDGE RESULT" in prompt
    assert "feeling better" in prompt
    assert response["details"]["core_formatter"]["think"] is False
    assert response["details"]["formatter_ms"] >= 0
    assert response["details"]["core_formatter"]["latency_ms"] == response["details"]["formatter_ms"]


def test_event_composition_receives_authoritative_clock_and_absolute_event_time():
    payload = {
        "result": {
            "ok": True,
            "operation": "FIND",
            "count": 1,
            "temporal_context": {
                "timezone": "America/Chicago",
                "current_date": "2026-09-20",
                "current_time": "08:00:00",
                "current_datetime_display": "Sunday, September 20, 2026 at 8:00 AM",
                "today": "2026-09-20",
                "tomorrow": "2026-09-21",
                "weekday": "Sunday",
            },
            "events": [{
                "title": "Robotics practice",
                "temporal_expression": "tmr morning",
                "when": "tomorrow 9am-11am",
                "start": "2026-09-21T09:00:00-05:00",
                "end": "2026-09-21T11:00:00-05:00",
                "timezone": "America/Chicago",
            }],
        }
    }
    composer = Composer("Robotics practice is tomorrow from 9am to 11am.")
    brain = _brain(payload, composer)

    response = brain.handle(text="what do I have tmr morning")

    prompt = composer.calls[0]["user_text"]
    assert "TEMPORAL GROUNDING" in prompt
    assert "today: 2026-09-20" in prompt
    assert "tomorrow / tmr means: 2026-09-21" in prompt
    assert "2026-09-21T09:00:00-05:00" in prompt
    assert "2026-09-21T11:00:00-05:00" in prompt
    assert response["details"]["core_formatter"]["think"] is False
    assert "PUCA_POLICY_V4" in composer.calls[0]["system_prompt"]
    assert "response_act=answer" in composer.calls[0]["system_prompt"]


def test_knowledge_api_failure_also_crosses_core_composer():
    class BrokenKnowledge(Knowledge):
        def process(self, text):
            raise RuntimeError("offline")

    composer = Composer("Casper says the knowledge service is offline while remembering my sister Jane is sick.")
    brain = Brain(
        knowledge=BrokenKnowledge({}),
        actions=Actions(),
        ollama=composer,
        log_requests=False,
    )
    response = brain.handle(
        text="remember my sister Jane is sick",
        session_context=[{"role": "user", "content": "We were discussing Jane"}],
    )

    assert response["route"] == "knowledge"
    assert response["reply"] == "Casper says the knowledge service is offline while remembering my sister Jane is sick."
    assert composer.calls[0]["think"] is False
    assert "remember my sister Jane is sick" in composer.calls[0]["user_text"]
    assert "KNOWLEDGE_UNAVAILABLE" in composer.calls[0]["user_text"]


def test_multi_clause_knowledge_clause_uses_original_request_and_context():
    payload = {
        "result": {
            "ok": True,
            "operation": "CREATE",
            "record_type": "fact",
            "data": {"value": "I like tea"},
        }
    }
    composer = Composer("Done.")
    brain = _brain(payload, composer)
    response = brain.handle(
        text="remember that I like tea and tell me a joke",
        session_context=[{"role": "user", "content": "Earlier context"}],
    )

    assert response["route"] == "knowledge+chat"
    # Fact writes use the exact deterministic stored-value response; only the
    # independent joke clause needs Casper generation.
    assert len(composer.calls) == 1
    chat_call = composer.calls[0]
    assert chat_call["think"] is False
    assert chat_call["user_text"] == "tell me a joke"
    assert response["details"]["routing_engine"]["latency_ms"] >= 0


def test_legacy_fact_payload_skips_casper_and_reports_zero_formatter_time():
    # Older Knowledge API versions omitted operation/record_type. Core must
    # still recognize the create_fact shape and avoid a full Casper call.
    payload = {
        "result": {
            "ok": True,
            "record_id": 42,
            "data": {"value": "I like tea"},
        }
    }
    composer = Composer("This should never be used.")
    brain = _brain(payload, composer)

    response = brain.handle(text="remember that I like tea")

    assert composer.calls == []
    assert response["details"]["core_formatter"]["attempted"] is False
    assert response["details"]["formatter_ms"] == 0.0
    assert response["details"]["routing_engine"]["latency_ms"] >= 0


def test_multi_clause_pipeline_has_real_routing_metadata():
    payload = {
        "result": {
            "ok": True,
            "operation": "CREATE",
            "record_type": "fact",
            "data": {"value": "I like tea"},
        }
    }
    brain = _brain(payload, Composer("A joke."))

    response = brain.handle(text="remember that I like tea and tell me a joke")
    routing = response["details"]["routing_engine"]

    assert routing["rule"] == "multi_clause"
    assert routing["latency_ms"] >= 0
    assert len(routing["clauses"]) == 2


def test_restore_grounding_casing_undoes_model_shouting():
    from brain import _restore_grounding_casing

    # Model SHOUTS an entity recalled from facts; grounding spelling wins.
    text = "Your coworker DINESH is afraid of HEIGHTS."
    grounding = ["my coworker dinesh is afraid of heights", "coworker Dinesh is afraid of heights"]
    assert _restore_grounding_casing(text, grounding) == (
        "Your coworker Dinesh is afraid of heights."
    )

    # Tokens absent from grounding are invented and must be untouched.
    assert _restore_grounding_casing("Weird CHATTER here", ["some fact"]) == (
        "Weird CHATTER here"
    )

    # Plain replies pass through unchanged.
    assert _restore_grounding_casing("All calm.", ["fact"]) == "All calm."


def test_stated_times_gate_rejects_invented_and_garbled_times():
    from brain import _stated_times_are_grounded

    ground = [
        "Scheduled 'dinner' with sister Emma tuesday 7pm. "
        "Action set: reminder fires Oct 13, 2026 at 07:00 PM.",
        "start=2026-10-13T19:00:00-05:00 "
        "(Tuesday, October 13, 2026 at 7:00:00 PM)",
    ]
    # Equivalent renderings of the authoritative time all pass.
    assert _stated_times_are_grounded("Dinner is set for Tuesday at 7pm.", ground)
    assert _stated_times_are_grounded("Dinner is set for Tuesday 19:00.", ground)
    # Hallucinated clock times fail (harness saw 19:30 / 19:59 for 19:00).
    assert not _stated_times_are_grounded("Dinner is set for Tuesday 19:30.", ground)
    assert not _stated_times_are_grounded("reminder fires 19:59", ground)
    # Tokenizer-garbled ISO timestamps never reach the user.
    assert not _stated_times_are_grounded(
        "fires 202 6-10 - 08 T 15: 00: 00 - 05: 00", ground
    )
    # No authoritative clock time -> nothing to contradict.
    assert _stated_times_are_grounded("see you at 9", ["no times here"])
    # am/pm-less bare numbers (dates like "13th", "2026") are not clock times.
    assert _stated_times_are_grounded(
        "Scheduled on the 13th of October 2026.", ground
    )


def test_format_knowledge_result_renders_fire_time_readably():
    from brain import format_knowledge_result

    payload = {
        "result": {
            "ok": True,
            "operation": "CREATE",
            "record_type": "event",
            "data": {"title": "dinner"},
            "temporal": {"when": "tuesday 7pm"},
            "actions": {
                "ok": True,
                "action_type": "reminder",
                "first_fire_at": "2026-10-13T19:00:00-05:00",
            },
        }
    }
    reply = format_knowledge_result(payload)
    # Human-readable local rendering instead of raw ISO the model garbles.
    assert "Oct 13, 2026 at 7:00 PM" in reply
    assert "19:00:00-05:00" not in reply


def test_ground_reply_clock_fixes_invented_wall_clock():
    from datetime import datetime

    from brain import _ground_reply_clock

    # Wrong stated time is replaced with the real local clock.
    fixed = _ground_reply_clock("what time is it right now?", "It's currently 10:45 AM.")
    assert "10:45" not in fixed
    assert "currently" in fixed
    # Correct (same minute) time passes through untouched.
    now = datetime.now().strftime("%I:%M %p").lstrip("0").lower()
    kept = _ground_reply_clock("what time is it?", f"It's currently {now}.")
    assert kept == f"It's currently {now}."
    # No time stated on a direct clock question: deterministic answer.
    answered = _ground_reply_clock("what's the time?", "Sure thing!")
    assert answered.startswith("It's currently ")
    # Non-clock questions and Knowledge lookups are never touched.
    assert _ground_reply_clock("how are you?", "It's currently 10:45 AM.") == (
        "It's currently 10:45 AM."
    )
    assert _ground_reply_clock(
        "what time is my meeting tomorrow?", "Your meeting is at 3pm."
    ) == "Your meeting is at 3pm."


def test_keys_reply_single_key_has_no_space_before_period():
    from brain import _keys_reply

    reply = _keys_reply(["user's coworker is named Dinesh"])
    assert reply.endswith(".")
    assert " ." not in reply


def test_humanize_worker_error_translates_network_dumps():
    from skill_runtime import _humanize_worker_error

    raw = (
        "SocketAPIError: Error connecting to [AddrInfo(family=<AddressFamily.AF_INET: 2>, "
        "type=<SocketKind.SOCK_STREAM: 1>, proto=6, sockaddr=IPv4Sockaddr("
        "address='192.168.0.55', port=6053))]: [Errno 113] Connect call failed "
        "('192.168.0.55', 6053)"
    )
    msg = _humanize_worker_error(raw)
    assert "no route to host" in msg
    assert "192.168.0.55:6053" in msg
    assert "AddrInfo" not in msg
    # Other error classes keep their meaning in plain language.
    assert "refused the connection" in _humanize_worker_error("[Errno 111] Connection refused")
    assert "did not respond in time" in _humanize_worker_error("[Errno 110] Connection timed out")
    # Unknown errors pass through untouched.
    assert _humanize_worker_error("device said no") == "device said no"
