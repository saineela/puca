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
