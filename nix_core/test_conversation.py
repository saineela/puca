"""Conversation-level Core contracts that do not require live services."""
from __future__ import annotations

from brain import Brain, format_knowledge_result


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


def test_creator_and_identity_questions_never_fall_through_to_model():
    knowledge = FakeKnowledge({"result": {"ok": True}})
    brain = Brain(
        knowledge=knowledge,
        actions=FakeActions(),
        ollama=NoChat(),
        log_requests=False,
    )
    for question in ("Who created you Casper?", "Who are you?", "Who is Casper?"):
        response = brain.handle(text=question)
        assert "Created and Built by Sai Neela" in response["reply"] or "Casper" in response["reply"]
        assert response["details"]["model_called"] is False


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
