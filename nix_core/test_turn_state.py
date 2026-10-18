"""The turn is predicted before a pipeline is chosen.

These tests are about the property that matters to a person using Luna: a
turn that is just talk is answered as talk, and a turn that is an instruction
still reaches the machinery that can carry it out.
"""
from __future__ import annotations

import turn_state
from turn_state import (
    CONVERSATION,
    CORRECTION,
    INSTRUCTION,
    QUESTION,
    TurnExpectation,
    continuation_topics,
    predict_turn,
    prediction_error,
    topic_terms,
)


def test_small_talk_is_conversation_and_cheap():
    for line in (
        "Hey",
        "thanks, that actually helped",
        "I had a rough day today",
        "I'm exhausted",
        "haha that's funny",
    ):
        expectation = predict_turn(line)
        # Small talk is conversation; a declarative remark is a statement that
        # requires no action. Both are answered without retrieval or planning.
        assert expectation.read in (CONVERSATION, turn_state.STATEMENT), line
        assert expectation.statement_action == turn_state.STATEMENT_NONE, line
        assert expectation.conversational, line
        assert expectation.salience < 0.45, line


def test_a_question_is_answered_but_does_not_become_an_instruction():
    expectation = predict_turn("What's the capital of Australia?")
    assert expectation.read == QUESTION
    assert expectation.salience < 0.45
    assert not prediction_error(None, expectation)


def test_an_instruction_earns_deliberation():
    expectation = predict_turn("Turn on the kitchen light")
    assert expectation.read == INSTRUCTION
    assert expectation.wants_deliberation
    assert expectation.salience >= 0.6


def test_a_bulk_deletion_is_the_most_deliberate_kind_of_turn():
    expectation = predict_turn("Delete all my reminders")
    assert expectation.read == INSTRUCTION
    assert expectation.salience >= 0.9
    assert "consequential_scope" in expectation.reasons


def test_a_calendar_statement_is_its_own_intent_not_conversation():
    expectation = predict_turn("alright bro, I have a TSA meeting tmr from 4pm to 6pm")
    assert expectation.read == turn_state.STATEMENT
    assert expectation.statement_action == turn_state.STATEMENT_MEMORY_WRITE
    assert expectation.conversational is False
    assert "schedulable_statement" in expectation.reasons


def test_a_memory_statement_is_not_an_instruction():
    expectation = predict_turn("My sister's name is Priya")
    assert expectation.read == turn_state.STATEMENT
    assert expectation.statement_action == turn_state.STATEMENT_MEMORY_WRITE
    assert expectation.wants_deliberation
    imperative = predict_turn("Remember that my locker code is 4471")
    assert imperative.read == INSTRUCTION


def test_the_three_statement_outcomes_are_decided_separately_from_effort():
    """A statement is not one thing (review item 2).

    Ordinary statement: answered, no retrieval, no planning.
    Memory statement: persisted.
    Missing value: one question.
    None of the three is decided by salience.
    """
    ordinary = predict_turn("I started a pottery class last week")
    assert ordinary.read == turn_state.STATEMENT
    assert ordinary.statement_action == turn_state.STATEMENT_NONE
    assert ordinary.conversational is True
    assert ordinary.wants_deliberation is False
    assert ordinary.salience < 0.45

    memory = predict_turn("I have a TSA meeting tomorrow from 4pm to 6pm")
    assert memory.read == turn_state.STATEMENT
    assert memory.statement_action == turn_state.STATEMENT_MEMORY_WRITE
    assert memory.wants_deliberation is True

    clarify = predict_turn("My sister's name is")
    assert clarify.read == turn_state.STATEMENT
    assert clarify.statement_action == turn_state.STATEMENT_CLARIFY
    assert clarify.conversational is False

    # The same intent at opposite effort scores keeps the same action: effort
    # is not the decision.
    assert TurnExpectation(
        read=turn_state.STATEMENT, statement_action=turn_state.STATEMENT_NONE, salience=0.95
    ).conversational is True
    assert TurnExpectation(
        read=turn_state.STATEMENT, statement_action=turn_state.STATEMENT_MEMORY_WRITE, salience=0.05
    ).wants_deliberation is True


def test_an_ordinary_statement_never_reaches_retrieval_or_planning():
    import brain as brain_module

    class Chat:
        model = "test-chat"

        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return "A pottery class sounds lovely."

    class Knowledge:
        def __init__(self):
            self.processed = []

        def process(self, text):
            self.processed.append(text)
            return {"result": {"ok": True, "operation": "READ", "facts": []}}

        def classify(self, text):
            raise AssertionError("an ordinary statement must not be classified")

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

    chat = Chat()
    knowledge = Knowledge()
    core = brain_module.Brain(
        knowledge=knowledge, actions=Actions(), ollama=chat, log_requests=False
    )
    response = core.handle(text="I started a pottery class last week")

    assert response["route"] == "chat"
    assert knowledge.processed == []
    assert response["details"]["turn_state"]["statement_action"] == "none"
    assert len(chat.calls) == 1


def test_salience_does_not_decide_intent():
    """The review's first change: effort must not gate the request machinery.

    A low-salience question may still need a tool, and a high-salience remark
    is still a remark. Intent does the routing; salience only sets the budget.
    """
    question = predict_turn("What's the weather in Seattle?")
    assert question.read == QUESTION
    assert question.salience < 0.45
    assert question.conversational is False

    remark = TurnExpectation(read=CONVERSATION, salience=0.95)
    assert remark.conversational is True


def test_a_correction_is_a_correction_not_a_new_request():
    expectation = predict_turn("No, I meant the other one")
    assert expectation.read == CORRECTION
    assert expectation.wants_deliberation
    assert prediction_error(TurnExpectation(read=INSTRUCTION), expectation) == 0.9


def test_a_short_follow_up_inherits_the_topic():
    history = [
        {"role": "user", "content": "How is my mother doing?"},
        {"role": "assistant", "content": "She's feeling better."},
    ]
    expectation = predict_turn("and my father?", history=history)
    assert "mother" in expectation.about or "father" in expectation.about
    assert expectation.read in (CONVERSATION, QUESTION)


def test_a_topic_shift_is_a_boundary_not_a_request():
    history = [
        {"role": "user", "content": "How is my mother doing?"},
        {"role": "assistant", "content": "She's feeling better."},
    ]
    expectation = predict_turn("I started a pottery class", history=history)
    assert expectation.read == turn_state.STATEMENT
    assert expectation.statement_action == turn_state.STATEMENT_NONE
    assert expectation.conversational
    assert expectation.continues_topic is False
    assert "topic_shift" in expectation.reasons


def test_a_continuation_is_marked_as_continuing():
    history = [
        {"role": "user", "content": "How is my mother doing?"},
        {"role": "assistant", "content": "She's feeling better."},
    ]
    expectation = predict_turn("and my father?", history=history)
    assert expectation.continues_topic is True


def test_a_correction_does_not_become_a_new_memory_or_request():
    """The correction is answered from the conversation, not stored."""
    import brain as brain_module

    class Chat:
        model = "test-chat"

        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return "Ah, your father — sorry."

    class Knowledge:
        def __init__(self):
            self.processed = []

        def process(self, text):
            self.processed.append(text)
            return {"result": {"ok": True, "operation": "READ", "facts": []}}

        def classify(self, text):
            raise AssertionError("a correction must not be classified as a request")

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

    chat = Chat()
    knowledge = Knowledge()
    core = brain_module.Brain(
        knowledge=knowledge,
        actions=Actions(),
        ollama=chat,
        log_requests=False,
    )
    core._last_expectations["correction-thread"] = TurnExpectation(read=QUESTION)
    response = core.handle(
        text="No, I meant my father",
        conversation_id="correction-thread",
        session_context=[
            {"role": "user", "content": "How is my mother doing?"},
            {"role": "assistant", "content": "She's feeling better."},
        ],
    )

    assert response["route"] == "chat"
    assert knowledge.processed == []
    assert response["details"]["turn_state"]["read"] == "correction"
    assert response["details"]["turn_state"]["prediction_error"] == 0.9


def test_a_statement_with_a_missing_value_asks_once_and_stores_nothing():
    import brain as brain_module

    class NoChat:
        model = "test-chat"

        def chat(self, **kwargs):
            raise AssertionError("a missing-value statement must not reach the model")

    class Knowledge:
        def __init__(self):
            self.processed = []

        def process(self, text):
            self.processed.append(text)
            return {"result": {"ok": True, "operation": "READ", "facts": []}}

        def classify(self, text):
            raise AssertionError("a missing-value statement must not be classified")

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

    knowledge = Knowledge()
    core = brain_module.Brain(
        knowledge=knowledge, actions=Actions(), ollama=NoChat(), log_requests=False
    )
    response = core.handle(text="My sister's name is")

    assert response["rule"] == "statement_clarification"
    assert response["reply"].endswith("?")
    assert knowledge.processed == []
    assert response["details"]["turn_state"]["statement_action"] == "clarify"


def test_topic_terms_ignore_filler():
    assert topic_terms("How is my mother doing?") == ["mother"]
    assert "doing" not in topic_terms("How is my mother doing?")


def test_continuation_topics_use_recent_user_turns():
    history = [
        {"role": "user", "content": "My sister Maanvi is visiting"},
        {"role": "assistant", "content": "That sounds nice."},
    ]
    assert "maanvi" in continuation_topics("what about her", history)


def test_an_answer_to_a_pending_question_is_an_answer():
    expectation = predict_turn(
        "Maanvi",
        pending_clarification={"question": "Which sister do you mean?"},
    )
    assert expectation.read == turn_state.ANSWER
    assert expectation.wants_deliberation


def test_a_conversational_turn_does_not_reach_the_knowledge_pipeline():
    """The pipeline decision, not just the prediction.

    A small talk turn whose words happen to contain a memory-ish noun must
    still be chat. The classifier is never consulted for it.
    """
    import brain as brain_module

    class Chat:
        model = "test-chat"

        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return "Yeah. Long day?"

    class Knowledge:
        def __init__(self):
            self.processed = []

        def process(self, text):
            self.processed.append(text)
            return {"result": {"ok": True, "operation": "READ", "facts": []}}

        def classify(self, text):
            raise AssertionError("conversation must not reach the classifier")

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

    chat = Chat()
    knowledge = Knowledge()
    core = brain_module.Brain(
        knowledge=knowledge,
        actions=Actions(),
        ollama=chat,
        log_requests=False,
    )
    response = core.handle(text="I had a rough day at work")

    assert response["route"] == "chat"
    assert knowledge.processed == []
    turn = response["details"]["turn_state"]
    assert turn["conversational"] is True
    assert turn["read"] in ("conversation", "statement")
    assert turn["statement_action"] == "none"
    assert len(chat.calls) == 1
