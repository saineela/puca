import pytest

from puca_v4_contract import build_envelope, verify_reply


def test_answer_defaults_to_no_unnecessary_question():
    envelope = build_envelope("hi")
    assert envelope.response_act == "answer"
    assert envelope.question_budget == 0
    assert "do not invent facts" in envelope.as_prompt_block()


def test_clarification_requires_question_budget():
    with pytest.raises(ValueError):
        build_envelope("Which sister?", response_act="clarify")


def test_tool_result_requires_structured_grounding():
    with pytest.raises(ValueError):
        build_envelope("remind me", response_act="tool_result")


def test_verifier_rejects_extra_tired_user_question():
    envelope = build_envelope("I am tired", response_act="brief_support", question_budget=0, emotion="tired")
    valid, failures = verify_reply(envelope, "I hear you. Do you want to talk?")
    assert valid is False
    assert "question_budget_exceeded" in failures


def test_verifier_rejects_success_claim_after_failed_action():
    envelope = build_envelope("remind me", response_act="tool_result", grounding="event")
    valid, failures = verify_reply(
        envelope,
        "Done, your reminder is scheduled.",
        knowledge_result={"result": {"ok": False, "operation": "ERROR"}},
    )
    assert valid is False
    assert "failed_operation_claimed_successfully" in failures


def test_current_state_and_dates_can_be_marked_authoritative():
    envelope = build_envelope(
        "Is my sister better now?",
        grounding="current_state",
        must_preserve=("person name", "absolute date"),
        authoritative_context="Current state: better; historical sickness is superseded.",
    )
    block = envelope.as_prompt_block()
    assert "current_state_over_history=true" in block
    assert "absolute date" in block
