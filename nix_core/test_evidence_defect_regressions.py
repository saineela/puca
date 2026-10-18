"""Regression tests for the Core half of the D2/D4 defect fixes.

The knowledge layer has its own failing-then-passing tests
(``nix_knowledge/tests/test_evidence_defects.py``). These pin the two bits of
Core that the end-to-end observations touched:

* a turn the gate placed on the personal/action boundary must not be demoted to
  chat by the turn predictor ("The renewal is due March 4" was predicted as
  conversation, and the write never reached the store - D4 c32);
* a fact deletion is answered with a confirmation first, and only a following
  explicit "yes" carries ``confirm=true`` to the store (D4 c38).

No model is loaded: the predictor reading is computed by the deterministic
turn-state module, and the pending-deletion path is exercised with a stub
knowledge client.
"""
from __future__ import annotations

import threading

import brain
import turn_state
from routing_engine import RouteDecision, route_request


def test_a_conversational_prediction_does_not_demote_a_due_write():
    decision = route_request("The renewal is due March 4")
    assert decision.route == "knowledge", decision
    assert decision.rule == "personal_boundary", decision

    expectation = turn_state.predict_turn("The renewal is due March 4", history=[])
    assert expectation.conversational is True

    assert brain.expectation_may_redirect_to_chat(decision, expectation) is False


def test_a_stored_deadline_is_read_from_the_store_not_from_chat():
    """A due-date read is the write's other half.

    Live verification of c32 found the follow-up "When is the renewal due?"
    answered by chat as "March 4, 2027" for a fact stored without a year. The
    deadline read belongs to the memory arm.
    """
    decision = route_request("When is the renewal due?")
    assert decision.route == "knowledge", decision
    assert decision.rule == "due_question", decision

    # A world deadline is not the user's stored data.
    assert route_request("When is the tax filing deadline?").route == "chat"


def test_a_conversational_turn_with_no_boundary_still_goes_to_chat():
    undecided = RouteDecision(
        route="unknown",
        confidence=0.0,
        reason="insufficient_route_evidence",
        rule="abstain",
        requires_knowledge=False,
        requires_model=True,
    )
    expectation = turn_state.predict_turn("that was a good film", history=[])
    assert brain.expectation_may_redirect_to_chat(undecided, expectation) is True


def test_a_plain_statement_keeps_its_normal_path():
    decision = route_request("The renewal is due March 4")
    expectation = turn_state.predict_turn("I finished the book", history=[])
    # Nothing in this expectation asks for a redirect at all.
    assert brain.expectation_may_redirect_to_chat(decision, expectation) is False


def test_a_composed_event_reply_may_not_move_the_date():
    """Live composition of the c31 write said "Monday, October 20, 2025" for
    an event stored at 2026-10-19T10:00. The clock was grounded; the day was
    not."""
    payload = {
        "result": {
            "operation": "CREATE",
            "record_type": "event",
            "temporal": {
                "resolved_start": "2026-10-19T10:00:00-05:00",
                "resolved_end": "2026-10-19T10:00:00-05:00",
                "when": "tomorrow 10am",
            },
        }
    }
    assert brain._stated_dates_are_grounded(
        "Scheduled 'optician' for Monday, October 19, 2026 at 10:00 AM.", payload
    )
    assert brain._stated_dates_are_grounded("Scheduled it for tomorrow at 10am.", payload)
    assert not brain._stated_dates_are_grounded(
        "The optician is in for Monday, October 20, 2025, at 10 am.", payload
    )
    assert not brain._stated_dates_are_grounded("It's set for Oct 18 at 10am.", payload)


def test_a_world_request_is_not_read_as_a_data_read():
    """The self-question rule must not swallow world chat.

    "Tell me one fun fact about octopuses" contains "tell" and "me"; only a
    possessive makes it a read of the user's own data. The end-to-end Luna
    suite asserts the world-chat route, and it caught this.
    """
    assert route_request("Tell me one fun fact about octopuses, keep it short").route == "chat"
    assert route_request("how do I make bread").route == "chat"
    assert route_request("Tell me my garage code").route == "knowledge"
    assert route_request("Show me my reminders").route == "knowledge"


def test_a_fact_read_reply_may_not_invent_a_year():
    """Live composition of "When is the renewal due?" produced "March 4, 2027"
    for a fact stored as "The renewal is due March 4" with no year at all."""
    payload = {
        "result": {
            "ok": True,
            "query": "renewal due",
            "facts": [{"value": "The renewal is due March 4"}],
            "previous": [],
        }
    }
    assert brain._stated_dates_are_grounded(
        "Your renewal is due March 4.", payload
    )
    assert not brain._stated_dates_are_grounded(
        "The renewal is due on March 4, 2027.", payload
    )


def test_the_event_baseline_names_the_stored_date():
    payload = {
        "result": {
            "operation": "CREATE",
            "record_type": "event",
            "data": {"title": "optician"},
            "temporal": {
                "when": "tomorrow 10am",
                "resolved_start": "2026-10-19T10:00:00-05:00",
            },
        }
    }
    reply = brain.format_knowledge_result(payload)
    assert "October 19, 2026" in reply, reply


class _StubKnowledge:
    def __init__(self):
        self.calls = []

    def delete_fact(self, query, confirm=False):
        self.calls.append({"query": query, "confirm": confirm})
        if not confirm:
            return {
                "result": {
                    "ok": True,
                    "status": "needs_confirmation",
                    "operation": "DELETE_FACT_PENDING",
                    "query": query,
                    "matches": [{"record_id": 7, "value": "My old address was 14 Willow Lane."}],
                    "deleted": [],
                }
            }
        return {
            "result": {
                "ok": True,
                "status": "deleted",
                "operation": "DELETE_FACT",
                "query": query,
                "count": 1,
                "deleted": [{"record_id": 7, "value": "My old address was 14 Willow Lane."}],
            }
        }


def _bare_brain(knowledge):
    instance = object.__new__(brain.Brain)
    instance.knowledge = knowledge
    instance._pending_fact_deletions = {}
    instance._pending_lock = threading.Lock()
    return instance


def _turn(instance, text, conversation_id="conv-1"):
    return instance._pending_fact_deletion_turn(
        text, conversation_id=conversation_id, location="test"
    )


def test_a_fact_deletion_is_confirmed_before_it_deletes():
    knowledge = _StubKnowledge()
    instance = _bare_brain(knowledge)

    # Nothing pending: the turn is not ours to answer.
    assert _turn(instance, "delete the note about my old address") is None

    # Core records the pending deletion from the knowledge reply.
    instance.remember_pending_fact_deletion(
        "conv-1",
        {
            "query": "my old address",
            "matches": [{"record_id": 7, "value": "My old address was 14 Willow Lane."}],
        },
    )

    # A different turn abandons it: a later "yes" must not delete something
    # the user has moved on from.
    assert _turn(instance, "what's the weather like") is None
    assert not instance._pending_fact_deletions

    instance.remember_pending_fact_deletion(
        "conv-1", {"query": "my old address", "matches": []}
    )
    pending_reply = _turn(instance, "yes")
    assert pending_reply is not None
    assert knowledge.calls == [{"query": "my old address", "confirm": True}]
    assert "14 Willow Lane" in pending_reply["reply"], pending_reply
    assert "Deleted" in pending_reply["reply"], pending_reply
    assert not instance._pending_fact_deletions


def test_a_declined_deletion_deletes_nothing():
    knowledge = _StubKnowledge()
    instance = _bare_brain(knowledge)
    instance.remember_pending_fact_deletion(
        "conv-1", {"query": "my old address", "matches": []}
    )
    reply = _turn(instance, "no")
    assert reply is not None
    assert knowledge.calls == []
    assert "Nothing was deleted" in reply["reply"], reply
    assert not instance._pending_fact_deletions


def test_the_pending_delete_reply_names_the_fact_and_asks():
    payload = {
        "result": {
            "ok": True,
            "status": "needs_confirmation",
            "operation": "DELETE_FACT_PENDING",
            "query": "my old address",
            "matches": [{"record_id": 7, "value": "My old address was 14 Willow Lane."}],
            "deleted": [],
        }
    }
    reply = brain.format_knowledge_result(payload)
    assert "14 Willow Lane" in reply, reply
    assert "Nothing has been deleted" in reply, reply
    assert "say yes" in reply.lower(), reply


def test_the_delete_result_reports_what_was_removed():
    payload = {
        "result": {
            "ok": True,
            "status": "deleted",
            "operation": "DELETE_FACT",
            "query": "my old address",
            "count": 1,
            "deleted": [{"record_id": 7, "value": "My old address was 14 Willow Lane."}],
        }
    }
    reply = brain.format_knowledge_result(payload)
    assert reply.startswith("Deleted"), reply
    assert "14 Willow Lane" in reply, reply
