"""Regression tests for the four open defects in the D2/D4 evidence.

Each test below failed against the code as it stood before the fix and pins one
of the documented defects:

* **stale value on a current-fact read** - a stored sentence that says
  "previously" was returned beside today's value for "What city do I live in?";
* **"put X in for <time>"** - the deterministic selector had no rule, so the
  turn ended in "no function selected";
* **a durable deadline statement** - "The renewal is due March 4" reached no
  write function, so nothing was stored;
* **deletion by description** - "delete the note about my old address" was read
  as a calendar cancel; the stored fact was never a deletion target.

The tests are deliberately at the knowledge boundary: no model, no GPU, and no
network, so they can run beside a formal evaluation without competing for the
device.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nix_knowledge.engine import KnowledgeEngine
from nix_knowledge.functions.facts import create_fact
from nix_knowledge.needle import KnowledgeNeedle
from nix_knowledge.rules import route
from nix_knowledge.temporal import TemporalResolver


def _resolver():
    return TemporalResolver(timezone="America/Chicago")


def _selected(text):
    return route(text, _resolver())


def _engine(tmp_path, name):
    return KnowledgeEngine(str(tmp_path / name))


def _close(engine, tmp_path, name):
    engine.close()
    for suffix in ("", "_vectors.db", "_entities.db"):
        try:
            os.remove(str(tmp_path / f"{name}{suffix}"))
        except FileNotFoundError:
            pass


def _values(needle, text):
    name, arguments = _selected(text)
    assert name == "find_facts", (text, name)
    result = needle.tool_functions[name](**arguments)
    return " | ".join(str(fact.get("value", "")) for fact in result["facts"]), result


# --------------------------------------------------------------- defect 1
def test_current_city_read_excludes_a_historical_fact(tmp_path):
    """The D4 c17 failure: the current read returned the old city too.

    The store keeps history on purpose, but a question without a past frame
    asks for today's value only.
    """
    engine = _engine(tmp_path, "stale.db")
    try:
        create_fact(engine, "I previously lived in Boise.", source="test")
        create_fact(engine, "My home city is Flagstaff.", source="test")
        needle = KnowledgeNeedle(engine)

        current, result = _values(needle, "What city do I live in?")
        assert "Flagstaff" in current, current
        assert "Boise" not in current, (current, result)
    finally:
        _close(engine, tmp_path, "stale.db")


def test_a_past_frame_question_still_reaches_the_historical_fact(tmp_path):
    """Removing history from the current read must not remove it entirely."""
    engine = _engine(tmp_path, "history.db")
    try:
        create_fact(engine, "I previously lived in Boise.", source="test")
        create_fact(engine, "My home city is Flagstaff.", source="test")
        needle = KnowledgeNeedle(engine)

        current, _ = _values(needle, "What city do I live in?")
        assert "Boise" not in current, current

        # The finder is asked for history explicitly, as the selector does for
        # "which city did I live in before".
        result = needle.tool_functions["find_facts"](query="live", history=True)
        values = " ".join(str(row.get("value", "")) for row in result["facts"])
        previous = " ".join(str(row.get("value", "")) for row in result.get("previous", []))
        assert "Boise" in (values + " " + previous), result
    finally:
        _close(engine, tmp_path, "history.db")


def test_an_unknown_city_question_invents_nothing(tmp_path):
    """Fail-closed: an empty store answers empty rather than borrowing a fact."""
    engine = _engine(tmp_path, "unknown.db")
    try:
        create_fact(engine, "My dog is named Biscuit.", source="test")
        needle = KnowledgeNeedle(engine)
        current, result = _values(needle, "What city do I live in?")
        assert current == "", result
        assert result["count"] == 0, result
    finally:
        _close(engine, tmp_path, "unknown.db")


# --------------------------------------------------------------- defect 2
def test_put_x_in_for_selects_a_calendar_write():
    """The D4 c31 failure: "put the optician in for Monday at 10am"."""
    selected = _selected("Put the optician in for Monday at 10am")
    assert selected is not None, selected
    name, arguments = selected
    assert name == "create_calendar_event", (name, arguments)
    assert "optician" in arguments["title"].lower(), arguments
    assert arguments["temporal_expression"], arguments
    resolved = _resolver().resolve(arguments["temporal_expression"])
    assert resolved is not None, arguments
    assert resolved.start == resolved.start.replace(hour=10, minute=0), resolved.start


def test_put_in_for_handles_an_indefinite_article_and_a_leading_put_in():
    for text, title in (
        ("put my dentist appointment in for friday at 3pm", "dentist"),
        ("put in a haircut for tomorrow at 9", "haircut"),
        ("Put the gym in for Monday", "gym"),
    ):
        selected = _selected(text)
        assert selected is not None, text
        name, arguments = selected
        assert name == "create_calendar_event", (text, name, arguments)
        assert title in arguments["title"].lower(), (text, arguments)


def test_put_in_for_without_a_resolvable_time_defers_rather_than_guessing():
    assert _selected("put the optician in for whenever") is None


def test_a_put_in_for_event_can_be_read_back(tmp_path):
    """Write then read: the new rule earns its place only if the event is
    reachable afterwards. The stored title is the user's noun ("optician"),
    and the follow-up question adds the generic word "appointment"."""
    engine = _engine(tmp_path, "put_read.db")
    try:
        needle = KnowledgeNeedle(engine)
        created = needle.tool_functions["create_calendar_event"](
            title="optician", temporal_expression="Monday at 10am"
        )
        assert created.get("record_id"), created

        name, arguments = _selected("When is my optician appointment?")
        assert name == "find_calendar_events", (name, arguments)
        found = needle.tool_functions[name](**arguments)
        assert found["events"], found
        assert "optician" in str(found["events"][0].get("title", "")).lower(), found
    finally:
        _close(engine, tmp_path, "put_read.db")


# --------------------------------------------------------------- defect 3
def test_a_due_statement_is_stored_as_a_fact():
    """The D4 c32 knowledge half: a durable deadline must reach a write."""
    selected = _selected("The renewal is due March 4")
    assert selected is not None, selected
    assert selected[0] == "create_fact", selected
    assert "renewal" in selected[1]["value"].lower(), selected


def test_a_due_question_is_a_read_not_a_write(tmp_path):
    """The stored deadline must be readable, and the question must not write.

    The selector may read it as a calendar lookup; the read-only fallback then
    reaches the fact store. What matters is that the answer comes back and the
    store does not grow a second record for a question.
    """
    engine = _engine(tmp_path, "due_read.db")
    try:
        create_fact(engine, "The renewal is due March 4", source="test")
        needle = KnowledgeNeedle(engine)
        before = len(engine.search("fact"))

        result = needle.process("When is the renewal due?")
        payload = result.get("result") or {}
        facts = payload.get("facts") or []
        assert any("March 4" in str(row.get("value", "")) for row in facts), payload
        assert len(engine.search("fact")) == before, "a question created a record"
    finally:
        _close(engine, tmp_path, "due_read.db")


# --------------------------------------------------------------- defect 4
def test_delete_by_description_asks_before_it_deletes(tmp_path):
    """The D4 c38 failure: deletion must target the fact, and only on confirm."""
    engine = _engine(tmp_path, "delete.db")
    try:
        create_fact(engine, "My old address was 14 Willow Lane.", source="test")
        needle = KnowledgeNeedle(engine)

        # routing: a note/fact deletion is not a calendar cancel
        name, arguments = _selected("Delete the note about my old address")
        assert name == "delete_fact", (name, arguments)
        assert "address" in arguments["query"].lower(), arguments

        pending = needle.tool_functions["delete_fact"](**arguments)
        assert pending["status"] == "needs_confirmation", pending
        assert pending["matches"], pending
        assert "14 Willow Lane" in " ".join(
            str(match.get("value", "")) for match in pending["matches"]
        ), pending
        # still there until the user says yes
        assert engine.search("fact"), "the fact was deleted before confirmation"

        done = needle.tool_functions["delete_fact"](**arguments, confirm=True)
        assert done["status"] == "deleted", done
        assert engine.search("fact") == [], "the confirmed delete did not remove the fact"
        assert engine.search("fact", status="deleted"), "the delete was not auditable"
    finally:
        _close(engine, tmp_path, "delete.db")


def test_delete_fails_closed_on_an_unknown_or_broad_description(tmp_path):
    engine = _engine(tmp_path, "delete_guard.db")
    try:
        create_fact(engine, "My old address was 14 Willow Lane.", source="test")
        create_fact(engine, "My old address was 8 Mill Road.", source="test")
        create_fact(engine, "My old address was 2 Quay Street.", source="test")
        create_fact(engine, "My old address was 77 Elm Close.", source="test")
        needle = KnowledgeNeedle(engine)

        # No match: nothing is deleted, and no record is invented.
        missing = needle.tool_functions["delete_fact"](
            query="the note about my boat licence", confirm=True
        )
        assert missing["status"] == "not_found", missing
        assert len(engine.search("fact")) == 4

        # Too broad: refuse rather than delete four records on one request.
        broad = needle.tool_functions["delete_fact"](
            query="my old address", confirm=True
        )
        assert broad["status"] == "refused_too_broad", broad
        assert len(engine.search("fact")) == 4
    finally:
        _close(engine, tmp_path, "delete_guard.db")


# --------------------------------------------------- adjacent phrasings
def test_a_deletion_paraphrase_still_targets_the_fact_store():
    """The same request in the shapes people actually use.

    "Delete the note about my old address" was the fixed row; these are its
    neighbours. Two of them still reached the calendar-cancel rule after the
    first fix, which is the same defect class, so they are pinned here.
    """
    for text, expected in (
        ("Can you remove the saved note about where I used to live?", "live"),
        ("Remove the note I saved about my old address", "address"),
        ("Delete my old address note", "address"),
        ("Delete the note about my old address", "address"),
        ("Forget my old address", "address"),
    ):
        name, arguments = _selected(text)
        assert name == "delete_fact", (text, name)
        assert expected in arguments["query"].lower(), (text, arguments)

    # A calendar noun is still a calendar cancel: the meta-noun is what marks
    # the target as memory.
    name, arguments = _selected("delete my dentist appointment")
    assert name == "cancel_calendar_event", (name, arguments)
    assert _selected("remove my alarm reminder for tomorrow")[0] == "cancel_calendar_event"


def test_a_book_in_for_paraphrase_selects_a_calendar_write():
    """\"put X in for Y\" with a different verb is the same calendar write."""
    for text, title in (
        ("Book the plumber in for Wednesday at 8am", "plumber"),
        ("Slot the dentist in for Monday", "dentist"),
        ("put my mother in for Tuesday at noon", "mother"),
    ):
        selected = _selected(text)
        assert selected is not None, text
        name, arguments = selected
        assert name == "create_calendar_event", (text, name, arguments)
        assert title in arguments["title"].lower(), (text, arguments)
        resolved = _resolver().resolve(arguments["temporal_expression"])
        assert resolved is not None, (text, arguments)


def test_a_paraphrased_deadline_statement_is_stored():
    """A durable deadline in the user's own words must reach a write."""
    for text in ("The permit is due April 2", "My passport renewal is due June 1"):
        selected = _selected(text)
        assert selected is not None, text
        assert selected[0] == "create_fact", (text, selected)


def _all_values(payload) -> list[str]:
    values: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "value" and isinstance(value, str):
                values.append(value)
            else:
                values.extend(_all_values(value))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_all_values(item))
    return values


def test_a_multi_fact_question_returns_both_stored_values(tmp_path):
    """\"Tell me my city and my dog\" is two reads, not one.

    The selector splits it into two find_facts actions and the needle runs both;
    a reply that returns one value would silently drop half the question.
    Measured limitation: the \"What is my X and my Y?\" shape is not split, and
    is recorded as a residual defect rather than asserted here.
    """
    engine = _engine(tmp_path, "multi.db")
    try:
        create_fact(engine, "My home city is Flagstaff.", source="test")
        create_fact(engine, "My dog is called Miso.", source="test")
        needle = KnowledgeNeedle(engine)

        result = needle.process("Tell me my city and my dog")
        values = " | ".join(_all_values(result))
        assert "Flagstaff" in values, values
        assert "Miso" in values, values
        assert len(engine.search("fact")) == 2, "a read created a record"
    finally:
        _close(engine, tmp_path, "multi.db")
