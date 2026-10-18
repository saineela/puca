"""WP6 regression tests: personal questions must select a function and reach
the store.

Measured in the WP2 confirmation run: 37 of 65 zero-recall answers were the
pipeline's own "no function selected" reply, because the deterministic selector
had no rule for question forms like "Where do I work?", and the fact finder
required every word of the question to appear in the stored sentence.

These tests pin both halves of the fix:

* the selector chooses a read function for each question family the owner asked
  to be covered — known facts, unknown facts, temporal questions, paraphrases,
  multi-fact questions — and keeps its existing precedence for scheduling,
  calendar vocabulary and statements;
* the finder's content-word/synonym comparison reaches the stored sentence from
  a paraphrase, so the question's own words ("colour", "opens", "name") cannot
  hide the answer.
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


def selected(text: str):
    return route(text, TemporalResolver(timezone="America/Chicago"))


# ---------------------------------------------------------------- selection
def test_known_and_unknown_fact_questions_select_the_fact_finder():
    """A question is a read whether or not the answer exists; the abstention
    belongs to the finder returning nothing, not to a missing selection."""
    known = [
        ("Where do I work?", "work"),
        ("What colour is my laptop?", "laptop"),
        ("Which code opens my garage?", "code garage"),
        ("What's the name of my dog?", "name dog"),
        ("What days am I at the gym now?", "gym"),
        ("What's the Wi-Fi key at home?", "wi-fi key"),
        ("Where does my sister live?", "sister live"),
    ]
    for text, query in known:
        name, arguments = selected(text)
        assert name == "find_facts", (text, name)
        assert arguments["query"] == query, (text, arguments)

    # Unknown facts route the same way: no keyword table decides whether a
    # value exists, and a wrong "no selection" is what produced the failure
    # reply in the D2 evidence.
    for text in (
        "What is my blood type?",
        "What size shoes do I wear?",
        "What is my mother's maiden name?",
    ):
        name, arguments = selected(text)
        assert name == "find_facts", (text, name)
        assert arguments["query"], (text, arguments)


def test_temporal_and_history_questions_keep_their_frame():
    name, arguments = selected("When does my passport expire?")
    assert name == "find_facts" and arguments["query"] == "passport"
    assert arguments["history"] is False

    name, arguments = selected("What was my garage code before?")
    assert name == "find_facts" and arguments["history"] is True

    name, arguments = selected("Which city did I live in before?")
    assert name == "find_facts" and arguments["history"] is True


def test_paraphrases_resolve_to_content_words_only():
    assert selected("Who do I rent from?") == (
        "find_facts", {"query": "rent", "history": False}
    )
    assert selected("What medication should I avoid?") == (
        "find_facts", {"query": "medication avoid", "history": False}
    )
    assert selected("How long is my passport good for?") == (
        "find_facts", {"query": "passport good", "history": False}
    )


def test_multi_fact_questions_become_one_multi_action():
    name, arguments = selected("Tell me my garage code and my Wi-Fi password.")
    assert name == "multi_action"
    assert [action["name"] for action in arguments["actions"]] == [
        "find_facts", "find_facts",
    ]
    assert arguments["actions"][0]["arguments"]["query"] == "garage code"
    assert arguments["actions"][1]["arguments"]["query"] == "wi-fi password"


def test_selection_precedence_is_unchanged():
    """The wider recall rule must not swallow scheduling, calendar, state or
    statement turns."""
    assert selected("Remind me to call mom tomorrow at 5pm")[0] == "create_calendar_event"
    assert selected("Set an alarm for 7am")[0] == "create_calendar_event"
    assert selected("When is my tsa meeting")[0] == "find_calendar_events"
    assert selected("what is my schedule")[0] == "find_calendar_events"
    assert selected("My flight is on Friday at 9am")[0] == "create_fact"
    assert selected("I moved to Boulder")[0] == "store_profile_keys"
    assert selected("How is my mother doing?")[0] == "find_states"
    assert selected("How many moons does Jupiter have?") is None
    assert selected("Write me a line about the sea") is None


# ------------------------------------------------------------------ reach
def _values(text: str, needle: KnowledgeNeedle) -> str:
    name, arguments = selected(text)
    assert name == "find_facts", (text, name)
    result = needle.tool_functions[name](**arguments)
    return " ".join(str(fact.get("value", "")) for fact in result["facts"])


def test_content_words_reach_the_stored_sentence(tmp_path):
    engine = KnowledgeEngine(str(tmp_path / "recall.db"))
    try:
        for value in (
            "My garage code is 8042.",
            "My dog is named Biscuit.",
            "My passport expires in 2031.",
            "My landlord is Dana.",
            "I am allergic to penicillin.",
            "My Wi-Fi password is maple-otter-42.",
            "I go to the gym on Monday and Thursday.",
        ):
            create_fact(engine, value, source="test", reason="wp6 regression")
        needle = KnowledgeNeedle(engine)

        for text, expected in (
            ("Which code opens my garage?", "8042"),
            ("What's the name of my dog?", "Biscuit"),
            ("How long is my passport good for?", "2031"),
            ("Who do I rent from?", "Dana"),
            ("What medication should I avoid?", "penicillin"),
            ("What's the Wi-Fi key at home?", "maple-otter-42"),
            ("What days am I at the gym now?", "Monday"),
        ):
            values = _values(text, needle)
            assert expected in values, (text, values)

        # An unknown question still finds nothing: the fix widens matching, it
        # does not invent an answer.
        name, arguments = selected("What is my blood type?")
        result = needle.tool_functions[name](**arguments)
        assert result["count"] == 0
    finally:
        engine.close()
        for suffix in ("", "_vectors.db", "_entities.db"):
            try:
                os.remove(str(tmp_path / f"recall.db{suffix}"))
            except FileNotFoundError:
                pass


def test_empty_fact_lookup_retries_the_calendar(tmp_path):
    """A question stored as a calendar event must still answer: the fact finder
    gets the first try, the calendar the second, only for read-only lookups."""
    engine = KnowledgeEngine(str(tmp_path / "fallback.db"))
    try:
        needle = KnowledgeNeedle(engine)
        needle.tool_functions["create_calendar_event"](
            title="dentist appointment", temporal_expression="friday at 3pm"
        )
        name, arguments = selected("When is my dentist appointment?")
        assert name == "find_calendar_events"
        result = needle.tool_functions[name](**arguments)
        assert result["events"], result

        # Nothing stored anywhere: the answer stays empty rather than becoming
        # a different event.
        name, arguments = selected("When is my optician appointment?")
        result = needle.tool_functions[name](**arguments)
        assert "optician" in arguments.get("title", ""), arguments
        assert not result["events"]
    finally:
        engine.close()
        for suffix in ("", "_vectors.db", "_entities.db"):
            try:
                os.remove(str(tmp_path / f"fallback.db{suffix}"))
            except FileNotFoundError:
                pass
