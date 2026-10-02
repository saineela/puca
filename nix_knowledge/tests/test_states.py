"""Tests for the current-state engine (close people)."""
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nix_knowledge.engine import KnowledgeEngine
from nix_knowledge.states import (
    find_states,
    follow_up_eligible,
    parse_state_statement,
    state_person_label,
    store_state,
)


@pytest.fixture()
def engine(tmp_path):
    db = str(tmp_path / "states_test.db")
    engine = KnowledgeEngine(db)
    try:
        yield engine
    finally:
        engine.close()
        for suffix in ("", "_vectors.db", "_entities.db"):
            try:
                os.remove(db + suffix)
            except FileNotFoundError:
                pass


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,role,name,state",
    [
        ("my sister is sick", "sister", None, "sick"),
        ("my sister is doing alright now", "sister", None, "alright"),
        ("my sister is doing fine now", "sister", None, "fine"),
        ("my sistser is doign alright now", "sister", None, "alright"),
        ("Jane is fine", None, "jane", "fine"),
        ("my sister maanvi is sick", "sister", "maanvi", "sick"),
        ("maanvi my sister is sick with flu", "sister", "maanvi", "sick"),
        ("my mom is happy today", "mom", None, "happy"),
        ("maanvi's cured now no flu yay", None, "maanvi", "cured"),
        ("my sister isnt sick anymore", "sister", None, "better"),
        ("my dad is tired after work", "dad", None, "tired"),
        ("my brother is stressed about exams", "brother", None, "stressed"),
    ],
)
def test_parse_positive(text, role, name, state):
    parsed = parse_state_statement(text)
    assert parsed is not None
    assert parsed["role"] == role
    assert parsed["name"] == name
    assert parsed["state"] == state


def test_close_person_follow_up_variables_are_written(engine):
    result = store_state(engine, parse_state_statement("my sister is sick"), "my sister is sick")
    data = result["data"]
    assert data["relationship_closeness"] == "close"
    assert data["follow_up_policy"] == "only_if_close_and_unwell_or_problem"
    assert data["follow_up_needed"] is True
    assert data["follow_up_answered"] is False
    assert data["follow_up_eligible"] is True
    assert follow_up_eligible(data) is True


def test_memory_policy_fields_distinguish_eligible_from_ordinary(engine):
    from nix_knowledge.memory_block import render_memory_block

    store_state(engine, parse_state_statement("my sister is sick"), "my sister is sick")
    block = render_memory_block(engine, current_text="my sister is sick")
    assert "eligible=true" in block
    assert "Your sister" in block

    store_state(engine, parse_state_statement("my sister is better now"), "my sister is better now")
    block = render_memory_block(engine, current_text="my sister is better now")
    assert "eligible=false" in block
    assert "answered=true" in block


def test_recovery_marks_prior_follow_up_answered(engine):
    store_state(engine, parse_state_statement("my sister is sick"), "my sister is sick")
    result = store_state(engine, parse_state_statement("my sister is better now"), "my sister is better now")
    data = result["data"]
    assert data["relationship_closeness"] == "close"
    assert data["follow_up_needed"] is False
    assert data["follow_up_answered"] is True
    assert data["follow_up_eligible"] is False
    assert follow_up_eligible(data) is False


@pytest.mark.parametrize(
    "text",
    [
        "what is my garage door code",
        "my garage door code is 4821",
        "remind me to stretch tomorrow at 9am",
        "my sister works at NASA",
        "my name is Sai",
        "alright is good",
        "fine is doing well",
        "okay is alright",
    ],
)
def test_parse_negative(text):
    assert parse_state_statement(text) is None


def test_typo_role_update_uses_canonical_person_state(engine):
    first = store_state(
        engine,
        parse_state_statement("my sister is sick"),
        "my sister is sick",
    )
    second = store_state(
        engine,
        parse_state_statement("my sistser is doing alright now"),
        "my sistser is doing alright now",
    )
    assert second["operation"] == "SUPERSEDE_STATE"
    assert first["record_id"] in second["superseded"]
    current = find_states(engine)["states"]
    assert len(current) == 1
    assert current[0]["role"] == "sister"
    assert current[0]["state"] == "alright"
    assert "sistser" not in current[0]["value"]


def test_non_close_person_is_never_follow_up_eligible(engine):
    result = store_state(
        engine,
        parse_state_statement("my cousin is sick"),
        "my cousin is sick",
    )
    data = result["data"]
    assert data["relationship_closeness"] == "ordinary"
    assert data["follow_up_needed"] is True
    assert data["follow_up_eligible"] is False
    assert data["follow_up_reason"] == "already_answered_or_not_close"


def test_alright_is_a_temporal_well_state_not_a_person():
    parsed = parse_state_statement("my sister is doing alright now")
    assert parsed is not None
    assert parsed["state"] == "alright"
    assert parsed["group"] == "well"
    assert parsed["valence"] == "good"
    assert parsed["name"] is None

    assert parse_state_statement("alright is good") is None
    assert parse_state_statement("fine is doing well") is None
    assert state_person_label({"name": "alright", "state": "alright"}) is None
    assert state_person_label({"subject": "someone close", "state": "alright"}) is None
    assert state_person_label({"subject": "your alright", "state": "alright"}) is None
    assert state_person_label({"role": "sister", "subject": "someone close"}) == "Your sister"


def test_follow_up_requires_real_close_relationship_and_unanswered_problem():
    eligible = {
        "role": "sister", "follow_up_policy": "only_if_close_and_unwell_or_problem",
        "follow_up_needed": True, "follow_up_answered": False, "valence": "bad",
    }
    assert follow_up_eligible(eligible)
    assert not follow_up_eligible({**eligible, "role": "cousin"})
    assert not follow_up_eligible({**eligible, "relationship_closeness": "ordinary"})
    assert not follow_up_eligible({**eligible, "follow_up_answered": True})
    assert not follow_up_eligible({**eligible, "valence": "good"})
    assert not follow_up_eligible({**eligible, "follow_up_needed": 1})


def test_valence_sides():
    sick = parse_state_statement("my sister is sick")
    cured = parse_state_statement("maanvi is cured now")
    assert sick["valence"] == "bad"
    assert cured["valence"] == "good"


# ----------------------------------------------------------------------
# Storage + supersede
# ----------------------------------------------------------------------


def test_store_and_supersede_roundtrip(engine):
    if engine.semantic is not None:
        engine.semantic.entities.register(
            "sister", "maanvi", record_id=0, evidence="seed"
        )

    r1 = store_state(
        engine,
        parse_state_statement("my sister is sick with flu"),
        "my sister is sick with flu",
    )
    assert r1["operation"] == "STORE_STATE"
    assert r1["about"] == "maanvi"  # resolved from registry

    # cured statement supersedes the sick one (name-only statement,
    # role resolved back through the registry)
    r2 = store_state(
        engine,
        parse_state_statement("maanvi's cured now no flu yay"),
        "maanvi's cured now no flu yay",
    )
    assert r2["operation"] == "SUPERSEDE_STATE"
    assert r1["record_id"] in r2["superseded"]

    states = find_states(engine)
    values = [s["value"] for s in states["states"]]
    assert values == ["maanvi's cured now no flu yay"]
    assert states["states"][0]["valence"] == "good"


def test_same_family_update_supersedes(engine):
    r1 = store_state(
        engine,
        parse_state_statement("my mom is sad today"),
        "my mom is sad today",
    )
    r2 = store_state(
        engine,
        parse_state_statement("my mom is happy today"),
        "my mom is happy today",
    )
    assert r2["operation"] == "SUPERSEDE_STATE"
    assert r1["record_id"] in r2["superseded"]
    assert find_states(engine)["count"] == 1


def test_identical_restatement_noops(engine):
    store_state(
        engine,
        parse_state_statement("my sister is sick"),
        "my sister is sick",
    )
    r2 = store_state(
        engine,
        parse_state_statement("my sister is sick"),
        "my sister is sick",
    )
    assert r2["operation"] == "STATE_NOOP"
    assert find_states(engine)["count"] == 1


def test_unrelated_records_not_touched(engine):
    engine.create("fact", {"value": "my garage door code is 4821"})
    r = store_state(
        engine,
        parse_state_statement("my sister is sick"),
        "my sister is sick",
    )
    assert r["operation"] == "STORE_STATE"
    assert r["superseded"] == []
    assert any(
        "garage" in str(rec.data.get("value", ""))
        for rec in engine.search("fact")
    )


def test_ambiguous_role_query_requests_clarification(engine):
    if engine.semantic is None:
        pytest.skip("local semantic dependencies/model are unavailable")
    engine.semantic.entities.register("sister", "maanvi")
    engine.semantic.entities.register("sister", "jane")
    store_state(engine, parse_state_statement("my sister maanvi is sick"), "my sister maanvi is sick")
    store_state(engine, parse_state_statement("my sister jane is happy"), "my sister jane is happy")

    result = find_states(engine, query="how is my sister doing")
    assert result["needs_clarification"] is True
    assert result["clarification"]["candidates"] == ["jane", "maanvi"]
    assert "Which sister" in result["clarification"]["question"]
    assert result["states"] == []

    selected = find_states(engine, query="how is my sister maanvi doing")
    assert selected["needs_clarification"] is False
    assert selected["states"][0]["name"] == "maanvi"


def test_ambiguous_role_state_write_is_not_assigned_randomly(engine):
    if engine.semantic is None:
        pytest.skip("local semantic dependencies/model are unavailable")
    engine.semantic.entities.register("sister", "maanvi")
    engine.semantic.entities.register("sister", "jane")
    result = store_state(
        engine,
        parse_state_statement("my sister is sick"),
        "my sister is sick",
    )
    assert result["operation"] == "NEEDS_CLARIFICATION"
    assert result["candidates"] == ["jane", "maanvi"]
    assert engine.search("person") == []


def test_find_states_hides_corrupt_state_word_person_records(engine):
    engine.create("person", {
        "statement_type": "current_state", "name": "alright",
        "subject": "alright", "state": "alright", "valence": "good",
        "value": "alright is good",
    })
    engine.create("person", {
        "statement_type": "current_state", "subject": "someone close",
        "state": "alright", "valence": "good", "value": "my sister is alright",
    })
    states = find_states(engine)["states"]
    assert len(states) == 1
    assert states[0]["subject"] == "Your sister"

    engine.create("person", {
        "statement_type": "current_state", "role": "sister",
        "subject": "user's sister", "state": "alright", "valence": "good",
        "value": "my sister is alright",
    })
    states = find_states(engine)["states"]
    assert len(states) == 2
    assert all(state["subject"] == "Your sister" for state in states)


def test_find_states_filters_by_query(engine):
    store_state(
        engine,
        parse_state_statement("my sister is sick"),
        "my sister is sick",
    )
    store_state(
        engine,
        parse_state_statement("my dad is tired"),
        "my dad is tired",
    )
    result = find_states(engine, query="dad")
    assert result["count"] == 1
    assert result["states"][0]["role"] == "dad"


# ----------------------------------------------------------------------
# Interjection guard: "bro" is not a name
# ----------------------------------------------------------------------


def test_interjection_not_a_name():
    parsed = parse_state_statement("bro my sister is sick")
    assert parsed is not None
    assert parsed["role"] == "sister"
    assert parsed["name"] is None  # "bro" must NOT become the name


def test_real_names_still_extracted():
    assert parse_state_statement("maanvi my sister is sick with flu")["name"] == "maanvi"
    assert parse_state_statement("my sister maanvi is sick")["name"] == "maanvi"


# ----------------------------------------------------------------------
# Moments: read-time rendering of superseded history (explicit dates)
# ----------------------------------------------------------------------


def test_moments_render_dated_history(engine):
    store_state(
        engine,
        parse_state_statement("my sister is sick"),
        "my sister is sick",
    )
    store_state(
        engine,
        parse_state_statement("my sister is cured"),
        "my sister is cured",
    )

    result = find_states(engine, query="sister")
    assert result["moments"], "moments missing from find_states result"
    text = result["moments"][0]["text"]

    # explicit dates, never relative words
    assert "was sick" in text
    assert "doing well now" in text or "cured now" in text
    assert "recently" not in text
    assert "yesterday" not in text
    # date shape: "Sep 12" or span "Sep 12-13" (or with year)
    assert re.search(r"[A-Z][a-z]{2} \d{1,2}", text), text


def test_moments_merge_repeats(engine):
    for text in ("my sister is sick", "my sister is sick with flu", "my sister is sick"):
        store_state(engine, parse_state_statement(text), text)
    store_state(engine, parse_state_statement("my sister is better"), "my sister is better")

    result = find_states(engine, query="sister")
    text = result["moments"][0]["text"]
    # one merged sick span, not three fragments
    assert text.count("was sick") == 1, text


def test_moments_no_history_no_moment(engine):
    store_state(engine, parse_state_statement("my sister is sick"), "my sister is sick")
    result = find_states(engine, query="sister")
    assert result["moments"] == []
