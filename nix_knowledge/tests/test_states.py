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
    parse_state_statement,
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


@pytest.mark.parametrize(
    "text",
    [
        "what is my garage door code",
        "my garage door code is 4821",
        "remind me to stretch tomorrow at 9am",
        "my sister works at NASA",
        "my name is Sai",
    ],
)
def test_parse_negative(text):
    assert parse_state_statement(text) is None


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
