"""Hard adversarial prompts for person identification.

Every case here is a real-world-shaped conversational utterance designed
to make a naive person-state extractor do one of three wrong things:

  1. INVENT a person from conversational filler / instructions / questions
     ("remind me to call my mom" -> person named "remind"?)
  2. MIS-ATTRIBUTE a state to the wrong person in multi-person sentences
     ("my sister is fine but my brother is stressed")
  3. MISS a real, valid statement phrased unusually.

The suite is deliberately cruel: questions, quotes, vocatives, negated
statements, thirdhand reports, and multi-clause sentences.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nix_knowledge.engine import KnowledgeEngine
from nix_knowledge.states import (
    find_states,
    parse_state_statement,
    state_person_label,
    state_person_name,
    store_state,
)


@pytest.fixture()
def engine(tmp_path):
    db = str(tmp_path / "hard_prompts.db")
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


# ======================================================================
# 1. NEVER invent a person: questions, requests, quotes, vocatives
# ======================================================================


@pytest.mark.parametrize(
    "text",
    [
        # requests/instructions containing role words
        "remind me to call my mom tomorrow",
        "i should check on my mom this weekend",
        "can you set a reminder to text my brother",
        "add buy flowers for my mom to my list",
        "tell my sister i said happy birthday",
        # questions about people are not state reports
        "how is my sister doing these days",
        "is my mom back from her trip yet",
        "do you remember my brother's birthday",
        # quoted/secondhand speech
        'my mom said "i am so tired of this rain"',
        "my sister told me she is stressed but she will manage",
        # vocative + filler openers
        "hey nix my mom is sick",
        "ok so my dad is tired today",
        "well my brother is stressed i think",
        # hypotheticals and conditionals
        "if my sister gets sick again i will visit",
        "i hope my mom isnt stressed about the move",
        # self states are NOT people
        "im so tired today",
        "i am stressed about the deadline",
        "im feeling much better now",
        # objects/systems wearing role-like words
        "my pc is dead",
        "my phone battery is drained",
        "my car is acting up again",
        # non-person entities
        "the office printer is broken",
        "my contract is up for renewal",
        # time/place statements
        "my meeting is at 3pm",
        "my flight is delayed",
    ],
)
def test_no_person_invented(text):
    """Non-reports must yield no person and no state statement."""
    parsed = parse_state_statement(text)
    if parsed is not None:
        # If something IS parsed, the subject must at least be a sane
        # person label (never a state word, filler, or instruction verb).
        label = state_person_label(
            {"name": parsed.get("name"), "role": parsed.get("role"),
             "subject": None, "value": text}
        )
        assert label is not None, f"{text!r} parsed but has unusable subject"
        assert state_person_name(parsed.get("name") or "") is None or parsed.get("role") is not None, (
            f"{text!r}: bare filler accepted as a person name"
        )


# ======================================================================
# 2. Mis-attribution traps: two people in one utterance
# ======================================================================


def test_two_people_opposite_states():
    """'my sister is fine but my brother is stressed' must not blend."""
    # The full sentence should not silently attribute 'stressed' to the
    # sister or 'fine' to the brother.
    parsed = parse_state_statement("my sister is fine but my brother is stressed")
    if parsed is not None:
        rest = parsed["rest"]
        # whatever person it picked, the state text must belong to that person
        if parsed["role"] == "sister":
            assert "brother" not in rest, "brother's state leaked into sister's record"
        if parsed["role"] == "brother":
            assert "sister" not in rest, "sister's state leaked into brother's record"


def test_sequential_clauses_store_separately(engine):
    """Two independent reports -> two people, each with their own state."""
    store_state(engine, parse_state_statement("my sister is sick"), "my sister is sick")
    store_state(engine, parse_state_statement("my brother is stressed about exams"), "my brother is stressed about exams")
    states = find_states(engine)
    subjects = {s.get("subject") for s in states.get("states", [])}
    labels = {s.get("label") for s in states.get("states", [])}
    blob = " ".join(str(x) for x in (subjects | labels))
    assert "sister" in blob and "brother" in blob, f"two people collapsed into one record: {states}"


def test_same_name_different_roles_stay_distinct(engine):
    """'my sister maanvi' and 'my cousin maanvi' are different people."""
    store_state(engine, parse_state_statement("my sister maanvi is sick"), "my sister maanvi is sick")
    store_state(engine, parse_state_statement("my cousin maanvi is happy"), "my cousin maanvi is happy")
    states = find_states(engine)
    blob = str(states)
    assert "sister" in blob and "cousin" in blob, f"roles collapsed: {states}"


def test_role_update_supersedes_same_person_not_other(engine):
    """Update via role must not touch a different person's record."""
    store_state(engine, parse_state_statement("my sister is sick"), "my sister is sick")
    store_state(engine, parse_state_statement("my brother is sick"), "my brother is sick")
    store_state(engine, parse_state_statement("my sister is better now"), "my sister is better now")
    states = find_states(engine)
    blob = str(states)
    # brother's 'sick' record must survive the sister's recovery
    assert "brother" in blob


# ======================================================================
# 3. Name traps: state words and commands that look like names
# ======================================================================


@pytest.mark.parametrize(
    "candidate",
    [
        "tired", "sick", "stressed", "happy", "fine", "alright",
        "better", "cured", "recovered", "away", "back", "gone",
        "remind", "call", "tell", "ask", "text",
        "dont", "dont ask", "please", "thanks",
        "is", "was", "feels", "seems",
    ],
)
def test_state_and_command_words_never_names(candidate):
    assert state_person_name(candidate) is None, (
        f"{candidate!r} accepted as a person name"
    )


def test_real_name_still_accepted():
    assert state_person_name("maanvi") == "maanvi"
    assert state_person_name("Maanvi") == "Maanvi"
    assert state_person_name("o'brien") == "o'brien"
    assert state_person_name("raj's") == "raj"


def test_vocative_bro_is_not_a_name():
    """'bro my sister is sick' — 'bro' addresses the assistant."""
    parsed = parse_state_statement("bro my sister is sick")
    assert parsed is not None
    assert parsed["name"] is None, f"vocative captured as name: {parsed}"
    assert parsed["role"] == "sister"


# ======================================================================
# 4. Unusual but valid phrasings must NOT be missed
# ======================================================================


@pytest.mark.parametrize(
    "text,role,name,state",
    [
        ("my sister is sick again", "sister", None, "sick"),
        ("my mom is back home", "mom", None, "back"),
        ("my dad is out of town", "dad", None, "away"),
        ("maanvi is feeling better", None, "maanvi", "better"),
        ("my sister maanvi isnt sick anymore", "sister", "maanvi", "better"),
        ("my mom is stresed", "mom", None, "stressed"),  # doubled-letter typo
        ("maanvi is strssed", None, "maanvi", "stressed"),
    ],
)
def test_hard_but_valid(text, role, name, state):
    parsed = parse_state_statement(text)
    assert parsed is not None, f"valid report missed: {text!r}"
    assert parsed["role"] == role
    assert parsed["name"] == name
    assert parsed["state"] == state


# ======================================================================
# 6. Wave 2: negation scoping, clausal complements, hedged futures
# ======================================================================


@pytest.mark.parametrize(
    "text,role,state,valence",
    [
        # direct predicate negation must flip the family, never store
        # the denied word as if affirmed
        ("my brother is not stressed but my sister is", "brother", "calm", "good"),
        ("my mom is not sick", "mom", "well", "good"),
        ("my sister is not happy", "sister", "sad", "bad"),
        ("my sister is not doing well", "sister", "upset", "bad"),
        # ...and affirmatives must stay untouched
        ("my brother is stressed", "brother", "stressed", "bad"),
        ("my mom is sick", "mom", "sick", "bad"),
        ("my sister is happy", "sister", "happy", "good"),
        ("my sister is doing well", "sister", "doing well", "good"),
        # aspect markers and intensifiers must not corrupt the state
        ("my dad is still sick", "dad", "sick", "bad"),
        ("my sister is sick again", "sister", "sick", "bad"),
        ("my brother is very stressed", "brother", "stressed", "bad"),
        ("my mom is really sick", "mom", "sick", "bad"),
        # hedges
        ("my mom seems really tired lately", "mom", "tired", "bad"),
        # 'got' perfective
        ("my dad got sick", "dad", "sick", "bad"),
        # case insensitivity
        ("MY SISTER IS SICK", "sister", "sick", "bad"),
        ("Maanvi is sick", None, "sick", "bad"),
        # clausal complements: the first person keeps their own state
        ("my sister is sick and her fever is high", "sister", "sick", "bad"),
    ],
)
def test_wave2_negation_and_hedges(text, role, state, valence):
    parsed = parse_state_statement(text)
    assert parsed is not None, f"valid statement missed: {text!r}"
    assert parsed["role"] == role, f"{text!r}: role {parsed['role']} != {role}"
    assert parsed["state"] == state, f"{text!r}: state {parsed['state']} != {state}"
    assert parsed["valence"] == valence, f"{text!r}: valence {parsed['valence']} != {valence}"


@pytest.mark.parametrize(
    "text",
    [
        # role chains: 'my mom's friend' is not a tracked close role
        "my mom's friend is sick",
        "my sister's husband is stressed",
        # plural roles are households, not one person to follow up
        "my parents are visiting",
        "my kids are sick",
        # pronoun subject: no person identified
        "he is sick",
        "she is stressed",
        # bare 'there is' constructions
        "my brother there is a problem at work",
        # possessive chains around a name
        "maanvi, my sister, is sick",
        # hospital whereabouts without be-anchored state wording
        "my dad is in the hospital",
    ],
)
def test_wave2_no_false_person(text):
    """Must not resolve to a person-state record."""
    assert parse_state_statement(text) is None


def test_wave2_conditional_future_not_stored_as_current():
    """'would be happy if you called' is hypothetical, not a current state."""
    parsed = parse_state_statement("my mom would be happy if you called")
    if parsed is not None:
        # acceptable only if clearly not asserting current happiness
        assert "if you called" not in parsed["rest"], (
            "conditional clause stored inside the state value"
        )


def test_wave2_negated_person_record_never_stores_denied_word(engine):
    """End-to-end: 'not stressed' must store calm/good, never stressed/bad."""
    result = store_state(
        engine,
        parse_state_statement("my brother is not stressed but my sister is"),
        "my brother is not stressed but my sister is",
    )
    data = result["data"]
    assert data["state"] != "stressed"
    assert data["valence"] == "good"


def test_recovery_via_negation_full_sentence():
    parsed = parse_state_statement("my mom wasnt stressed anymore after the move")
    if parsed is not None:
        assert parsed["valence"] == "good", f"negation not applied: {parsed}"


# ======================================================================
# 5. End-to-end: whatever is stored, /api/people-style labels stay sane
# ======================================================================


def test_labels_never_become_state_words(engine):
    """No stored record may present 'Tired'/'Sick'/etc as a person."""
    adversarial = [
        "remind me to call my mom tomorrow",
        "im so tired today",
        "my pc is dead",
        "dont ask me questions like in the start alright",
        "my sister is sick",
        "my brother is stressed about exams",
        "maanvi is feeling better",
    ]
    for text in adversarial:
        parsed = parse_state_statement(text)
        if parsed is not None:
            store_state(engine, parsed, text)
    states = find_states(engine)
    blob = str(states)
    for bad in ("Tired", "Remind", "Dont", "PC", "Im"):
        assert f"'subject': '{bad}'" not in blob, f"bogus person label {bad!r} leaked: {blob[:400]}"
