"""
Key Finding Algorithm tests: deterministic micro-fact extraction.

Covers the extraction rules, the human rendering, and deduplicating
storage through a real engine + temp database.
"""

from __future__ import annotations

import pytest

from nix_knowledge.engine import KnowledgeEngine
from nix_knowledge.keys import extract_keys, key_text, store_keys


# ----------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------


def test_sister_example_yields_three_keys():
    keys = extract_keys("my sister, named Maanvi is very naughty")
    texts = [key_text(k) for k in keys]

    assert "user has a sister" in texts
    assert "user's sister is named Maanvi" in texts
    assert any("sister Maanvi" in t and "naughty" in t for t in texts)


def test_named_pet():
    keys = extract_keys("my dog is named Rex")
    texts = [key_text(k) for k in keys]

    assert "user has a dog" in texts
    assert "user's dog is named Rex" in texts


def test_relation_then_name():
    keys = extract_keys("my brother Alex lives in Austin")
    texts = [key_text(k) for k in keys]

    assert "user has a brother" in texts
    assert "user's brother is named Alex" in texts


def test_named_person_possessive_fact():
    keys = extract_keys("Maanvi's birthday is June 3")
    texts = [key_text(k) for k in keys]

    assert texts == ["Maanvi's birthday is June 3"]


def test_first_person_preference():
    keys = extract_keys("I love hiking")
    texts = [key_text(k) for k in keys]

    assert texts == ["user loves hiking"]


def test_first_person_dislike():
    keys = extract_keys("I hate early mornings")
    texts = [key_text(k) for k in keys]

    assert texts == ["user hates early mornings"]


def test_progressive_event_is_not_a_trait_but_existence_is():
    # "coming to visit" is an event, not a durable trait - but the
    # existence of the relation is still a legitimate key.
    keys = extract_keys("my mom is coming to visit")
    texts = [key_text(k) for k in keys]

    assert texts == ["user has a mom"]


def test_questions_yield_no_keys():
    assert extract_keys("what is the capital of france") == []
    assert extract_keys("who won the game last night") == []
    assert extract_keys("is it going to rain today") == []


def test_commands_yield_no_keys():
    assert extract_keys("remind me to call mom tomorrow at 5pm") == []
    assert extract_keys("remember that my parking spot is 42") == []
    assert extract_keys("set an alarm for 6am") == []


def test_empty_and_garbage_input():
    assert extract_keys("") == []
    assert extract_keys("   ") == []
    assert extract_keys("what time is it") == []


# ----------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------


def test_key_text_predicates():
    assert key_text(
        {"subject": "sister Maanvi", "predicate": "is", "value": "naughty"}
    ) == "sister Maanvi is naughty"
    assert key_text(
        {"subject": "user's sister", "predicate": "is_named",
         "value": "Maanvi"}
    ) == "user's sister is named Maanvi"
    assert key_text(
        {"subject": "user", "predicate": "has_sister", "value": "true"}
    ) == "user has a sister"
    assert key_text(
        {"subject": "Maanvi", "predicate": "birthday", "value": "June 3"}
    ) == "Maanvi's birthday is June 3"


# ----------------------------------------------------------------------
# storage through a real engine
# ----------------------------------------------------------------------


@pytest.fixture
def engine(tmp_path):
    return KnowledgeEngine(database_path=tmp_path / "kb.db")


def test_store_keys_creates_key_records(engine):
    keys = extract_keys("my sister, named Maanvi is very naughty")
    stored = store_keys(engine, keys)

    assert len(stored) == 3

    records = engine.search("key")
    values = {r.data.get("value") for r in records}

    assert "user has a sister" in values
    assert "user's sister is named Maanvi" in values
    assert any("Maanvi" in v and "naughty" in v for v in values)

    for record in records:
        assert record.source == "key_finder"
        assert (record.data.get("context") or {}).get("source")


def test_store_keys_deduplicates(engine):
    keys = extract_keys("my sister, named Maanvi is very naughty")

    first = store_keys(engine, keys)
    second = store_keys(engine, keys)

    assert len(first) == 3
    assert second == []
    assert len(engine.search("key")) == 3


def test_store_keys_empty_input(engine):
    assert store_keys(engine, []) == []


# ----------------------------------------------------------------------
# expanded attribute families
# ----------------------------------------------------------------------


import pytest


def _texts(text):
    return [key_text(k) for k in extract_keys(text)]


def test_identity_name():
    assert _texts("my name is Alex") == ["user's name is Alex"]
    # "call me Al" is a NICKNAME, not a name replacement
    assert _texts("call me Al") == ["user goes by Al"]
    assert _texts("people call me sai") == ["user goes by Sai"]
    assert _texts("hey nix, my name is Alex") == ["user's name is Alex"]


def test_intro_compound():
    keys = extract_keys(
        "So, My name is Sai Neela, people call me sai, "
        "I am born on February 25 2009, and love programmign and "
        "hardware and wish to pursue Computer Engineering in college"
    )
    texts = [key_text(k) for k in keys]
    assert "user's name is Sai Neela" in texts
    assert "user goes by Sai" in texts
    assert "user's birthday is February 25, 2009" in texts
    assert "user loves programmign and hardware" in texts
    assert "user wants to pursue Computer Engineering" in texts
    assert len(texts) == 5


def test_birthday():
    texts = _texts("my birthday is June 3")
    assert texts == ["user's birthday is June 3"]
    keys = extract_keys("my birthday is June 3")
    assert keys[0].get("remind_birthday") == {"month": 6, "day": 3}


def test_birthday_formats():
    assert _texts("my birthday is 3 june")[0].endswith("June 3")
    assert _texts("my birthday is June 3, 2009")[0].endswith("June 3, 2009")
    assert _texts("i was born on june 3") == ["user's birthday is June 3"]
    assert _texts("my birthday is 6/3")[0].endswith("June 3")


def test_favorites():
    assert _texts("my favorite band is Radiohead") == [
        "user's favorite band is Radiohead"
    ]
    assert _texts("my favourite color is blue") == [
        "user's favorite color is blue"
    ]


def test_allergy_sensitive():
    keys = extract_keys("i'm allergic to peanuts")
    assert len(keys) == 1
    assert keys[0]["sensitive"] is True
    assert key_text(keys[0]) == "user is allergic to peanuts"


def test_health_condition_sensitive():
    keys = extract_keys("i have asthma")
    assert keys[0]["sensitive"] is True
    assert key_text(keys[0]) == "user has asthma"


def test_relation_attributes():
    assert "user's sister works at NASA" in _texts("my sister works at NASA")
    assert "user's brother lives in Austin" in _texts(
        "my brother lives in Austin"
    )
    assert "user's son is 5 years old" in _texts("my son is 5 years old")


def test_relation_age_suppresses_generic_attribute():
    # "my son is 5" must not ALSO produce "user's son is 5" twice
    texts = _texts("my son is 5 years old")
    assert texts.count("user's son is 5 years old") == 1


def test_occupation():
    assert _texts("i work as an electrician") == [
        "user works as an electrician"
    ]
    assert _texts("i'm a nurse") == ["user works as a nurse"]
    assert _texts("i work at NASA") == ["user works at NASA"]


def test_school_not_occupation():
    texts = _texts("i'm a student at Lincoln High")
    assert texts == ["user is a student at Lincoln High"]


def test_location_and_age():
    assert _texts("i live in Dallas") == ["user lives in Dallas"]
    assert _texts("i'm 16 years old") == ["user is 16 years old"]
    assert _texts("i'm 5 minutes late") == []


def test_goal():
    assert _texts("i want to learn Spanish") == [
        "user wants to learn Spanish"
    ]
    assert _texts("i'm training for a 5k") == [
        "user is training for a 5k"
    ]


def test_routines():
    assert _texts("i wake up at 6am") == ["user wakes up at 6am"]
    assert _texts("i go to bed at 11pm") == ["user goes to bed at 11pm"]
    assert "user goes to the gym every morning" in _texts(
        "i go to the gym every morning"
    )


def test_devices():
    assert _texts("my phone is an iPhone 15") == [
        "user's phone is a iPhone 15"
    ]
    assert _texts("i drive a Civic") == ["user drives a Civic"]


def test_diet():
    assert _texts("i'm vegetarian") == ["user is vegetarian"]
    assert _texts("i never eat cilantro") == ["user never eats cilantro"]
    assert _texts("i'm lactose intolerant") == [
        "user is lactose intolerant"
    ]


def test_sizes():
    assert _texts("i wear size 10 shoes") == ["user's shoe size is 10"]
    assert _texts("my shirt size is medium") == [
        "user's shirt size is medium"
    ]


def test_contact_sensitive():
    keys = extract_keys("my email is alex@example.com")
    assert keys[0]["sensitive"] is True
    assert key_text(keys[0]) == "user's email is alex@example.com"

    assert _texts("my phone number is 555-123-4567")[0].startswith(
        "user's phone"
    )


def test_nix_prefs():
    assert _texts("use metric") == ["nix uses metric units"]
    assert _texts("speak spanish to me") == [
        "nix speaks spanish with the user"
    ]


# ----------------------------------------------------------------------
# supersession
# ----------------------------------------------------------------------


def test_supersession_location(engine):
    store_keys(engine, extract_keys("i live in Dallas"))
    assert len(engine.search("key")) == 1

    store_keys(engine, extract_keys("i moved to Austin"))
    active = [r for r in engine.search("key") if "lives in" in r.data["value"]]
    superseded = engine.search("key", status="superseded")

    assert any("Austin" in r.data["value"] for r in active)
    assert len(superseded) == 1
    assert "Dallas" in superseded[0].data["value"]


def test_supersession_name(engine):
    store_keys(engine, extract_keys("my name is Alex"))
    store_keys(engine, extract_keys("my name is Robert"))

    superseded = engine.search("key", status="superseded")
    assert len(superseded) == 1
    assert "Alex" in superseded[0].data["value"]

    # nickname coexists with the name (different predicate)
    store_keys(engine, extract_keys("call me Al"))
    assert len(engine.search("key", status="superseded")) == 1


def test_no_supersession_for_relations(engine):
    # two different facts about two people coexist
    store_keys(engine, extract_keys("my sister works at NASA"))
    store_keys(engine, extract_keys("my brother lives in Austin"))

    assert len(engine.search("key")) == 4
    assert engine.search("key", status="superseded") == []


# ----------------------------------------------------------------------
# sensitivity flag
# ----------------------------------------------------------------------


def test_sensitive_flag_persisted(engine):
    store_keys(engine, extract_keys("i'm allergic to peanuts"))
    record = engine.search("key")[0]
    assert record.data.get("sensitive") is True


def test_non_sensitive_keys_have_no_flag(engine):
    store_keys(engine, extract_keys("my name is Alex"))
    record = engine.search("key")[0]
    assert record.data.get("sensitive") is None


# ----------------------------------------------------------------------
# birthday auto-reminder (fake actions engine)
# ----------------------------------------------------------------------


class _FakeActions:
    def __init__(self):
        self.scheduled = []
        self.cancelled = []

    def schedule(self, **kwargs):
        self.scheduled.append(kwargs)
        return type("A", (), {"id": len(self.scheduled)})()

    def cancel(self, **kwargs):
        self.cancelled.append(kwargs)
        return 1


def test_birthday_auto_reminder(engine):
    from datetime import timezone as tz
    from zoneinfo import ZoneInfo

    actions = _FakeActions()
    keys = extract_keys("my birthday is June 3")
    stored = store_keys(
        engine, keys, actions_engine=actions,
        timezone=ZoneInfo("America/Chicago"),
    )

    assert len(stored) == 1
    assert len(actions.scheduled) == 1
    call = actions.scheduled[0]
    assert call["action_type"] == "reminder"
    assert "birthday" in call["payload"]["title"].lower()
    assert call["metadata"]["auto"] == "birthday_reminder"


def test_birthday_reminder_needs_actions_engine(engine):
    keys = extract_keys("my birthday is June 3")
    stored = store_keys(engine, keys)  # no actions engine
    assert len(stored) == 1  # key still stored, just no reminder


# ----------------------------------------------------------------------
# negative cases for new families
# ----------------------------------------------------------------------


def test_questions_still_yield_nothing():
    assert extract_keys("what is my favorite color?") == []
    assert extract_keys("who lives in texas") == []


def test_commands_still_yield_nothing():
    assert extract_keys("remind me to buy peanuts") == []
    assert extract_keys("set an alarm for my birthday") == []


# ----------------------------------------------------------------------
# Mem0-style memory management: NOOP / ADD / SUPERSEDE / audit
# ----------------------------------------------------------------------


def test_noop_on_alias_equivalent(engine):
    # contraction + number-word variants are the SAME key
    first = store_keys(engine, extract_keys("i'm 16 years old"))
    assert len(first) == 1

    second = store_keys(engine, extract_keys("im 16 years old"))
    assert second == []
    assert len(engine.search("key")) == 1


def test_noop_on_number_word_equivalent(engine):
    store_keys(engine, extract_keys("i'm 16 years old"))
    store_keys(engine, extract_keys("i am sixteen years old"))
    assert len(engine.search("key")) == 1


def test_age_change_supersedes(engine):
    store_keys(engine, extract_keys("i'm 16 years old"))
    store_keys(engine, extract_keys("i'm 17 years old"))

    active = engine.search("key")
    superseded = engine.search("key", status="superseded")

    assert any("17" in r.data["value"] for r in active)
    assert len(superseded) == 1
    assert "16" in superseded[0].data["value"]
    assert superseded[0].valid_until is not None


def test_relation_attribute_contradiction_supersedes(engine):
    store_keys(engine, extract_keys("my sister Maanvi lives in Austin"))
    store_keys(engine, extract_keys("my sister Maanvi lives in Dallas"))

    active = engine.search("key")
    superseded = engine.search("key", status="superseded")

    assert any("Dallas" in r.data["value"] for r in active)
    assert len(superseded) == 1


def test_different_people_coexist(engine):
    store_keys(engine, extract_keys("my sister Maanvi lives in Austin"))
    store_keys(engine, extract_keys("my cousin Dev lives in Dallas"))

    assert len(engine.search("key")) >= 2
    assert engine.search("key", status="superseded") == []


def test_multiple_allergies_coexist(engine):
    store_keys(engine, extract_keys("i'm allergic to peanuts"))
    store_keys(engine, extract_keys("i'm allergic to shellfish"))

    active = engine.search("key")
    assert len([r for r in active if "allergic" in r.data["value"]]) == 2
    assert engine.search("key", status="superseded") == []


def test_goals_accumulate(engine):
    store_keys(engine, extract_keys("i want to learn Spanish"))
    store_keys(engine, extract_keys("i want to run a marathon"))

    assert len(engine.search("key")) == 2


def test_supersede_writes_audit_row(engine):
    store_keys(engine, extract_keys("i live in Dallas"))
    store_keys(engine, extract_keys("i moved to Austin"))

    rows = engine.database.execute(
        """
        SELECT operation, reason FROM knowledge_changes
        WHERE operation = 'supersede'
        """
    ).fetchall()
    assert len(rows) == 1
    assert "Austin" in rows[0][1]


def test_supersede_cancels_old_birthday_reminder(engine):
    from datetime import timezone as tz
    from zoneinfo import ZoneInfo

    actions = _FakeActions()
    tzinfo = ZoneInfo("America/Chicago")

    store_keys(
        engine,
        extract_keys("my birthday is June 3"),
        actions_engine=actions,
        timezone=tzinfo,
    )
    assert len(actions.scheduled) == 1

    store_keys(
        engine,
        extract_keys("my birthday is August 9"),
        actions_engine=actions,
        timezone=tzinfo,
    )

    # old birthday key superseded -> its reminder cancelled
    superseded = engine.search("key", status="superseded")
    assert len(superseded) == 1
    assert "June" in superseded[0].data["value"]
    assert len(actions.cancelled) == 1
    assert actions.cancelled[0]["reason"] is not None
