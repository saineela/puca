from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from nix_knowledge.rules import decompose, route
from nix_knowledge.temporal import TemporalResolver


TZ = ZoneInfo("America/Chicago")

NOW = datetime(2026, 9, 4, 10, 0, tzinfo=TZ)


@pytest.fixture
def resolver():
    return TemporalResolver("America/Chicago")


# ---------------------------------------------------------------------
# Compound requests -> independent subtasks
# ---------------------------------------------------------------------


def test_two_creates(resolver):
    result = decompose(
        "I have a dentist appointment tomorrow and a meeting with bob on friday",
        resolver,
    )

    assert result is not None
    assert len(result) == 2

    first, second = result

    assert first[0] == "create_calendar_event"
    assert first[1]["title"] == "dentist appointment"
    assert first[1]["temporal_expression"] == "tomorrow"

    assert second[0] == "create_calendar_event"
    assert second[1]["title"] == "meeting with bob"
    assert second[1]["temporal_expression"] == "on friday"


def test_cancel_then_reschedule(resolver):
    result = decompose(
        "cancel my haircut and reschedule standup to monday at 9am",
        resolver,
    )

    assert result is not None
    assert result[0][0] == "cancel_calendar_event"
    assert result[0][1] == {"title": "haircut"}

    assert result[1][0] == "update_calendar_event"
    assert result[1][1]["title"] == "standup"
    assert result[1][1]["new_temporal_expression"] == "monday at 9am"


def test_semicolons_split(resolver):
    result = decompose(
        "cancel my haircut; reschedule standup to monday at 9am",
        resolver,
    )

    assert result is not None
    assert len(result) == 2


def test_mixed_fact_and_event(resolver):
    result = decompose(
        "remember that I like tea and I have a dentist appointment tomorrow",
        resolver,
    )

    assert result is not None

    names = {name for name, _ in result}

    assert names == {"create_fact", "create_calendar_event"}


def test_single_clause_is_not_decomposed(resolver):
    assert (
        decompose(
            "I have a dentist appointment tomorrow",
            resolver,
        )
        is None
    )


def test_unroutable_part_defers_everything(resolver):
    # "something else" routes nowhere even with the lead-in retry;
    # no safe decomposition exists
    assert (
        decompose(
            "cancel my haircut and something else entirely",
            resolver,
        )
        is None
    )


def test_lists_of_items_stay_together(resolver):
    # "bob and alice" is one title, not two tasks
    routed = route(
        "I have a meeting with bob and alice tomorrow",
        resolver,
    )

    assert routed is not None
    name, arguments = routed

    assert name == "create_calendar_event"
    assert arguments["title"] == "meeting with bob and alice"
    assert arguments["temporal_expression"] == "tomorrow"


def test_compound_leading_temporal(resolver):
    result = decompose(
        "This weekend I have pizza party and tomorrow I have dentist",
        resolver,
    )

    assert result is not None

    pizza = result[0]
    dentist = result[1]

    assert pizza[1]["temporal_expression"] == "this weekend"
    assert dentist[1]["temporal_expression"] == "tomorrow"


# ---------------------------------------------------------------------
# Preference clauses: comma-separated independent stances
# ---------------------------------------------------------------------


def test_comma_separated_stances(resolver):
    routed = route(
        "I love programming, i hate robotics and electrical is fun",
        resolver,
    )

    assert routed is not None
    name, arguments = routed

    assert name == "create_fact"
    assert arguments["value"] == (
        "I love programming; I hate robotics; electrical is fun"
    )


# ---------------------------------------------------------------------
# Move guard: leftover temporal words defer to the model
# ---------------------------------------------------------------------


def test_move_with_leftover_temporal_defers(resolver):
    assert (
        route(
            "move standup to monday at 9am and gym to tuesday",
            resolver,
        )
        is None
    )


# ---------------------------------------------------------------------
# Temporal grammar: fuzzy durations and month-only dates
# ---------------------------------------------------------------------


def test_in_a_couple_hours(resolver):
    result = resolver.resolve("in a couple hours", now=NOW)

    assert result is not None
    assert result.start.hour == 12


def test_in_a_few_days(resolver):
    result = resolver.resolve("in a few days", now=NOW)

    assert result is not None
    assert result.start.date() == datetime(2026, 9, 7).date()


def test_in_several_weeks(resolver):
    result = resolver.resolve("in several weeks", now=NOW)

    assert result is not None
    assert result.start.date() == datetime(2026, 10, 2).date()


def test_next_june_month_window(resolver):
    result = resolver.resolve("next june", now=NOW)

    assert result is not None
    assert result.all_day
    assert result.start.date() == datetime(2027, 6, 1).date()
    assert result.end.date() == datetime(2027, 6, 30).date()


def test_last_december_month_window(resolver):
    result = resolver.resolve("last december", now=NOW)

    assert result is not None
    assert result.start.date() == datetime(2025, 12, 1).date()
    assert result.end.date() == datetime(2025, 12, 31).date()


def test_in_may_month_window(resolver):
    result = resolver.resolve("in may", now=NOW)

    assert result is not None
    assert result.start.date() == datetime(2027, 5, 1).date()


def test_bare_may_is_not_a_date(resolver):
    # "may" alone is a modal verb
    assert resolver.resolve("may", now=NOW) is None


def test_this_september_month_window(resolver):
    result = resolver.resolve("this september", now=NOW)

    assert result is not None
    assert result.start.date() == datetime(2026, 9, 1).date()
    assert result.end.date() == datetime(2026, 9, 30).date()


def test_numeric_in_duration_still_works(resolver):
    result = resolver.resolve("in 3 days", now=NOW)

    assert result is not None
    assert result.start.date() == datetime(2026, 9, 7).date()
