from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from nix_knowledge.rules import route
from nix_knowledge.temporal import TemporalResolver


TZ = ZoneInfo("America/Chicago")

# Fixed clock: Friday 2026-09-04, 10:00 local (matches test_temporal)
NOW = datetime(2026, 9, 4, 10, 0, tzinfo=TZ)


@pytest.fixture
def resolver():
    return TemporalResolver("America/Chicago")


def names_of(resolver, request):
    routed = route(request, resolver)
    return routed


# ---------------------------------------------------------------------
# Preference statements -> create_fact
# ---------------------------------------------------------------------


def test_hate_but_love_becomes_two_facts(resolver):
    routed = route("I hate robotics, but love programming", resolver)

    assert routed is not None
    name, arguments = routed

    assert name == "create_fact"
    assert arguments["value"] == (
        "I hate robotics; I love programming"
    )


def test_plain_like_statement(resolver):
    routed = route("I love electrical engineering", resolver)

    assert routed is not None
    name, arguments = routed

    assert name == "create_fact"
    assert arguments["value"] == "I love electrical engineering"


def test_dont_like_statement(resolver):
    routed = route("I dont like programming", resolver)

    assert routed is not None
    name, arguments = routed

    assert name == "create_fact"
    assert arguments["value"] == "I don't like programming"


def test_dont_like_but_is_fun(resolver):
    routed = route(
        "I dont like programming but electrical is fun",
        resolver,
    )

    assert routed is not None
    name, arguments = routed

    assert name == "create_fact"
    assert arguments["value"] == (
        "I don't like programming; electrical is fun"
    )


def test_negated_event_statement_is_not_a_preference(resolver):
    # "dont" targets "have", not a stance verb: not a preference
    routed = route("I dont have a meeting today", resolver)

    assert routed is None


# ---------------------------------------------------------------------
# "when is my X" -> find_calendar_events
# ---------------------------------------------------------------------


def test_when_is_lookup(resolver):
    routed = route("When is my TSA meeting", resolver)

    assert routed is not None
    name, arguments = routed

    assert name == "find_calendar_events"
    assert arguments == {"title": "tsa meeting"}


def test_what_time_is_lookup(resolver):
    routed = route("what time is the dentist appointment", resolver)

    assert routed is not None
    name, arguments = routed

    assert name == "find_calendar_events"
    assert arguments == {"title": "dentist appointment"}


# ---------------------------------------------------------------------
# Temporal-first create phrasing
# ---------------------------------------------------------------------


def test_this_weekend_leading_clause(resolver):
    routed = route(
        "This weekend I have pizza party with my friends",
        resolver,
    )

    assert routed is not None
    name, arguments = routed

    assert name == "create_calendar_event"
    assert arguments["title"] == "pizza party with my friends"
    assert arguments["temporal_expression"] == "this weekend"


def test_typo_ridden_weekend_request(resolver):
    # the exact transcript phrasing ("ym friends")
    routed = route(
        "This weekend I have pizza party with ym friends",
        resolver,
    )

    assert routed is not None
    name, arguments = routed

    assert name == "create_calendar_event"
    assert arguments["title"] == "pizza party with ym friends"
    assert arguments["temporal_expression"] == "this weekend"


def test_tomorrow_leading_clause(resolver):
    routed = route(
        "tomorrow I have a dentist appointment",
        resolver,
    )

    assert routed is not None
    name, arguments = routed

    assert name == "create_calendar_event"
    assert arguments["title"] == "dentist appointment"
    assert arguments["temporal_expression"] == "tomorrow"


def test_trailing_expression_beats_leading_clause(resolver):
    routed = route(
        "this weekend i have pizza party on saturday",
        resolver,
    )

    assert routed is not None
    name, arguments = routed

    assert name == "create_calendar_event"
    assert arguments["temporal_expression"] == "on saturday"


# ---------------------------------------------------------------------
# Time-range creates through the full pipeline
# ---------------------------------------------------------------------


def test_time_range_create(resolver):
    routed = route(
        "I have my FRC presentation today from 7pm to 8pm",
        resolver,
    )

    assert routed is not None
    name, arguments = routed

    assert name == "create_calendar_event"
    assert arguments["title"] == "frc presentation"
    assert arguments["temporal_expression"] == "today from 7pm to 8pm"

    resolved = resolver.resolve(arguments["temporal_expression"], now=NOW)
    assert resolved is not None
    assert resolved.start.hour == 19
    assert resolved.end.hour == 20
    assert not resolved.all_day


def test_frc_presentation_without_today(resolver):
    routed = route(
        "I have my FRC presentation from 7pm to 8pm",
        resolver,
    )

    assert routed is not None
    name, arguments = routed

    assert name == "create_calendar_event"
    assert arguments["temporal_expression"] == "from 7pm to 8pm"


# ---------------------------------------------------------------------
# Regressions: previously routed forms stay intact
# ---------------------------------------------------------------------


def test_plain_create_with_suffix(resolver):
    routed = route("I have a dentist appointment tomorrow", resolver)

    assert routed is not None
    name, arguments = routed

    assert name == "create_calendar_event"
    assert arguments == {
        "title": "dentist appointment",
        "temporal_expression": "tomorrow",
    }


def test_recurring_create(resolver):
    routed = route("I have FTC meeting every monday", resolver)

    assert routed is not None
    name, arguments = routed

    assert name == "create_calendar_event"
    assert arguments == {
        "title": "ftc meeting",
        "temporal_expression": "every monday",
    }


def test_recall_about(resolver):
    routed = route("what do you remember about robotics", resolver)

    assert routed is not None
    name, arguments = routed

    assert name == "find_facts"
    assert arguments == {"query": "robotics"}


def test_cancel(resolver):
    routed = route("cancel my tsa meeting", resolver)

    assert routed is not None
    name, arguments = routed

    assert name == "cancel_calendar_event"
    assert arguments == {"title": "tsa meeting"}


def test_bare_calendar(resolver):
    routed = route("what is on my calendar", resolver)

    assert routed is not None
    name, arguments = routed

    assert name == "find_calendar_events"
    assert arguments == {}


# ---------------------------------------------------------------------
# Windowed calendar browsing and existence probes
# ---------------------------------------------------------------------


def test_windowed_events_today(resolver):
    routed = route("what events do I have today", resolver)

    assert routed is not None
    name, arguments = routed

    assert name == "find_calendar_events"
    assert arguments == {"window": "today"}


def test_windowed_schedule_this_week(resolver):
    routed = route("what's on my schedule this week", resolver)

    assert routed is not None
    name, arguments = routed

    assert name == "find_calendar_events"
    assert arguments == {"window": "this week"}


def test_windowed_browse_weekday(resolver):
    routed = route("what do i have friday", resolver)

    assert routed is not None
    name, arguments = routed

    assert name == "find_calendar_events"
    assert arguments == {"window": "friday"}


def test_windowed_probe_tomorrow(resolver):
    routed = route("do I have anything tomorrow", resolver)

    assert routed is not None
    name, arguments = routed

    assert name == "find_calendar_events"
    assert arguments == {"window": "tomorrow"}


def test_windowed_probe_on_friday(resolver):
    routed = route("is there anything on friday", resolver)

    assert routed is not None
    name, arguments = routed

    assert name == "find_calendar_events"
    assert arguments == {"window": "friday"}


def test_bare_weekday_browse_falls_to_model(resolver):
    # "what do i have" with no time frame is not lexically
    # unambiguous enough for the deterministic layer
    routed = route("what do i have", resolver)

    assert routed is None


def test_empty_request_is_deferred(resolver):
    assert route("", resolver) is None


def test_gibberish_is_deferred_to_model(resolver):
    assert route("the sky is quite blue today", resolver) is None
