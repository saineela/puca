from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from nix_knowledge.temporal import TemporalResolver
from nix_knowledge.temporal_hybrid import (
    parse_event,
    repair_calendar_arguments,
    validate_calendar_update_arguments,
)


TZ = ZoneInfo("America/Chicago")
NOW = datetime(2026, 9, 20, 8, 0, tzinfo=TZ)


@pytest.fixture
def resolver():
    return TemporalResolver("America/Chicago")


# Each case represents the structure a neural selector may propose, while
# the expected result is decided by symbolic date/time rules.
@pytest.mark.parametrize(
    "utterance, expected_date, expected_start, expected_end",
    [
        (
            "I have robotics practice 5 days after today during the morning time from 9am to 11am",
            "2026-09-25",
            (9, 0),
            (11, 0),
        ),
        (
            "schedule a dentist visit in five days at 3pm",
            "2026-09-25",
            (15, 0),
            (15, 0),
        ),
        (
            "add team practice two weeks from today in the evening",
            "2026-10-04",
            (17, 0),
            (22, 0),
        ),
        (
            "book a review tomorrow from 9am to 10:30am",
            "2026-09-21",
            (9, 0),
            (10, 30),
        ),
    ],
)
def test_hybrid_complex_temporal_benchmark(
    resolver,
    utterance,
    expected_date,
    expected_start,
    expected_end,
):
    parsed = parse_event(utterance, resolver, now=NOW)
    assert parsed is not None, utterance
    assert parsed.resolved.start.date().isoformat() == expected_date
    assert (parsed.resolved.start.hour, parsed.resolved.start.minute) == expected_start
    assert (parsed.resolved.end.hour, parsed.resolved.end.minute) == expected_end
    assert parsed.slots["resolved_start"].startswith(expected_date)
    assert parsed.title
    assert parsed.expression


def test_hybrid_repairs_partial_neural_proposal(resolver):
    request = (
        "I have an RObotics practice 5 days after today during the morning time "
        "from 9am to 11am"
    )
    partial = {
        "title": "robotics practice 5 days after today during the morning time",
        "temporal_expression": "from 9am to 11am",
    }

    arguments, parsed = repair_calendar_arguments(
        request,
        partial,
        resolver,
        now=NOW,
    )

    assert parsed is not None
    assert parsed.source == "symbolic_repair"
    assert arguments == {
        "title": "RObotics practice",
        "temporal_expression": (
            "5 days after today during the morning time from 9am to 11am"
        ),
    }
    assert parsed.resolved.start.date() == datetime(2026, 9, 25).date()
    assert parsed.resolved.start.hour == 9
    assert parsed.resolved.end.hour == 11
    assert parsed.slots["offset_amount"] == "5"
    assert parsed.slots["offset_relation"] == "after"
    assert parsed.slots["day_part"] == "morning"


def test_hybrid_rejects_temporal_residue_in_neural_title(resolver):
    parsed = parse_event(
        "I have robotics practice 5 days after today",
        resolver,
        proposal={
            "title": "robotics practice 5 days after today",
            "temporal_expression": "today",
        },
        now=NOW,
    )
    assert parsed is not None
    assert parsed.title == "robotics practice"
    assert parsed.expression == "5 days after today"
    assert parsed.resolved.start.date() == datetime(2026, 9, 25).date()


def test_hybrid_does_not_guess_unresolved_temporal_language(resolver):
    parsed = parse_event(
        "I have robotics practice sometime after the thing",
        resolver,
        proposal={
            "title": "robotics practice",
            "temporal_expression": "sometime after the thing",
        },
        now=NOW,
    )
    assert parsed is None


def test_hybrid_parses_casual_tsa_meeting_with_tomorrow_time_range(resolver):
    request = "alright bro, I have a TSA meeting tmr from 4pm to 6pm"

    parsed = parse_event(request, resolver, now=NOW)

    assert parsed is not None
    assert parsed.title == "TSA meeting"
    assert parsed.expression == "tmr from 4pm to 6pm"
    assert parsed.resolved.start.isoformat() == "2026-09-21T16:00:00-05:00"
    assert parsed.resolved.end.isoformat() == "2026-09-21T18:00:00-05:00"


def test_hybrid_preserves_simple_symbolic_request(resolver):
    parsed = parse_event(
        "I have a dentist appointment tomorrow",
        resolver,
        proposal={
            "title": "dentist appointment",
            "temporal_expression": "tomorrow",
        },
        now=NOW,
    )
    assert parsed is not None
    assert parsed.source == "neural_validated"
    assert parsed.resolved.start.date() == datetime(2026, 9, 21).date()


def test_hybrid_validates_update_without_changing_event_identity(resolver):
    arguments, parsed = validate_calendar_update_arguments(
        "move robotics practice to 5 days after today during the morning from 9am to 11am",
        {
            "title": "robotics practice",
            "new_temporal_expression": "from 9am to 11am",
        },
        resolver,
        now=NOW,
    )
    assert parsed is not None
    assert parsed.source == "symbolic_repair"
    assert arguments["title"] == "robotics practice"
    assert arguments["new_temporal_expression"] == (
        "5 days after today during the morning from 9am to 11am"
    )
    assert parsed.resolved.start.date() == datetime(2026, 9, 25).date()
    assert parsed.resolved.start.hour == 9
    assert parsed.resolved.end.hour == 11


def test_hybrid_rejects_invalid_update_before_mutation(resolver):
    arguments, parsed = validate_calendar_update_arguments(
        "move robotics practice to sometime after the thing",
        {
            "title": "robotics practice",
            "new_temporal_expression": "sometime after the thing",
        },
        resolver,
        now=NOW,
    )
    assert parsed is None
    assert arguments["new_temporal_expression"] == "sometime after the thing"
