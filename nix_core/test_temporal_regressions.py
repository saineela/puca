from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

sys.path.append(str(Path(__file__).resolve().parents[1] / "nix_knowledge" / "nix_knowledge"))

from brain import format_knowledge_result, is_assistant_identity_request, is_creator_identity_request
from rules import route
from temporal import TemporalResolver
from temporal_hybrid import parse_event, repair_calendar_arguments


TZ = ZoneInfo("America/Chicago")


class FrozenResolver(TemporalResolver):
    def now(self):
        return datetime(2026, 9, 20, 12, 0, tzinfo=self.timezone)


def test_tmr_schedule_query_is_a_tomorrow_window():
    resolver = FrozenResolver("America/Chicago")
    name, arguments = route("what is on my schedule for tmr", resolver)
    assert name == "find_calendar_events"
    assert arguments == {"window": "tmr"}


def test_calendar_typo_and_tmr_still_use_symbolic_window():
    resolver = FrozenResolver("America/Chicago")
    name, arguments = route("what do I have on my calender tmr", resolver)
    assert name == "find_calendar_events"
    assert arguments == {"window": "tmr"}


def test_weekend_clock_is_one_upcoming_occurrence():
    resolver = FrozenResolver("America/Chicago")
    parsed = resolver.resolve("this weekend at 7pm")
    assert parsed is not None
    assert parsed.start.date().isoformat() == "2026-09-20"
    assert parsed.start.hour == 19
    assert parsed.end == parsed.start


def test_in_more_days_and_around_clock_are_resolved_symbolically():
    resolver = FrozenResolver("America/Chicago")
    arguments, parsed = repair_calendar_arguments(
        "Hey Casper, I have a dinner with Angela and everyone near Galveston Beach in 2 more days around 4pm",
        {
            "title": "dinner with Angela",
            "temporal_expression": "tomorrow at 4pm",
        },
        resolver,
    )
    assert parsed is not None
    assert "everyone" in arguments["title"].lower()
    assert "galveston beach" in arguments["title"].lower()
    assert parsed.resolved.start.date().isoformat() == "2026-09-22"
    assert parsed.resolved.start.hour == 16


def test_partial_selector_does_not_drop_five_day_offset():
    resolver = FrozenResolver("America/Chicago")
    arguments, parsed = repair_calendar_arguments(
        "I have an RObotics practice 5 days after today during the morning time from 9am to 11am",
        {
            "title": "robotics practice 5 days after today during the morning time",
            "temporal_expression": "from 9am to 11am",
        },
        resolver,
    )
    assert parsed is not None
    assert arguments["title"].lower() == "robotics practice"
    assert arguments["temporal_expression"] == (
        "5 days after today during the morning time from 9am to 11am"
    )
    assert parsed.resolved.start.date().isoformat() == "2026-09-25"
    assert parsed.resolved.start.hour == 9
    assert parsed.resolved.end.hour == 11


def test_selector_filler_does_not_pollute_event_title():
    resolver = TemporalResolver(timezone="America/Chicago")
    parsed = parse_event(
        "Schedule my dentist appointment which is happening in 2 days",
        resolver,
        proposal={
            "title": "dentist appointment which is happening",
            "temporal_expression": "in 2 days",
        },
    )
    assert parsed is not None
    assert parsed.title == "dentist appointment"
    assert parsed.slots["offset_amount"] == "2"


def test_event_baseline_exposes_absolute_local_dates():
    reply = format_knowledge_result(
        {
            "result": {
                "ok": True,
                "count": 1,
                "events": [
                    {
                        "title": "FRC meeting",
                        "when": "tomorrow 5pm-9pm",
                        "start_local": "Monday, September 21, 2026 at 5:00:00 PM",
                        "end_local": "Monday, September 21, 2026 at 9:00:00 PM",
                    }
                ],
            }
        }
    )
    assert "Monday, September 21, 2026" in reply
    assert "tomorrow 5pm-9pm" not in reply


def test_identity_variants_bypass_model():
    assert is_creator_identity_request("Who created you casper")
    assert is_creator_identity_request("Who made you?")
    assert is_assistant_identity_request("Who are you again?")
    assert is_assistant_identity_request("bro who are you")
