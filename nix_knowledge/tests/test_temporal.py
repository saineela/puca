from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from nix_knowledge.temporal import TemporalResolver


TZ = ZoneInfo("America/Chicago")

# Fixed clock: Friday 2026-09-04, 10:00 local
NOW = datetime(2026, 9, 4, 10, 0, tzinfo=TZ)


@pytest.fixture
def resolver():
    return TemporalResolver("America/Chicago")


def date_of(resolver, expression):
    result = resolver.resolve(expression, now=NOW)
    assert result is not None, f"{expression!r} did not resolve"
    return result.start.date()


# ---------------------------------------------------------------------
# Bare weekday: next occurrence, today counts
# ---------------------------------------------------------------------

def test_bare_weekday_is_next_occurrence(resolver):
    # Friday Sep 4 -> next Thursday is Sep 10
    assert date_of(resolver, "thursday") == datetime(2026, 9, 10).date()


def test_bare_weekday_today_counts(resolver):
    # said on Friday -> this Friday
    assert date_of(resolver, "friday") == datetime(2026, 9, 4).date()


def test_on_weekday_same_as_bare(resolver):
    assert date_of(resolver, "on thursday") == datetime(2026, 9, 10).date()


# ---------------------------------------------------------------------
# this week / next week qualifiers (Monday-based)
# ---------------------------------------------------------------------

def test_this_week_thursday_is_literal(resolver):
    # week of Aug 31-Sep 6 -> Thursday is Sep 3 (in the past!)
    assert date_of(resolver, "this week thursday") == datetime(2026, 9, 3).date()


def test_next_week_thursday(resolver):
    # following week: Sep 7-13 -> Thursday Sep 10
    assert date_of(resolver, "next week thursday") == datetime(2026, 9, 10).date()


def test_next_thursday_following_week(resolver):
    assert date_of(resolver, "next thursday") == datetime(2026, 9, 10).date()


def test_thursday_next_week_word_order(resolver):
    assert date_of(resolver, "thursday next week") == datetime(2026, 9, 10).date()


# ---------------------------------------------------------------------
# Month-name dates
# ---------------------------------------------------------------------

def test_month_day_this_year(resolver):
    assert date_of(resolver, "september 12th") == datetime(2026, 9, 12).date()


def test_month_day_with_year(resolver):
    assert date_of(resolver, "september 12 2027") == datetime(2027, 9, 12).date()


def test_month_day_rolls_to_next_year_when_past(resolver):
    # Sep 4 -> january 5th already passed -> next January
    assert date_of(resolver, "january 5th") == datetime(2027, 1, 5).date()


def test_day_month_word_order(resolver):
    assert date_of(resolver, "12 september 2027") == datetime(2027, 9, 12).date()


def test_the_of_form(resolver):
    assert date_of(resolver, "the 12th of september") == datetime(2026, 9, 12).date()


def test_month_day_abbreviated(resolver):
    assert date_of(resolver, "sep 12") == datetime(2026, 9, 12).date()


def test_month_day_with_time(resolver):
    result = resolver.resolve("september 12th at 3pm", now=NOW)
    assert result is not None
    assert not result.all_day
    assert result.start.hour == 15
    assert result.start.date() == datetime(2026, 9, 12).date()


def test_month_day_leap_day_invalid_year_rolls(resolver):
    # Feb 29 2026 does not exist -> unresolvable (no year given would
    # roll to 2027, also invalid -> None is correct)
    assert resolver.resolve("february 29th", now=NOW) is None


def test_month_day_leap_day_valid_year(resolver):
    assert date_of(resolver, "february 29 2028") == datetime(2028, 2, 29).date()


# ---------------------------------------------------------------------
# Numeric and ISO dates
# ---------------------------------------------------------------------

def test_numeric_month_day(resolver):
    assert date_of(resolver, "9/12") == datetime(2026, 9, 12).date()


def test_numeric_full_date(resolver):
    assert date_of(resolver, "9/12/2027") == datetime(2027, 9, 12).date()


def test_numeric_invalid_month(resolver):
    assert resolver.resolve("13/12", now=NOW) is None


def test_iso_date(resolver):
    assert date_of(resolver, "2027-09-12") == datetime(2027, 9, 12).date()


def test_iso_date_with_time(resolver):
    result = resolver.resolve("2027-09-12 at 2pm", now=NOW)
    assert result is not None
    assert result.start.hour == 14


# ---------------------------------------------------------------------
# Regression: previous behaviors intact
# ---------------------------------------------------------------------

def test_tomorrow(resolver):
    assert date_of(resolver, "tomorrow") == datetime(2026, 9, 5).date()


def test_friday_with_time(resolver):
    result = resolver.resolve("friday at 6pm", now=NOW)
    assert result is not None
    assert result.start.date() == datetime(2026, 9, 4).date()
    assert result.start.hour == 18


def test_next_week_window(resolver):
    result = resolver.resolve("next week", now=NOW)
    assert result is not None
    assert result.start.date() == datetime(2026, 9, 7).date()
    assert result.end.date() == datetime(2026, 9, 13).date()


def test_this_weekend_window(resolver):
    result = resolver.resolve("this weekend", now=NOW)
    assert result is not None
    assert result.start.date() == datetime(2026, 9, 5).date()
    assert result.end.date() == datetime(2026, 9, 6).date()


def test_every_sunday_recurring(resolver):
    result = resolver.resolve("every sunday", now=NOW)
    assert result is not None
    assert result.recurring
    assert result.recurrence == "weekly"


def test_every_weekday_without_time_is_all_day(resolver):
    # NOW is Friday; "every monday" first occurs Monday Sep 7
    result = resolver.resolve("every monday", now=NOW)
    assert result is not None
    assert result.recurring
    assert result.recurrence == "weekly"
    assert result.all_day
    assert result.start.date() == datetime(2026, 9, 7).date()
    assert result.end.date() == datetime(2026, 9, 7).date()


def test_every_weekday_today_counts(resolver):
    result = resolver.resolve("every friday", now=NOW)
    assert result is not None
    assert result.all_day
    assert result.start.date() == datetime(2026, 9, 4).date()


def test_every_weekday_with_passed_time_rolls_to_next_week(resolver):
    result = resolver.resolve("every monday at 9am", now=NOW)
    assert result is not None
    assert result.recurring
    assert not result.all_day
    assert result.start.date() == datetime(2026, 9, 7).date()
    assert result.start.hour == 9


def test_every_day_without_time_is_all_day_today(resolver):
    result = resolver.resolve("every day", now=NOW)
    assert result is not None
    assert result.recurring
    assert result.recurrence == "daily"
    assert result.all_day
    assert result.start.date() == datetime(2026, 9, 4).date()


def test_in_days(resolver):
    result = resolver.resolve("in 3 days", now=NOW)
    assert result is not None
    assert result.start.date() == datetime(2026, 9, 7).date()


def test_noon_clock(resolver):
    result = resolver.resolve("friday at noon", now=NOW)
    assert result is not None
    assert result.start.hour == 12


# ---------------------------------------------------------------------
# Time ranges
# ---------------------------------------------------------------------

def test_time_range_today(resolver):
    # NOW is 10:00; 19:00-20:00 is still ahead today
    result = resolver.resolve("from 7pm to 8pm", now=NOW)
    assert result is not None
    assert not result.all_day
    assert result.start.date() == datetime(2026, 9, 4).date()
    assert result.start.hour == 19
    assert result.end.hour == 20


def test_time_range_rolls_to_tomorrow_when_passed(resolver):
    # the whole 7-8pm window is in the past at 22:00
    late = datetime(2026, 9, 4, 22, 0, tzinfo=TZ)
    result = resolver.resolve("from 7pm to 8pm", now=late)
    assert result is not None
    assert result.start.date() == datetime(2026, 9, 5).date()
    assert result.end.date() == datetime(2026, 9, 5).date()


def test_time_range_bare_clocks(resolver):
    result = resolver.resolve("7pm to 8:30pm", now=NOW)
    assert result is not None
    assert result.start.hour == 19
    assert result.end.hour == 20
    assert result.end.minute == 30


def test_time_range_until(resolver):
    result = resolver.resolve("from 9am until 5pm", now=NOW)
    assert result is not None
    assert result.start.hour == 9
    assert result.end.hour == 17


def test_today_time_range(resolver):
    result = resolver.resolve("today from 7pm to 8pm", now=NOW)
    assert result is not None
    assert result.start.date() == datetime(2026, 9, 4).date()
    assert result.start.hour == 19
    assert result.end.hour == 20


def test_tomorrow_time_range(resolver):
    result = resolver.resolve("tomorrow from 9am to 10am", now=NOW)
    assert result is not None
    assert result.start.date() == datetime(2026, 9, 5).date()
    assert result.start.hour == 9
    assert result.end.hour == 10


def test_weekday_time_range_next_occurrence(resolver):
    # Friday 10:00 -> this Friday 19:00 is still ahead
    result = resolver.resolve("friday from 7pm to 9pm", now=NOW)
    assert result is not None
    assert result.start.date() == datetime(2026, 9, 4).date()
    assert result.start.hour == 19
    assert result.end.hour == 21


def test_weekday_time_range_passed_rolls_forward(resolver):
    # Friday 22:00 -> Friday 7pm already passed -> next Friday Sep 11
    late = datetime(2026, 9, 4, 22, 0, tzinfo=TZ)
    result = resolver.resolve("friday from 7pm to 8pm", now=late)
    assert result is not None
    assert result.start.date() == datetime(2026, 9, 11).date()


def test_range_end_before_start_crosses_midnight(resolver):
    result = resolver.resolve("from 11pm to 1am", now=NOW)
    assert result is not None
    assert result.start.day == 4
    assert result.start.hour == 23
    assert result.end.day == 5
    assert result.end.hour == 1


def test_gibberish_is_none(resolver):
    assert resolver.resolve("gibberish nonsense", now=NOW) is None
