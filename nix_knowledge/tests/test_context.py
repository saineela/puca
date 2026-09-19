from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from nix_knowledge.context import TemporalContext
from nix_knowledge.engine import KnowledgeEngine
from nix_knowledge.functions.calendar import find_calendar_events


TZ = ZoneInfo("America/Chicago")


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "test.db"

    engine = KnowledgeEngine(db)

    yield engine

    engine.close()


@pytest.fixture
def context():
    return TemporalContext("America/Chicago")


class _FrozenContext(TemporalContext):
    """Clock frozen to a fixed hour so ranking tests are deterministic
    no matter what time of day the suite runs."""

    def now(self) -> datetime:
        base = datetime.now(self.timezone)
        return base.replace(hour=10, minute=0, second=0, microsecond=0)


def _create(engine, context, title, expression, **overrides):
    from nix_knowledge.temporal import TemporalResolver

    resolved = TemporalResolver("America/Chicago").resolve(expression)

    assert resolved is not None, f"{expression!r} did not resolve"

    data = {
        "title": title,
        "start": resolved.start.isoformat(),
        "end": resolved.end.isoformat(),
        "all_day": resolved.all_day,
        "temporal_expression": expression,
        "recurring": resolved.recurring,
        "recurrence": resolved.recurrence,
        "status": "scheduled",
    }

    data.update(overrides)

    record = engine.create("calendar_event", data)

    return record.id


# ---------------------------------------------------------------------
# Window matching
# ---------------------------------------------------------------------


def test_today_window_matches_only_today(engine, context):
    # "next week" and a fixed future date can never overlap today,
    # whatever weekday it is
    _create(engine, context, "today thing", "today")
    _create(engine, context, "next week thing", "next week")
    _create(engine, context, "far thing", "september 12 2027")

    result = find_calendar_events(
        engine,
        window="today",
        context=context,
    )

    titles = [event["title"] for event in result["events"]]

    assert titles == ["today thing"]
    assert result["window"]["expression"] == "today"


def test_window_matches_overlapping_events(engine, context):
    # today's bounds always overlap "this week" (Monday-based window
    # containing today) and never overlap "next week"
    _create(engine, context, "today thing", "today")

    this_week = find_calendar_events(
        engine,
        window="this week",
        context=context,
    )

    assert this_week["count"] == 1

    next_week = find_calendar_events(
        engine,
        window="next week",
        context=context,
    )

    assert next_week["count"] == 0


def test_weekend_window_matches_weekend_event(engine, context):
    # build the event directly from the weekend window so the test is
    # independent of which weekday "today" is
    weekend = context.window("this_weekend")

    engine.create(
        "calendar_event",
        {
            "title": "weekend thing",
            "start": weekend.start.isoformat(),
            "end": weekend.end.isoformat(),
            "all_day": True,
            "temporal_expression": "this weekend",
            "status": "scheduled",
        },
    )

    result = find_calendar_events(
        engine,
        window="this weekend",
        context=context,
    )

    assert result["count"] == 1
    assert result["events"][0]["title"] == "weekend thing"


def test_bad_window_is_structured_error(engine, context):
    result = find_calendar_events(
        engine,
        window="gibberish nonsense",
        context=context,
    )

    assert result["ok"] is False
    assert "window" in result["error"]


# ---------------------------------------------------------------------
# Recurring occurrences
# ---------------------------------------------------------------------


def test_recurring_event_expands_into_window(engine, context):
    _create(engine, context, "ftc meeting", "every monday")

    result = find_calendar_events(
        engine,
        window="next week",
        context=context,
    )

    assert result["count"] == 1

    occurrence = result["events"][0]

    assert occurrence["title"] == "ftc meeting"
    assert occurrence["occurrence"] is True
    assert occurrence["all_day"] is True
    assert occurrence["recurring"] is True

    start = datetime.fromisoformat(occurrence["start"])
    assert start.weekday() == 0  # monday


def test_recurring_daily_expands_per_day(engine, context):
    _create(engine, context, "standup", "every day")

    result = find_calendar_events(
        engine,
        window="next week",
        context=context,
    )

    assert result["count"] == 7
    assert all(
        event["occurrence"] for event in result["events"]
    )


def test_recurring_series_without_window_returns_anchor(engine, context):
    _create(engine, context, "ftc meeting", "every monday")

    result = find_calendar_events(engine, context=context)

    assert result["count"] == 1
    assert result["events"][0]["occurrence"] is False


# ---------------------------------------------------------------------
# Status and ranking
# ---------------------------------------------------------------------


def test_cancelled_excluded_from_browse_included_in_title_lookup(
    engine,
    context,
):
    _create(
        engine,
        context,
        "tsa meeting",
        "today",
        status="cancelled",
    )
    # "next week" never overlaps today, whatever weekday it is
    _create(engine, context, "pizza party", "next week")

    browse = find_calendar_events(engine, window="today", context=context)
    assert browse["count"] == 0

    lookup = find_calendar_events(
        engine,
        title="tsa meeting",
        context=context,
    )
    assert lookup["count"] == 1
    assert lookup["events"][0]["status"] == "cancelled"


def test_results_ranked_by_relevance(engine):
    # Explicit offsets from a frozen clock make the ranking assertion
    # deterministic. (The original used natural expressions resolved
    # on the live clock: after ~9pm, "tomorrow" = midnight tonight is
    # nearer than "in 3 hours", flipping the expected order.)
    context = _FrozenContext("America/Chicago")
    now = context.now()

    def _event(title, start):
        return engine.create(
            "calendar_event",
            {
                "title": title,
                "start": start.isoformat(),
                "end": (start + timedelta(hours=1)).isoformat(),
                "all_day": False,
                "temporal_expression": "custom",
                "recurring": False,
                "recurrence": "none",
                "status": "scheduled",
            },
        )

    _event("far thing", now + timedelta(days=365))
    _event("soon thing", now + timedelta(hours=34))
    _event("near thing", now + timedelta(hours=3))

    result = find_calendar_events(engine, context=context)

    titles = [event["title"] for event in result["events"]]

    assert titles == ["near thing", "soon thing", "far thing"]


def test_entries_carry_human_when_labels(engine, context):
    _create(engine, context, "timed thing", "tomorrow at 6pm")

    result = find_calendar_events(engine, context=context)

    assert result["events"][0]["when"] == "tomorrow 6pm"


# ---------------------------------------------------------------------
# TemporalContext primitives
# ---------------------------------------------------------------------


def test_window_names_resolve(context):
    for name in [
        "today",
        "tomorrow",
        "yesterday",
        "this_week",
        "next_week",
        "last_week",
        "this_month",
        "this_weekend",
        "upcoming",
        "recent",
    ]:
        window = context.window(name)

        assert window.start <= window.end


def test_unknown_window_raises(context):
    with pytest.raises(ValueError):
        context.window("gibberish")


def test_parse_window_falls_back_to_resolver(context):
    window = context.parse_window("on friday")

    assert window is not None
    assert window.start.date().weekday() == 4


def test_occurrence_never_precedes_anchor(context):
    anchor = datetime(2026, 9, 7, 14, 0, tzinfo=TZ)

    occurrence = context.occurrence(
        start=anchor,
        recurrence="weekly",
        target_date=datetime(2026, 9, 5, tzinfo=TZ).date(),
    )

    assert occurrence is None


def test_relative_labels(context):
    now = context.now()

    assert context.relative_label(now) == "now"
    assert context.relative_label(now + timedelta(days=1)) == "tomorrow"
    assert context.relative_label(now - timedelta(days=1)) == "yesterday"
