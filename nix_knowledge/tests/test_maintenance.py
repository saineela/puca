from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from nix_knowledge.engine import KnowledgeEngine
from nix_knowledge.maintenance import TemporalMaintenance


TZ = ZoneInfo("America/Chicago")

# Anchored on a fixed moment: Friday 2026-09-04, 10:00 local
NOW = datetime(2026, 9, 4, 10, 0, tzinfo=TZ)


def make_event(engine, **data):
    return engine.create("calendar_event", data)


@pytest.fixture
def engine(tmp_path):
    e = KnowledgeEngine(tmp_path / "test.db")
    yield e
    e.close()


@pytest.fixture
def maintenance(engine):
    return TemporalMaintenance(engine, timezone="America/Chicago")


def status_of(engine, record_id):
    return engine.get(record_id).data.get("status")


# ---------------------------------------------------------------------
# Single-day events: expire at the end of that day
# ---------------------------------------------------------------------

def test_tomorrow_event_expires_after_that_day(engine, maintenance):
    record = make_event(
        engine,
        title="dentist",
        start="2026-09-05T00:00:00-05:00",   # Saturday
        end="2026-09-05T23:59:59.999999-05:00",
        all_day=True,
        temporal_expression="tomorrow",
    )

    # still Friday: not expired
    summary = maintenance.expire_due(now=NOW)
    assert status_of(engine, record.id) != "expired"

    # Sunday 00:30 (after Saturday ended): expired
    sunday = datetime(2026, 9, 6, 0, 30, tzinfo=TZ)
    summary = maintenance.expire_due(now=sunday)

    assert summary["expired_count"] == 1
    assert status_of(engine, record.id) == "expired"


def test_event_same_day_is_not_expired(engine, maintenance):
    record = make_event(
        engine,
        title="today thing",
        start="2026-09-04T09:00:00-05:00",  # today at 9, now is 10
        end=None,
        all_day=False,
        temporal_expression="today at 9am",
    )

    maintenance.expire_due(now=NOW)

    assert status_of(engine, record.id) != "expired"


# ---------------------------------------------------------------------
# Weekend events: expire at end of Sunday
# ---------------------------------------------------------------------

def test_weekend_event_expires_after_sunday(engine, maintenance):
    record = make_event(
        engine,
        title="robotics meetup",
        start="2026-09-05T12:00:00-05:00",   # Saturday
        end=None,
        all_day=False,
        temporal_expression="this weekend",
    )

    # Saturday 20:00 - weekend still ongoing
    saturday_evening = datetime(2026, 9, 5, 20, 0, tzinfo=TZ)
    maintenance.expire_due(now=saturday_evening)
    assert status_of(engine, record.id) != "expired"

    # Monday 00:30 - weekend over
    monday = datetime(2026, 9, 7, 0, 30, tzinfo=TZ)
    summary = maintenance.expire_due(now=monday)

    assert summary["expired_count"] == 1
    assert status_of(engine, record.id) == "expired"


# ---------------------------------------------------------------------
# Timed events: expire after the moment passes (midnight of that day)
# ---------------------------------------------------------------------

def test_timed_event_expires_after_its_day(engine, maintenance):
    record = make_event(
        engine,
        title="meeting with Bob",
        start="2026-09-04T15:00:00-05:00",   # today 3pm
        end=None,
        all_day=False,
        temporal_expression="today at 3pm",
    )

    # 16:00 same day: still the same calendar day -> kept
    after = datetime(2026, 9, 4, 16, 0, tzinfo=TZ)
    maintenance.expire_due(now=after)
    assert status_of(engine, record.id) != "expired"

    # next day: expired
    next_day = datetime(2026, 9, 5, 0, 30, tzinfo=TZ)
    maintenance.expire_due(now=next_day)
    assert status_of(engine, record.id) == "expired"


# ---------------------------------------------------------------------
# Events with explicit end (duration): expire at end, not midnight
# ---------------------------------------------------------------------

def test_event_with_explicit_end_uses_end(engine, maintenance):
    record = make_event(
        engine,
        title="conference",
        start="2026-09-04T09:00:00-05:00",
        end="2026-09-06T17:00:00-05:00",     # runs until Sunday 5pm
        all_day=False,
        temporal_expression="this weekend",
    )

    sunday_afternoon = datetime(2026, 9, 6, 16, 0, tzinfo=TZ)
    maintenance.expire_due(now=sunday_afternoon)
    assert status_of(engine, record.id) != "expired"

    sunday_evening = datetime(2026, 9, 6, 18, 0, tzinfo=TZ)
    maintenance.expire_due(now=sunday_evening)
    assert status_of(engine, record.id) == "expired"


# ---------------------------------------------------------------------
# Recurring events never expire
# ---------------------------------------------------------------------

def test_recurring_event_never_expires(engine, maintenance):
    record = make_event(
        engine,
        title="robotics",
        start="2026-08-30T07:00:00-05:00",   # a past Sunday
        end="2026-08-30T07:00:00-05:00",
        all_day=False,
        temporal_expression="every sunday",
        recurring=True,
        recurrence="weekly",
    )

    far_future = NOW + timedelta(days=30)
    summary = maintenance.expire_due(now=far_future)

    assert status_of(engine, record.id) != "expired"
    assert summary["expired_count"] == 0


# ---------------------------------------------------------------------
# Cancelled events are still expired by the runner
# ---------------------------------------------------------------------

def test_cancelled_event_is_expired(engine, maintenance):
    record = make_event(
        engine,
        title="party",
        start="2026-09-01T20:00:00-05:00",   # past Tuesday
        end=None,
        all_day=False,
        temporal_expression="tuesday",
        status="cancelled",
    )

    summary = maintenance.expire_due(now=NOW)

    assert summary["expired_count"] == 1
    assert status_of(engine, record.id) == "expired"


# ---------------------------------------------------------------------
# Midnight nap calculation
# ---------------------------------------------------------------------

def test_seconds_until_next_midnight(maintenance):
    import math

    just_before = datetime(2026, 9, 4, 23, 59, 0, tzinfo=TZ)
    seconds = maintenance.seconds_until_next_midnight()
    assert 0 < seconds <= 24 * 3600

    # deterministic check with a manual instance math
    assert math.isclose(
        (datetime(2026, 9, 5, 0, 0, tzinfo=TZ) - just_before).total_seconds(),
        60,
    )
