from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from nix_actions.engine import ActionsEngine


TZ = ZoneInfo("America/Chicago")


@pytest.fixture
def engine(tmp_path):
    e = ActionsEngine(tmp_path / "actions.db", timezone="America/Chicago")
    yield e
    e.close()


def test_schedule_requires_valid_type(engine):
    now = datetime.now(TZ)

    with pytest.raises(ValueError):
        engine.schedule(
            action_type="explode",
            scheduled_for=now,
        )


def test_schedule_and_fire(engine):
    now = datetime.now(TZ)

    action = engine.schedule(
        action_type="alarm",
        scheduled_for=now - timedelta(seconds=1),
        payload={"message": "wake up"},
    )

    fired = engine.run_due()

    assert len(fired) == 1
    assert fired[0].id == action.id
    assert fired[0].status == "fired"


def test_future_action_does_not_fire(engine):
    now = datetime.now(TZ)

    engine.schedule(
        action_type="reminder",
        scheduled_for=now + timedelta(hours=1),
        payload={"message": "later"},
    )

    assert engine.run_due() == []


def test_weekly_recurrence_materializes_next(engine):
    now = datetime.now(TZ)

    first = engine.schedule(
        action_type="alarm",
        scheduled_for=now - timedelta(seconds=1),
        payload={"message": "robotics class"},
        recurrence="weekly",
    )

    fired = engine.run_due()
    assert [a.id for a in fired] == [first.id]

    pending = engine.list_actions(status="pending")
    assert len(pending) == 1

    nxt = pending[0]
    assert nxt.scheduled_for == (
        first.scheduled_for + timedelta(weeks=1)
    )
    assert nxt.payload == first.payload
    assert nxt.source_record_id == first.source_record_id


def test_recurrence_end_stops_chain(engine):
    now = datetime.now(TZ)

    engine.schedule(
        action_type="alarm",
        scheduled_for=now - timedelta(days=8),
        payload={"message": "class"},
        recurrence="weekly",
        recurrence_end=now - timedelta(days=1),
    )

    fired = engine.run_due()

    assert len(fired) == 1
    # next occurrence would exceed recurrence_end -> nothing pending
    assert engine.list_actions(status="pending") == []


def test_cancel_by_source_record(engine):
    now = datetime.now(TZ)

    engine.schedule(
        action_type="alarm",
        scheduled_for=now + timedelta(days=1),
        payload={"message": "x"},
        source_record_id=42,
    )
    engine.schedule(
        action_type="reminder",
        scheduled_for=now + timedelta(days=2),
        payload={"message": "y"},
        source_record_id=42,
    )
    engine.schedule(
        action_type="alarm",
        scheduled_for=now + timedelta(days=3),
        payload={"message": "z"},
        source_record_id=99,
    )

    cancelled = engine.cancel(
        source_record_id=42,
        reason="event cancelled",
    )

    assert cancelled == 2

    remaining = engine.list_actions(status="pending")
    assert [a.source_record_id for a in remaining] == [99]


def test_failed_handler_records_error(engine):
    now = datetime.now(TZ)

    from nix_actions.handlers import register_handler

    def boom(action):
        raise RuntimeError("device offline")

    register_handler("notify", boom)

    action = engine.schedule(
        action_type="notify",
        scheduled_for=now - timedelta(seconds=1),
        payload={"message": "hi"},
    )

    fired = engine.run_due()

    assert fired[0].status == "failed"
    assert "device offline" in fired[0].last_error
