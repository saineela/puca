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


def test_capture_validates_source(engine):
    now = datetime.now(TZ)

    with pytest.raises(ValueError):
        engine.capture(
            action_type="alarm",
            scheduled_for=now,
            source="mystery_component",
        )


def test_capture_assigns_session_and_event(engine):
    now = datetime.now(TZ)

    action = engine.capture(
        action_type="alarm",
        scheduled_for=now,
        payload={"message": "wake"},
        source="nix_core",
    )

    # tag is derived from scheduled_for (same moment we passed in),
    # so compute the expectation instead of hardcoding a date that
    # rots as real weeks pass
    assert action.session_tag == engine.session_tag_for(now, bucket="week")
    assert action.session_tag.startswith("week-")

    captured = [
        e for e in engine.list_events(kind="captured")
        if e.action_id == action.id
    ]
    assert len(captured) == 1
    assert "nix_core" in captured[0].detail
    assert captured[0].session_tag == action.session_tag


def test_session_tag_buckets(engine):
    sunday = datetime(2026, 9, 6, 10, 0, tzinfo=TZ)
    monday = datetime(2026, 9, 7, 8, 0, tzinfo=TZ)

    assert engine.session_tag_for(sunday, bucket="week") == "week-2026-08-31"
    assert engine.session_tag_for(monday, bucket="week") == "week-2026-09-07"
    assert engine.session_tag_for(monday, bucket="day") == "day-2026-09-07"

    with pytest.raises(ValueError):
        engine.session_tag_for(monday, bucket="month")


def test_list_sessions_aggregates(engine):
    now = datetime.now(TZ)

    engine.capture(
        action_type="alarm",
        scheduled_for=now,
        source="nix_core",
    )
    engine.capture(
        action_type="reminder",
        scheduled_for=now,
        source="nix_knowledge",
    )

    sessions = engine.list_sessions()
    assert len(sessions) == 1

    session = sessions[0]
    assert session.total == 2
    assert session.pending == 2
    assert session.session_tag.startswith("week-")


def test_mock_run_survives_broken_handler(engine):
    now = datetime.now(TZ)

    from nix_actions.handlers import register_handler

    def boom(action):
        raise RuntimeError("device offline")

    register_handler("alarm", boom)

    engine.capture(
        action_type="alarm",
        scheduled_for=now - timedelta(seconds=1),
        payload={"message": "hi"},
        source="nix_core",
    )

    # live run fails loudly...
    fired = engine.run_due(mock=False)
    assert fired[0].status == "failed"
    assert "device offline" in fired[0].last_error


def test_mock_run_flips_status_without_handlers(engine):
    now = datetime.now(TZ)

    from nix_actions.handlers import register_handler

    def boom(action):
        raise RuntimeError("device offline")

    register_handler("alarm", boom)

    engine.capture(
        action_type="alarm",
        scheduled_for=now - timedelta(seconds=1),
        payload={"message": "hi"},
        source="nix_core",
    )

    # ...while the mock run ignores handler code entirely
    fired = engine.run_due(mock=True)

    assert len(fired) == 1
    assert fired[0].status == "fired"
    assert fired[0].last_error is None


def test_knowledge_update_hook_cancels_pending(engine):
    now = datetime.now(TZ)

    engine.capture(
        action_type="reminder",
        scheduled_for=now + timedelta(days=1),
        source="knowledge",
        source_record_id=7,
        knowledge_type="task",
    )
    engine.capture(
        action_type="alarm",
        scheduled_for=now + timedelta(days=2),
        source="knowledge",
        source_record_id=7,
        knowledge_type="event",
    )
    other = engine.capture(
        action_type="alarm",
        scheduled_for=now + timedelta(days=3),
        source="knowledge",
        source_record_id=9,
    )

    cancelled = engine.on_knowledge_updated(
        knowledge_type="task",
        source_record_id=7,
        reason="task edited",
    )

    assert cancelled == 1

    remaining = engine.list_actions(status="pending")
    # the 'event' record-7 action and the record-9 action stay pending
    assert other.id in [a.id for a in remaining]
    assert len(remaining) == 2

    kinds = [
        e.kind
        for e in engine.list_events(kind="knowledge_updated")
    ]
    assert kinds == ["knowledge_updated"]


def test_stats_counts_by_source_and_type(engine):
    now = datetime.now(TZ)

    engine.capture(
        action_type="alarm",
        scheduled_for=now,
        source="nix_core",
    )
    engine.capture(
        action_type="alarm",
        scheduled_for=now,
        source="nix_knowledge",
    )

    stats = engine.stats()

    assert stats["total"] == 2
    assert stats["pending"] == 2
    assert stats["by_source"]["nix_core"] == 1
    assert stats["by_source"]["nix_knowledge"] == 1
    assert stats["by_type"]["alarm"] == 2


def test_events_trail_records_lifecycle(engine):
    now = datetime.now(TZ)

    due = engine.capture(
        action_type="alarm",
        scheduled_for=now - timedelta(seconds=1),
        source="nix_core",
    )
    future = engine.capture(
        action_type="reminder",
        scheduled_for=now + timedelta(hours=1),
        source="nix_core",
    )

    engine.run_due(mock=True)
    engine.cancel(action_id=future.id, reason="not needed")

    kinds = [e.kind for e in engine.list_events()]

    assert kinds == ["cancelled", "fired", "captured", "captured"]


def test_recurring_capture_keeps_session_tag(engine):
    now = datetime.now(TZ)

    first = engine.capture(
        action_type="alarm",
        scheduled_for=now - timedelta(seconds=1),
        recurrence="weekly",
        source="nix_core",
        session_bucket="day",
    )
    engine.run_due(mock=True)

    pending = engine.list_actions(status="pending")
    assert len(pending) == 1
    assert pending[0].session_tag == first.session_tag
