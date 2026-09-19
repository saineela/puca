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


def test_upstream_cancel_propagates_across_sources(engine):
    """An upstream record cancelled by nix_knowledge must cancel the
    actions it spawned no matter which component captured them."""
    now = datetime.now(TZ)

    by_knowledge = engine.capture(
        action_type="reminder",
        scheduled_for=now + timedelta(days=1),
        source="knowledge",
        source_record_id=100,
    )
    by_nk = engine.capture(
        action_type="alarm",
        scheduled_for=now + timedelta(days=2),
        source="nix_knowledge",
        source_record_id=100,
    )
    by_core = engine.capture(
        action_type="notify",
        scheduled_for=now + timedelta(days=3),
        source="nix_core",
        source_record_id=100,
    )
    unrelated = engine.capture(
        action_type="alarm",
        scheduled_for=now + timedelta(days=4),
        source="nix_knowledge",
        source_record_id=200,
    )

    # upstream says: record 100 is cancelled
    cancelled = engine.cancel(
        source_record_id=100,
        reason="event cancelled upstream",
    )

    assert cancelled == 3

    pending = engine.list_actions(status="pending")
    assert [a.id for a in pending] == [unrelated.id]

    for action in (by_knowledge, by_nk, by_core):
        assert engine.get(action.id).status == "cancelled"

    kinds = [e.kind for e in engine.list_events(limit=10)]
    assert kinds.count("cancelled") >= 3


def test_upstream_cancel_can_be_narrowed_to_one_source(engine):
    now = datetime.now(TZ)

    engine.capture(
        action_type="alarm",
        scheduled_for=now + timedelta(days=1),
        source="knowledge",
        source_record_id=5,
    )
    nk_action = engine.capture(
        action_type="alarm",
        scheduled_for=now + timedelta(days=2),
        source="nix_knowledge",
        source_record_id=5,
    )

    cancelled = engine.cancel(
        source_record_id=5,
        sources=("nix_knowledge",),
        reason="only nix_knowledge record changed",
    )

    assert cancelled == 1
    assert engine.get(nk_action.id).status == "cancelled"


def test_upstream_cancel_narrows_by_knowledge_type(engine):
    now = datetime.now(TZ)

    task = engine.capture(
        action_type="reminder",
        scheduled_for=now + timedelta(days=1),
        source="nix_knowledge",
        source_record_id=7,
        knowledge_type="task",
    )
    event = engine.capture(
        action_type="alarm",
        scheduled_for=now + timedelta(days=2),
        source="nix_knowledge",
        source_record_id=7,
        knowledge_type="event",
    )

    engine.cancel(
        source_record_id=7,
        knowledge_type="event",
        reason="event record changed",
    )

    assert engine.get(task.id).status == "pending"
    assert engine.get(event.id).status == "cancelled"


def test_reschedule_propagates_time_change(engine):
    now = datetime.now(TZ)
    new_time = now + timedelta(days=10)

    action = engine.capture(
        action_type="alarm",
        scheduled_for=now + timedelta(days=1),
        payload={"message": "standup"},
        source="nix_knowledge",
        source_record_id=300,
    )

    moved = engine.reschedule(
        source_record_id=300,
        scheduled_for=new_time,
        reason="postponed",
    )

    assert moved == 1

    updated = engine.get(action.id)
    assert updated.scheduled_for == new_time
    assert updated.status == "pending"
    assert updated.session_tag == engine.session_tag_for(
        new_time, bucket="week"
    )

    kinds = [e.kind for e in engine.list_events(limit=5)]
    assert "rescheduled" in kinds


def test_reschedule_only_touches_pending(engine):
    now = datetime.now(TZ)

    done = engine.capture(
        action_type="alarm",
        scheduled_for=now - timedelta(seconds=1),
        source="nix_core",
        source_record_id=400,
    )
    engine.run_due(mock=True)

    future = engine.capture(
        action_type="alarm",
        scheduled_for=now + timedelta(days=1),
        source="nix_core",
        source_record_id=400,
    )

    moved = engine.reschedule(
        source_record_id=400,
        scheduled_for=now + timedelta(days=5),
    )

    assert moved == 1
    assert engine.get(done.id).status == "fired"  # untouched
    assert engine.get(future.id).scheduled_for == now + timedelta(days=5)


def test_knowledge_updated_hook_cancels_across_sources(engine):
    now = datetime.now(TZ)

    engine.capture(
        action_type="reminder",
        scheduled_for=now + timedelta(days=1),
        source="nix_knowledge",
        source_record_id=50,
        knowledge_type="task",
    )
    engine.capture(
        action_type="alarm",
        scheduled_for=now + timedelta(days=2),
        source="knowledge",
        source_record_id=50,
        knowledge_type="task",
    )

    count = engine.on_knowledge_updated(
        source_record_id=50,
        knowledge_type="task",
        reason="task rewritten",
    )

    assert count == 2
    assert engine.list_actions(status="pending") == []

    kinds = [e.kind for e in engine.list_events(kind="knowledge_updated")]
    assert kinds == ["knowledge_updated"]


def test_non_upstream_sources_are_not_matched(engine):
    """user/cli/demo actions are local to this engine; upstream
    record changes must not reach them."""
    now = datetime.now(TZ)

    local = engine.schedule(
        action_type="alarm",
        scheduled_for=now + timedelta(days=1),
        source="cli",
        source_record_id=999,
    )

    assert engine.cancel(source_record_id=999, reason="upstream") == 0
    assert engine.get(local.id).status == "pending"
