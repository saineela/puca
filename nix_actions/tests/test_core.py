from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from nix_actions.engine import ActionsEngine
from nix_core.core import NixCore
from nix_core.knowledge import (
    KnowledgeRecord,
    StaticKnowledgeProvider,
)
from nix_core.sessions import SessionStore, session_tag_for

TZ = ZoneInfo("America/Chicago")


@pytest.fixture
def provider():
    return StaticKnowledgeProvider(
        records=[
            KnowledgeRecord(
                id=7,
                knowledge_type="task",
                content="Robotics class Tuesday 18:00",
                version=1,
            ),
            KnowledgeRecord(
                id=8,
                knowledge_type="note",
                content="Gym membership renews in October",
                version=3,
            ),
        ]
    )


@pytest.fixture
def core(tmp_path, provider):
    c = NixCore(
        knowledge=provider,
        database_path=tmp_path / "nix_core.db",
        actions_database_path=tmp_path / "actions.db",
        timezone="America/Chicago",
        session_bucket="week",
    )
    yield c
    c.close()


def test_session_tags_match_nix_actions(tmp_path):
    assert session_tag_for(
        datetime(2026, 9, 6, 10, 0, tzinfo=TZ), bucket="week"
    ) == ActionsEngine(":memory:").session_tag_for(
        datetime(2026, 9, 6, 10, 0, tzinfo=TZ), bucket="week"
    )


def test_turns_land_in_current_session(core):
    response = core.handle_request(text="hello there")

    turns = core.sessions.list_turns()
    assert len(turns) == 2  # user + assistant

    week_tag = session_tag_for(
        datetime.now(TZ), bucket="week"
    )
    assert all(t.session_tag == week_tag for t in turns)
    assert response.session_tag == week_tag


def test_context_window_scopes_source_and_conversation(core):
    core.sessions.add_turn(
        role="user",
        content="dashboard thread one",
        refs={"location": "dashboard", "conversation_id": "dash-1"},
    )
    core.sessions.add_turn(
        role="assistant",
        content="dashboard reply one",
        refs={"location": "dashboard", "conversation_id": "dash-1"},
    )
    core.sessions.add_turn(
        role="user",
        content="dashboard thread two",
        refs={"location": "dashboard", "conversation_id": "dash-2"},
    )
    core.sessions.add_turn(
        role="user",
        content="API thread one",
        refs={"location": "openai-api", "conversation_id": "api-1"},
    )

    thread_one = core.sessions.context_window(
        location="dashboard",
        conversation_id="dash-1",
    )
    assert [turn.content for turn in thread_one] == [
        "dashboard thread one",
        "dashboard reply one",
    ]
    assert core.sessions.context_window(
        location="openai-api",
        conversation_id="api-1",
    )[0].content == "API thread one"
    assert core.sessions.context_window(
        location="dashboard",
        conversation_id="missing-thread",
    ) == []
    assert core.sessions.context_window(location="dashboard") == []
    assert len(core.sessions.context_window()) == 4


def test_context_window_excludes_pruned(core):
    core.handle_request(text="tell me about the robotics class")

    # upstream updates task#7
    core.knowledge = StaticKnowledgeProvider(
        records=[
            KnowledgeRecord(
                id=7,
                knowledge_type="task",
                content="Robotics class MOVED to Thursday",
                version=2,
            )
        ]
    )

    core.on_knowledge_updated(
        knowledge_type="task",
        source_record_id=7,
        reason="class moved",
    )

    window = core.sessions.context_window()

    # turns referencing task#7 are pruned out of context
    assert all(
        7 not in t.refs.get("knowledge_record_ids", [])
        for t in window
    )

    # but still on disk when explicitly requested
    all_turns = core.sessions.list_turns(include_pruned=True)
    assert any(t.pruned for t in all_turns)


def test_stale_knowledge_pruned_on_request(core, capsys):
    # request referencing task#7 v1
    core.handle_request(text="robotics class please")

    # simulate version drift discovered on a later request
    stale = KnowledgeRecord(
        id=7,
        knowledge_type="task",
        content="old snapshot",
        version=1,
    )
    core.knowledge = StaticKnowledgeProvider(
        records=[
            KnowledgeRecord(
                id=7,
                knowledge_type="task",
                content="Robotics class is now Thursday",
                version=2,
            ),
        ]
    )

    pruned = core._reconcile_knowledge([stale])
    assert pruned >= 1


def test_request_with_reminder_captures_action(core):
    response = core.handle_request(
        text="remind me about the robotics class"
    )

    assert len(response.actions_captured) == 1

    action = core.actions.get(response.actions_captured[0])
    assert action.action_type == "reminder"
    assert action.source == "nix_core"
    assert action.source_record_id == 7
    assert action.session_tag == response.session_tag


def test_knowledge_update_cancels_linked_actions(core):
    captured = core.handle_request(
        text="remind me about the robotics class"
    )
    action_id = captured.actions_captured[0]

    result = core.on_knowledge_updated(
        knowledge_type="task",
        source_record_id=7,
        reason="class moved",
    )

    assert result["cancelled_actions"] == 1
    assert result["pruned_turns"] >= 1

    action = core.actions.get(action_id)
    assert action.status == "cancelled"
    assert "knowledge_updated" in (action.last_error or "")


def test_reschedule_passthrough(core):
    captured = core.handle_request(
        text="remind me about the robotics class"
    )

    new_time = datetime.now(TZ) + timedelta(days=3)
    moved = core.reschedule(
        source_record_id=7,
        scheduled_for=new_time,
    )

    assert moved == 1
    assert (
        core.actions.get(captured.actions_captured[0]).scheduled_for
        == new_time
    )


def test_clear_stale_sessions_keeps_current(core):
    now = datetime.now(TZ)
    old_week = now - timedelta(days=14)

    core.sessions.add_turn(
        role="user",
        content="old session turn",
        when=old_week,
    )
    core.sessions.add_turn(
        role="user",
        content="current session turn",
        when=now,
    )

    cleared = core.sessions.clear_stale_sessions()

    assert cleared == 1

    remaining = core.sessions.list_turns(include_pruned=True)
    assert [t.content for t in remaining] == ["current session turn"]


def test_maintenance_reports_counts(core):
    core.handle_request(text="hello")
    result = core.maintain()

    assert set(result) == {"cleared_turns", "purged_actions"}
    assert result["cleared_turns"] == 0  # current week stays


def test_event_log_written(tmp_path, provider):
    log_path = tmp_path / "events.jsonl"

    core = NixCore(
        knowledge=provider,
        database_path=tmp_path / "core.db",
        actions_database_path=tmp_path / "actions.db",
        event_log_path=log_path,
    )
    try:
        core.handle_request(text="robotics class")
        core.on_knowledge_updated(
            knowledge_type="task",
            source_record_id=7,
            reason="moved",
        )
    finally:
        core.close()

    lines = log_path.read_text().strip().splitlines()
    kinds = [json.loads(line)["kind"] for line in lines]

    assert "request" in kinds
    assert "knowledge_updated" in kinds
