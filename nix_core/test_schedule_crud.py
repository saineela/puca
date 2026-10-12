"""Manual schedule CRUD: create, edit, and delete calendar events through
the same Knowledge + Actions pipeline the assistant uses.

Mock-first: every test runs against temporary databases, never the live ones.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, "/root/nix_actions")
sys.path.insert(0, "/root/nix_knowledge")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_TMP = tempfile.mkdtemp(prefix="nix-schedule-test-")
os.environ["NIX_KNOWLEDGE_DB"] = os.path.join(_TMP, "knowledge.db")
os.environ["NIX_ACTIONS_DB"] = os.path.join(_TMP, "actions.db")

import console as console_module  # noqa: E402


@pytest.fixture()
def fresh_dbs(monkeypatch):
    """Point the module-level DB paths at fresh temp files per test."""
    knowledge = os.path.join(_TMP, f"k{os.getpid()}.db")
    actions = os.path.join(_TMP, f"a{os.getpid()}.db")
    for suffix in ("", "-wal", "-shm"):
        for path in (knowledge, actions):
            try:
                os.remove(path + suffix)
            except FileNotFoundError:
                pass
    monkeypatch.setattr(console_module, "KNOWLEDGE_DB", knowledge)
    monkeypatch.setattr(console_module, "ACTIONS_DB", actions)
    yield knowledge, actions


def _actions_rows(actions_db):
    import sqlite3

    conn = sqlite3.connect(actions_db)
    try:
        rows = conn.execute(
            "SELECT id, action_type, status, scheduled_for, source_record_id FROM actions"
        ).fetchall()
    finally:
        conn.close()
    return rows


def _event_statuses(knowledge_db):
    import sqlite3

    conn = sqlite3.connect(knowledge_db)
    try:
        return dict(
            conn.execute(
                "SELECT id, status FROM knowledge WHERE knowledge_type = 'calendar_event'"
            ).fetchall()
        )
    finally:
        conn.close()


def test_create_event_stores_record_and_schedules_reminder(fresh_dbs):
    knowledge, actions = fresh_dbs
    result = console_module._schedule_event_create({
        "title": "Dinner with Emma",
        "start": "2026-10-13 19:00",
        "end": "2026-10-13 21:00",
        "action_type": "reminder",
    })
    assert result["ok"] is True
    event = result["event"]
    assert event["title"] == "Dinner with Emma"
    assert event["start"].startswith("2026-10-13T19:00")
    rows = _actions_rows(actions)
    assert len(rows) == 1
    assert rows[0][1] == "reminder"
    assert rows[0][2] == "pending"
    assert rows[0][4] == result["record_id"]


def test_create_all_day_event_accepts_date_only(fresh_dbs):
    result = console_module._schedule_event_create({
        "title": "Trip to Austin",
        "start": "2026-11-02",
        "all_day": True,
        "action_type": "none",
    })
    assert result["event"]["all_day"] is True
    assert result["event"]["start"].startswith("2026-11-02T00:00")
    # action_type none -> no reminder row
    assert _actions_rows(fresh_dbs[1]) == []


def test_create_rejects_bad_input(fresh_dbs):
    with pytest.raises(ValueError, match="title is required"):
        console_module._schedule_event_create({"title": "  ", "start": "2026-10-13 19:00"})
    with pytest.raises(ValueError, match="Start is required"):
        console_module._schedule_event_create({"title": "No time"})
    with pytest.raises(ValueError, match="must look like"):
        console_module._schedule_event_create({"title": "Bad", "start": "sometime soon"})
    with pytest.raises(ValueError, match="End must not be before start"):
        console_module._schedule_event_create({
            "title": "Backwards", "start": "2026-10-13 20:00", "end": "2026-10-13 18:00",
        })
    with pytest.raises(ValueError, match="Recurrence"):
        console_module._schedule_event_create({
            "title": "Weird", "start": "2026-10-13 19:00", "recurrence": "hourly",
        })
    with pytest.raises(ValueError, match="120"):
        console_module._schedule_event_create({
            "title": "x" * 130, "start": "2026-10-13 19:00",
        })


def test_create_accepts_ampm_and_slash_formats(fresh_dbs):
    result = console_module._schedule_event_create({
        "title": "Formatted", "start": "10/13/2026 7:00 PM",
    })
    assert result["event"]["start"].startswith("2026-10-13T19:00")
    result2 = console_module._schedule_event_create({
        "title": "Slashed", "start": "2026/10/14 08:30",
    })
    assert result2["event"]["start"].startswith("2026-10-14T08:30")


def test_update_changes_time_and_rebuilds_reminder(fresh_dbs):
    knowledge, actions = fresh_dbs
    created = console_module._schedule_event_create({
        "title": "Call Marcus", "start": "2026-10-09 16:00", "action_type": "reminder",
    })
    record_id = created["record_id"]
    before = _actions_rows(actions)
    assert len(before) == 1 and before[0][2] == "pending"

    updated = console_module._schedule_event_update(
        record_id, {"start": "2026-10-10 09:30", "title": "Call Marcus about garage"}
    )
    assert updated["event"]["start"].startswith("2026-10-10T09:30")
    after = _actions_rows(actions)
    pending = [row for row in after if row[2] == "pending"]
    assert len(pending) == 1
    assert pending[0][3].startswith("2026-10-10T09:30")
    # old action cancelled, not deleted
    cancelled = [row for row in after if row[2] == "cancelled"]
    assert len(cancelled) == 1


def test_update_unknown_event_raises(fresh_dbs):
    with pytest.raises(ValueError, match="not found"):
        console_module._schedule_event_update(999999, {"start": "2026-10-10 09:30"})


def test_delete_cancels_actions_and_retires_record(fresh_dbs):
    knowledge, actions = fresh_dbs
    created = console_module._schedule_event_create({
        "title": "Temp", "start": "2026-10-20 10:00", "action_type": "reminder",
    })
    record_id = created["record_id"]
    deleted = console_module._schedule_event_delete(record_id)
    assert deleted["ok"] is True
    assert deleted["cancelled_actions"] >= 1
    rows = _actions_rows(actions)
    assert all(row[2] != "pending" for row in rows)
    # record still exists but is cancelled
    data = console_module._schedule_event_row(record_id)
    assert data["title"] == "Temp"
    with pytest.raises(ValueError, match="not found"):
        console_module._schedule_event_delete(999999)


def test_recurring_event_creates_pending_action(fresh_dbs):
    knowledge, actions = fresh_dbs
    result = console_module._schedule_event_create({
        "title": "Weekly sync", "start": "2026-10-12 09:00",
        "recurrence": "weekly", "action_type": "reminder",
    })
    assert result["event"]["recurring"] is True
    rows = _actions_rows(actions)
    assert len(rows) == 1 and rows[0][2] == "pending"


def test_identical_events_merge_into_a_single_event(fresh_dbs):
    knowledge, actions = fresh_dbs
    first = console_module._schedule_event_create({
        "title": "Standup", "start": "2027-01-15 09:00", "action_type": "reminder",
    })
    second = console_module._schedule_event_create({
        "title": "Standup", "start": "2027-01-15 09:00", "action_type": "reminder",
    })
    assert first["record_id"] != second["record_id"]

    view = console_module._schedule_view()
    standups = [e for e in view["upcoming"] if e["title"] == "Standup"]
    assert len(standups) == 1
    survivor = standups[0]
    assert survivor["id"] == first["record_id"]  # earliest id wins
    assert survivor["merged_count"] == 2
    assert sorted(survivor["merged_ids"]) == sorted(
        [first["record_id"], second["record_id"]]
    )

    # Exactly one reminder survives; the merged-away duplicate was cancelled.
    rows = _actions_rows(actions)
    assert len([r for r in rows if r[2] == "pending"]) == 1
    assert len([r for r in rows if r[2] == "cancelled"]) == 1

    statuses = _event_statuses(knowledge)
    assert statuses[first["record_id"]] == "active"
    assert statuses[second["record_id"]] == "cancelled"

    # Idempotent: a second read still returns a single Standup.
    again = console_module._schedule_view()
    assert len([e for e in again["upcoming"] if e["title"] == "Standup"]) == 1


def test_same_title_different_start_is_not_merged(fresh_dbs):
    console_module._schedule_event_create({
        "title": "Gym", "start": "2027-02-01 07:00", "action_type": "none",
    })
    console_module._schedule_event_create({
        "title": "Gym", "start": "2027-02-01 08:00", "action_type": "none",
    })
    view = console_module._schedule_view()
    assert len([e for e in view["upcoming"] if e["title"] == "Gym"]) == 2


def test_same_start_different_title_is_not_merged(fresh_dbs):
    console_module._schedule_event_create({
        "title": "Alpha", "start": "2027-03-01 10:00", "action_type": "none",
    })
    console_module._schedule_event_create({
        "title": "Beta", "start": "2027-03-01 10:00", "action_type": "none",
    })
    view = console_module._schedule_view()
    titles = {e["title"] for e in view["upcoming"]}
    assert {"Alpha", "Beta"} <= titles
