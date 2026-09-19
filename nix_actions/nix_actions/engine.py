from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


VALID_ACTION_TYPES = {
    # device / software triggers
    "alarm",
    "reminder",
    "notify",
    "webhook",
    "maintenance",
}

# Who can capture actions into the engine. nix_core and nix_knowledge
# are the upstream producers; once they exist they may capture freely
# with no code changes on this side.
VALID_SOURCE_TYPES = {
    "knowledge",
    "nix_core",
    "nix_knowledge",
    "user",
    "cli",
    "demo",
}

# Upstream producers whose lifecycle changes (cancel / reschedule /
# delete) propagate to the actions they spawned. Anything else (user,
# cli, demo) is local to this engine and is not auto-matched.
UPSTREAM_SOURCES = {"knowledge", "nix_core", "nix_knowledge"}

# Lifecycle events worth surfacing in the dashboard's activity feed.
EVENT_KINDS = {
    "captured",
    "fired",
    "failed",
    "cancelled",
    "rescheduled",
    "purged",
    "knowledge_updated",
}

VALID_STATUSES = {
    "pending",
    "fired",
    "cancelled",
    "failed",
}


@dataclass(frozen=True)
class CaptureEvent:
    """One audit-trail row: capture, fire, fail, cancel, purge..."""

    id: int
    action_id: int | None
    kind: str
    detail: str
    session_tag: str | None
    created_at: datetime


@dataclass(frozen=True)
class SessionInfo:
    """Aggregated view of one session tag (day or week bucket)."""

    session_tag: str
    bucket: str
    start: date
    end: date | None
    total: int
    pending: int
    fired: int
    failed: int
    cancelled: int


@dataclass(frozen=True)
class Action:
    id: int
    action_type: str
    scheduled_for: datetime
    payload: dict[str, Any]

    recurrence: str | None
    recurrence_end: datetime | None

    source: str
    source_record_id: int | None
    knowledge_type: str | None

    session_tag: str | None

    status: str
    created_at: datetime
    fired_at: datetime | None
    last_error: str | None

    metadata: dict[str, Any] = field(default_factory=dict)


def _now(tz: ZoneInfo) -> datetime:
    return datetime.now(tz)


class ActionsEngine:
    """
    Deterministic action scheduler and capture service.

    Responsibilities:
      - store actions captured from upstream components
        (nix_core, nix_knowledge, users)
      - fire actions whose time has come via registered handlers
      - materialize recurring actions one occurrence at a time
      - keep an audit trail of what fired and when
      - propagate upstream lifecycle changes: when the source record is
        cancelled, moved, edited, or deleted, the linked actions here
        follow

    This component contains NO intelligence. It never interprets
    natural language and never decides policy. It executes.
    """

    def __init__(
        self,
        database_path: str | Path = "actions.db",
        timezone: str = "America/Chicago",
    ):
        self.timezone = ZoneInfo(timezone)

        self.connection = sqlite3.connect(
            database_path,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row

        self._initialize()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                action_type TEXT NOT NULL,

                scheduled_for TEXT NOT NULL,

                payload TEXT NOT NULL,

                recurrence TEXT,
                recurrence_end TEXT,

                source TEXT NOT NULL DEFAULT 'knowledge',
                source_record_id INTEGER,
                knowledge_type TEXT,

                session_tag TEXT,

                status TEXT NOT NULL DEFAULT 'pending',

                created_at TEXT NOT NULL,
                fired_at TEXT,
                last_error TEXT,

                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS capture_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                action_id INTEGER,

                kind TEXT NOT NULL,

                detail TEXT NOT NULL DEFAULT '',

                session_tag TEXT,

                created_at TEXT NOT NULL
            );
            """
        )

        # Migration for databases created before session tagging.
        # This MUST run before the index below: creating
        # idx_actions_session on a pre-migration table would fail and
        # leave the database unusable.
        existing = {
            row["name"]
            for row in self.connection.execute(
                "PRAGMA table_info(actions)"
            )
        }
        if "session_tag" not in existing:
            self.connection.execute(
                "ALTER TABLE actions ADD COLUMN session_tag TEXT"
            )

        self.connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_actions_status_time
                ON actions(status, scheduled_for);

            CREATE INDEX IF NOT EXISTS idx_actions_source
                ON actions(source, source_record_id);

            CREATE INDEX IF NOT EXISTS idx_actions_session
                ON actions(session_tag);

            CREATE INDEX IF NOT EXISTS idx_events_created
                ON capture_events(created_at);
            """
        )

        self.connection.commit()

    @staticmethod
    def _encode(data: Any) -> str:
        return json.dumps(data, ensure_ascii=False, default=str)

    @staticmethod
    def _decode(data: str | None) -> Any:
        if data is None:
            return None
        return json.loads(data)

    @staticmethod
    def _iso(dt: datetime | None) -> str | None:
        return dt.isoformat() if dt is not None else None

    @staticmethod
    def session_tag_for(
        when: datetime,
        *,
        bucket: str = "week",
    ) -> str:
        """
        Stable bucket key for grouping actions into sessions.

        Days are calendar days; weeks are Monday-based, tagged by the
        Monday, so every moment maps to exactly one session.
        """
        if bucket == "day":
            return f"day-{when.date().isoformat()}"

        if bucket == "week":
            monday = when.date() - timedelta(days=when.weekday())
            return f"week-{monday.isoformat()}"

        raise ValueError(
            f"Unsupported session bucket: {bucket} "
            f"(use 'day' or 'week')"
        )

    def log_event(
        self,
        *,
        kind: str,
        action_id: int | None = None,
        detail: str = "",
        session_tag: str | None = None,
        when: datetime | None = None,
    ) -> CaptureEvent:
        if kind not in EVENT_KINDS:
            raise ValueError(
                f"Invalid event kind: {kind}. "
                f"Valid kinds: {sorted(EVENT_KINDS)}"
            )

        moment = when or _now(self.timezone)

        cursor = self.connection.execute(
            """
            INSERT INTO capture_events
                (action_id, kind, detail, session_tag, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                action_id,
                kind,
                detail,
                session_tag,
                moment.isoformat(),
            ),
        )
        self.connection.commit()

        return CaptureEvent(
            id=int(cursor.lastrowid or 0),
            action_id=action_id,
            kind=kind,
            detail=detail,
            session_tag=session_tag,
            created_at=moment,
        )

    def close(self) -> None:
        self.connection.close()

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def schedule(
        self,
        *,
        action_type: str,
        scheduled_for: datetime,
        payload: dict[str, Any] | None = None,
        recurrence: str | None = None,
        recurrence_end: datetime | None = None,
        source: str = "knowledge",
        source_record_id: int | None = None,
        knowledge_type: str | None = None,
        session_tag: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Action:
        """
        Schedule an action.

        scheduled_for must be timezone-aware; naive datetimes are
        interpreted in the engine's local timezone.
        """
        if action_type not in VALID_ACTION_TYPES:
            raise ValueError(
                f"Invalid action type: {action_type}. "
                f"Valid types: {sorted(VALID_ACTION_TYPES)}"
            )

        if recurrence is not None and recurrence not in {
            "daily",
            "weekly",
        }:
            raise ValueError(
                f"Unsupported recurrence: {recurrence}"
            )

        if scheduled_for.tzinfo is None:
            scheduled_for = scheduled_for.replace(
                tzinfo=self.timezone
            )

        if recurrence == "weekly":
            # normalize weekly occurrences to the same local time
            scheduled_for = scheduled_for.replace(
                second=0,
                microsecond=0,
            )

        now = _now(self.timezone)

        cursor = self.connection.execute(
            """
            INSERT INTO actions (
                action_type,
                scheduled_for,
                payload,
                recurrence,
                recurrence_end,
                source,
                source_record_id,
                knowledge_type,
                session_tag,
                status,
                created_at,
                metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (
                action_type,
                scheduled_for.isoformat(),
                self._encode(payload or {}),
                recurrence,
                self._iso(recurrence_end),
                source,
                source_record_id,
                knowledge_type,
                session_tag,
                now.isoformat(),
                self._encode(metadata or {}),
            ),
        )
        self.connection.commit()

        return self.get(int(cursor.lastrowid or 0))

    def capture(
        self,
        *,
        action_type: str,
        scheduled_for: datetime,
        payload: dict[str, Any] | None = None,
        recurrence: str | None = None,
        recurrence_end: datetime | None = None,
        source: str = "nix_core",
        source_record_id: int | None = None,
        knowledge_type: str | None = None,
        session_bucket: str = "week",
        metadata: dict[str, Any] | None = None,
    ) -> Action:
        """
        Capture service intake: an upstream component (nix_core or
        nix_knowledge) hands in an action to organize and track.

        - sources are validated so upstream typos fail loudly
        - the session tag is derived from scheduled_for, so every
          action lands in exactly one day/week session
        - a 'captured' event lands in the audit trail
        """
        if source not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"Invalid source: {source}. "
                f"Valid sources: {sorted(VALID_SOURCE_TYPES)}"
            )

        action = self.schedule(
            action_type=action_type,
            scheduled_for=scheduled_for,
            payload=payload,
            recurrence=recurrence,
            recurrence_end=recurrence_end,
            source=source,
            source_record_id=source_record_id,
            knowledge_type=knowledge_type,
            metadata=metadata,
        )

        tagged = self._set_session_tag(
            action.id,
            self.session_tag_for(action.scheduled_for, bucket=session_bucket),
        )

        self.log_event(
            kind="captured",
            action_id=action.id,
            detail=(
                f"{source} captured {action.action_type} "
                f"for {tagged.scheduled_for.isoformat()}"
            ),
            session_tag=tagged.session_tag,
        )

        return tagged

    def _set_session_tag(
        self,
        action_id: int,
        session_tag: str | None,
    ) -> Action:
        self.connection.execute(
            "UPDATE actions SET session_tag = ? WHERE id = ?",
            (session_tag, action_id),
        )
        self.connection.commit()
        return self.get(action_id)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, action_id: int) -> Action:
        row = self.connection.execute(
            "SELECT * FROM actions WHERE id = ?",
            (action_id,),
        ).fetchone()

        if row is None:
            raise KeyError(f"Action {action_id} does not exist")

        return self._row_to_action(row)

    def list_actions(
        self,
        *,
        status: str | None = "pending",
        due_before: datetime | None = None,
        session_tag: str | None = None,
        source: str | None = None,
    ) -> list[Action]:
        query = "SELECT * FROM actions"
        conditions = []
        parameters: list[Any] = []

        if status is not None:
            conditions.append("status = ?")
            parameters.append(status)

        if due_before is not None:
            conditions.append("scheduled_for <= ?")
            parameters.append(due_before.isoformat())

        if session_tag is not None:
            conditions.append("session_tag = ?")
            parameters.append(session_tag)

        if source is not None:
            conditions.append("source = ?")
            parameters.append(source)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY scheduled_for ASC"

        rows = self.connection.execute(
            query, parameters
        ).fetchall()

        return [self._row_to_action(row) for row in rows]

    def _source_clause(
        self,
        sources: tuple[str, ...] | None,
    ) -> tuple[str, list[str]]:
        """
        Build the WHERE fragment matching actions by source.

        None means "any upstream producer" so lifecycle changes made by
        nix_core / nix_knowledge always find the actions they spawned,
        no matter which of them captured the action.
        """
        if sources is None:
            placeholders = ", ".join(
                "?" for _ in UPSTREAM_SOURCES
            )
            return (
                f"source IN ({placeholders})",
                sorted(UPSTREAM_SOURCES),
            )

        placeholders = ", ".join("?" for _ in sources)
        return (
            f"source IN ({placeholders})",
            list(sources),
        )

    def find_by_source(
        self,
        source_record_id: int,
        knowledge_type: str | None = None,
        *,
        sources: tuple[str, ...] | None = None,
    ) -> list[Action]:
        """
        All actions linked to an upstream record, across any upstream
        source. Pass sources=("nix_knowledge",) to narrow the match.
        """
        clause, source_values = self._source_clause(sources)

        query = f"""
            SELECT * FROM actions
            WHERE source_record_id = ?
              AND {clause}
        """
        parameters: list[Any] = [source_record_id, *source_values]

        if knowledge_type is not None:
            query += " AND knowledge_type = ?"
            parameters.append(knowledge_type)

        rows = self.connection.execute(
            query, parameters
        ).fetchall()

        return [self._row_to_action(row) for row in rows]

    def list_sessions(
        self,
        *,
        bucket: str | None = None,
    ) -> list[SessionInfo]:
        """
        Summarize actions grouped by session tag, oldest first.
        """
        rows = self.connection.execute(
            """
            SELECT session_tag,
                   COUNT(*)                  AS total,
                   SUM(status = 'pending')   AS pending,
                   SUM(status = 'fired')     AS fired,
                   SUM(status = 'failed')    AS failed,
                   SUM(status = 'cancelled') AS cancelled
            FROM actions
            WHERE session_tag IS NOT NULL
            GROUP BY session_tag
            ORDER BY session_tag ASC
            """
        ).fetchall()

        sessions: list[SessionInfo] = []

        for row in rows:
            tag = row["session_tag"]

            if bucket and not tag.startswith(f"{bucket}-"):
                continue

            stem = tag.split("-", 1)[1]
            start = date.fromisoformat(stem)

            end = None
            if tag.startswith("week-"):
                end = start + timedelta(days=6)

            sessions.append(
                SessionInfo(
                    session_tag=tag,
                    bucket="week" if tag.startswith("week-") else "day",
                    start=start,
                    end=end,
                    total=row["total"],
                    pending=row["pending"] or 0,
                    fired=row["fired"] or 0,
                    failed=row["failed"] or 0,
                    cancelled=row["cancelled"] or 0,
                )
            )
        return sessions

    def list_events(
        self,
        *,
        kind: str | None = None,
        session_tag: str | None = None,
        limit: int = 50,
    ) -> list[CaptureEvent]:
        """Recent audit-trail entries, newest first."""
        query = "SELECT * FROM capture_events"
        conditions = []
        parameters: list[Any] = []

        if kind is not None:
            conditions.append("kind = ?")
            parameters.append(kind)

        if session_tag is not None:
            conditions.append("session_tag = ?")
            parameters.append(session_tag)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        parameters.append(limit)

        rows = self.connection.execute(
            query, parameters
        ).fetchall()

        return [
            CaptureEvent(
                id=row["id"],
                action_id=row["action_id"],
                kind=row["kind"],
                detail=row["detail"],
                session_tag=row["session_tag"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def stats(self) -> dict[str, Any]:
        """Counters for the dashboard header."""
        actions_row = self.connection.execute(
            """
            SELECT COUNT(*)                                        AS total,
                   SUM(status = 'pending')                         AS pending,
                   SUM(status = 'fired')                           AS fired,
                   SUM(status = 'failed')                          AS failed,
                   SUM(status = 'cancelled')                       AS cancelled,
                   SUM(status = 'pending' AND scheduled_for <= ?)  AS due_now
            FROM actions
            """,
            (_now(self.timezone).isoformat(),),
        ).fetchone()

        sources_row = self.connection.execute(
            """
            SELECT source, COUNT(*) AS n
            FROM actions
            GROUP BY source
            ORDER BY n DESC
            """
        ).fetchall()

        return {
            "total": actions_row["total"],
            "pending": actions_row["pending"] or 0,
            "fired": actions_row["fired"] or 0,
            "failed": actions_row["failed"] or 0,
            "cancelled": actions_row["cancelled"] or 0,
            "due_now": actions_row["due_now"] or 0,
            "by_source": {
                row["source"]: row["n"] for row in sources_row
            },
            "by_type": {
                row["action_type"]: row["n"]
                for row in self.connection.execute(
                    """
                    SELECT action_type, COUNT(*) AS n
                    FROM actions GROUP BY action_type ORDER BY n DESC
                    """
                )
            },
        }

    @classmethod
    def _row_to_action(cls, row) -> Action:
        from datetime import datetime as _dt

        def _parse(value: str | None):
            return (
                _dt.fromisoformat(value)
                if value
                else None
            )

        return Action(
            id=row["id"],
            action_type=row["action_type"],
            scheduled_for=_parse(row["scheduled_for"]),
            payload=cls._decode(row["payload"]),
            recurrence=row["recurrence"],
            recurrence_end=_parse(row["recurrence_end"]),
            source=row["source"],
            source_record_id=row["source_record_id"],
            knowledge_type=row["knowledge_type"],
            session_tag=row["session_tag"],
            status=row["status"],
            created_at=_parse(row["created_at"]),
            fired_at=_parse(row["fired_at"]),
            last_error=row["last_error"],
            metadata=cls._decode(row["metadata"]) or {},
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_due(
        self,
        *,
        now: datetime | None = None,
        handlers: dict[str, Any] | None = None,
        mock: bool = False,
    ) -> list[Action]:
        """
        Fire every pending action whose time has come.

        Recurring actions are marked fired and their next occurrence
        is scheduled automatically, so a weekly alarm keeps existing
        until it is cancelled or its recurrence_end passes.

        With mock=True no handler code runs at all; statuses still
        flip and events still record. That is the dashboard's capture
        mode until real devices are wired in.
        """
        from .handlers import get_handler

        tz = self.timezone
        now = now or _now(tz)

        if now.tzinfo is None:
            now = now.replace(tzinfo=tz)

        handlers = handlers or {}

        due = self.list_actions(
            status="pending",
            due_before=now,
        )

        fired: list[Action] = []

        for action in due:
            error: str | None = None

            if not mock:
                handler = handlers.get(
                    action.action_type
                ) or get_handler(action.action_type)

                if handler is None:
                    error = (
                        f"No handler registered for "
                        f"'{action.action_type}'"
                    )
                else:
                    try:
                        handler(action)
                    except Exception as exc:  # noqa: BLE001
                        error = f"{type(exc).__name__}: {exc}"

            fired_at = _now(tz)

            self.connection.execute(
                """
                UPDATE actions
                SET status = ?,
                    fired_at = ?,
                    last_error = ?
                WHERE id = ?
                """,
                (
                    "failed" if error else "fired",
                    fired_at.isoformat(),
                    error,
                    action.id,
                ),
            )
            self.connection.commit()

            fired.append(self.get(action.id))

            self.log_event(
                kind="failed" if error else "fired",
                action_id=action.id,
                detail=(
                    (f"mock run: {error}" if error else "mock run")
                    if mock
                    else (error or "executed")
                ),
                session_tag=action.session_tag,
                when=fired_at,
            )

            if error is None and action.recurrence:
                self._materialize_next(action, now)

        return fired

    def _materialize_next(
        self,
        action: Action,
        now: datetime,
    ) -> Action | None:
        if action.recurrence == "daily":
            step = timedelta(days=1)
        elif action.recurrence == "weekly":
            step = timedelta(weeks=1)
        else:
            return None

        next_at = action.scheduled_for + step

        # skip past any occurrences missed while the engine was down
        while next_at <= now:
            next_at += step

        if (
            action.recurrence_end
            and next_at > action.recurrence_end
        ):
            return None

        return self.schedule(
            action_type=action.action_type,
            scheduled_for=next_at,
            payload=action.payload,
            recurrence=action.recurrence,
            recurrence_end=action.recurrence_end,
            source=action.source,
            source_record_id=action.source_record_id,
            knowledge_type=action.knowledge_type,
            session_tag=action.session_tag,
            metadata=action.metadata,
        )

    # ------------------------------------------------------------------
    # Upstream lifecycle propagation
    # ------------------------------------------------------------------

    def _linked_actions(
        self,
        *,
        source_record_id: int,
        knowledge_type: str | None,
        sources: tuple[str, ...] | None,
    ) -> list[Action]:
        clause, source_values = self._source_clause(sources)

        parameters: list[Any] = [
            source_record_id,
            *source_values,
        ]

        knowledge_clause = ""
        if knowledge_type is not None:
            knowledge_clause = " AND knowledge_type = ?"
            parameters.append(knowledge_type)

        rows = self.connection.execute(
            f"""
            SELECT * FROM actions
            WHERE source_record_id = ?
              AND {clause}
              AND status = 'pending'
              {knowledge_clause}
            ORDER BY id ASC
            """,
            parameters,
        ).fetchall()

        return [self._row_to_action(row) for row in rows]

    def cancel(
        self,
        *,
        source_record_id: int | None = None,
        knowledge_type: str | None = None,
        action_id: int | None = None,
        reason: str | None = None,
        sources: tuple[str, ...] | None = None,
    ) -> int:
        """
        Cancel pending actions, either by action id or by the upstream
        record that spawned them.

        Record-based cancellation matches actions from ANY upstream
        source (knowledge, nix_core, nix_knowledge) so an upstream
        cancellation propagates here regardless of who captured the
        action. Narrow with sources=("nix_knowledge",) if needed.
        Returns the number of actions cancelled.
        """
        if action_id is not None:
            cursor = self.connection.execute(
                """
                UPDATE actions
                SET status = 'cancelled',
                    last_error = ?
                WHERE id = ?
                  AND status = 'pending'
                """,
                (reason, action_id),
            )
            self.connection.commit()

            if cursor.rowcount:
                row = self.connection.execute(
                    "SELECT session_tag FROM actions WHERE id = ?",
                    (action_id,),
                ).fetchone()
                self.log_event(
                    kind="cancelled",
                    action_id=action_id,
                    detail=reason or "cancelled",
                    session_tag=(
                        row["session_tag"] if row else None
                    ),
                )

            return cursor.rowcount

        if source_record_id is not None:
            affected = self._linked_actions(
                source_record_id=source_record_id,
                knowledge_type=knowledge_type,
                sources=sources,
            )

            clause, source_values = self._source_clause(sources)

            parameters: list[Any] = [
                reason,
                source_record_id,
                *source_values,
            ]

            knowledge_clause = ""
            if knowledge_type is not None:
                knowledge_clause = " AND knowledge_type = ?"
                parameters.append(knowledge_type)

            cursor = self.connection.execute(
                f"""
                UPDATE actions
                SET status = 'cancelled',
                    last_error = ?
                WHERE source_record_id = ?
                  AND {clause}
                  AND status = 'pending'
                  {knowledge_clause}
                """,
                parameters,
            )
            self.connection.commit()

            for action in affected:
                self.log_event(
                    kind="cancelled",
                    action_id=action.id,
                    detail=reason or "cancelled",
                    session_tag=action.session_tag,
                )

            return cursor.rowcount

        raise ValueError(
            "cancel requires action_id or source_record_id"
        )

    def reschedule(
        self,
        *,
        source_record_id: int,
        scheduled_for: datetime,
        knowledge_type: str | None = None,
        payload: dict[str, Any] | None = None,
        sources: tuple[str, ...] | None = None,
        reason: str | None = None,
        session_bucket: str = "week",
    ) -> int:
        """
        Propagate an upstream time (and optionally payload) change:
        every pending action linked to the source record moves to the
        new time.

        The session tag is re-derived from the new time (so the action
        lands in the right day/week session) and a 'rescheduled' event
        is recorded. Returns the number of actions moved.
        """
        if scheduled_for.tzinfo is None:
            scheduled_for = scheduled_for.replace(
                tzinfo=self.timezone
            )

        affected = self._linked_actions(
            source_record_id=source_record_id,
            knowledge_type=knowledge_type,
            sources=sources,
        )

        for action in affected:
            new_tag = self.session_tag_for(
                scheduled_for,
                bucket=session_bucket,
            )

            self.connection.execute(
                """
                UPDATE actions
                SET scheduled_for = ?,
                    session_tag = ?,
                    payload = ?
                WHERE id = ?
                """,
                (
                    scheduled_for.isoformat(),
                    new_tag,
                    (
                        self._encode(payload)
                        if payload is not None
                        else self._encode(action.payload)
                    ),
                    action.id,
                ),
            )
            self.connection.commit()

            self.log_event(
                kind="rescheduled",
                action_id=action.id,
                detail=(
                    f"upstream moved {action.scheduled_for.isoformat()} "
                    f"-> {scheduled_for.isoformat()}"
                    + (f" ({reason})" if reason else "")
                ),
                session_tag=new_tag,
            )

        return len(affected)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def purge_finished(
        self,
        *,
        older_than: timedelta = timedelta(days=30),
        now: datetime | None = None,
    ) -> int:
        """
        Delete fired/cancelled/failed actions older than the cutoff.
        """
        now = now or _now(self.timezone)

        cutoff = (now - older_than).isoformat()

        cursor = self.connection.execute(
            """
            DELETE FROM actions
            WHERE status IN ('fired', 'cancelled', 'failed')
              AND fired_at IS NOT NULL
              AND fired_at < ?
            """,
            (cutoff,),
        )
        self.connection.commit()

        if cursor.rowcount:
            self.log_event(
                kind="purged",
                detail=f"purged {cursor.rowcount} finished action(s)",
            )

        return cursor.rowcount

    def on_knowledge_updated(
        self,
        *,
        source_record_id: int,
        knowledge_type: str | None = None,
        sources: tuple[str, ...] | None = None,
        reason: str = "knowledge updated",
    ) -> int:
        """
        Hook for nix_knowledge / nix_core: when an upstream record
        changes, cancel every pending action tied to the old version
        and stamp the audit trail. Matches actions from any upstream
        source unless narrowed via `sources`.

        nix_core then prunes this record out of its session history so
        a stale log entry can never be mistaken for current context.
        """
        count = self.cancel(
            source_record_id=source_record_id,
            knowledge_type=knowledge_type,
            sources=sources,
            reason=f"knowledge_updated: {reason}",
        )

        self.log_event(
            kind="knowledge_updated",
            detail=(
                f"{knowledge_type or 'record'}#{source_record_id} "
                f"updated; {count} pending action(s) cancelled"
            ),
        )

        return count
