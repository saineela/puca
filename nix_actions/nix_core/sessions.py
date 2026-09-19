from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

DEFAULT_TZ = "America/Chicago"

SESSION_BUCKETS = ("day", "week")


def session_tag_for(when: datetime, *, bucket: str = "week") -> str:
    """
    Session tags shared with nix_actions so both components speak the
    same language: day-YYYY-MM-DD and Monday-based week-YYYY-MM-DD.
    """
    if bucket == "day":
        return f"day-{when.date().isoformat()}"

    if bucket == "week":
        monday = when.date() - timedelta(days=when.weekday())
        return f"week-{monday.isoformat()}"

    raise ValueError(
        f"Unsupported session bucket: {bucket} (use 'day' or 'week')"
    )


@dataclass(frozen=True)
class SessionTurn:
    """One conversational turn inside a session."""

    id: int
    session_tag: str
    role: str  # user | assistant | system
    content: str

    created_at: datetime

    # Set when the referenced knowledge was updated after this turn was
    # logged. Pruned turns are never replayed into context again.
    pruned: bool
    pruned_reason: str | None

    refs: dict[str, Any] = field(default_factory=dict)


class SessionStore:
    """
    Conversation sessions, bucketed by day or week.

    Sessions are cleared when their bucket rolls over; the knowledge
    base never clears. Turns that referenced knowledge which was later
    updated are pruned so a stale log entry can never be mistaken for
    current context.
    """

    def __init__(
        self,
        database_path: str | Path = "nix_core.db",
        timezone: str = DEFAULT_TZ,
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
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                session_tag TEXT NOT NULL,

                role TEXT NOT NULL,
                content TEXT NOT NULL,

                refs TEXT NOT NULL DEFAULT '{}',

                pruned INTEGER NOT NULL DEFAULT 0,
                pruned_reason TEXT,

                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_turns_session
                ON turns(session_tag, id);

            CREATE INDEX IF NOT EXISTS idx_turns_created
                ON turns(created_at);
            """
        )
        self.connection.commit()

    @staticmethod
    def _encode(data: Any) -> str:
        return json.dumps(data, ensure_ascii=False, default=str)

    def close(self) -> None:
        self.connection.close()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_turn(
        self,
        *,
        role: str,
        content: str,
        refs: dict[str, Any] | None = None,
        when: datetime | None = None,
        session_bucket: str = "week",
    ) -> SessionTurn:
        """
        Log one conversational turn into its session bucket.

        refs is free-form; nix_core writes keys like
        {"knowledge_record_ids": [7], "action_ids": [3]} so pruning and
        dashboards can trace what a turn touched.
        """
        if role not in {"user", "assistant", "system"}:
            raise ValueError(
                f"Invalid role: {role} (use user, assistant or system)"
            )

        moment = when or datetime.now(self.timezone)
        tag = session_tag_for(moment, bucket=session_bucket)

        cursor = self.connection.execute(
            """
            INSERT INTO turns
                (session_tag, role, content, refs, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                tag,
                role,
                content,
                self._encode(refs or {}),
                moment.isoformat(),
            ),
        )
        self.connection.commit()

        return self.get_turn(int(cursor.lastrowid or 0))

    def get_turn(self, turn_id: int) -> SessionTurn:
        row = self.connection.execute(
            "SELECT * FROM turns WHERE id = ?",
            (turn_id,),
        ).fetchone()

        if row is None:
            raise KeyError(f"Turn {turn_id} does not exist")

        return self._row_to_turn(row)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @staticmethod
    def current_tags(
        now: datetime | None = None,
        *,
        timezone: ZoneInfo | None = None,
    ) -> dict[str, str]:
        """The day and week tags covering `now`."""
        tz = timezone or ZoneInfo(DEFAULT_TZ)
        moment = now or datetime.now(tz)
        return {
            "day": session_tag_for(moment, bucket="day"),
            "week": session_tag_for(moment, bucket="week"),
        }

    def list_turns(
        self,
        *,
        session_tag: str | None = None,
        include_pruned: bool = False,
        limit: int = 200,
    ) -> list[SessionTurn]:
        query = "SELECT * FROM turns"
        conditions = []
        parameters: list[Any] = []

        if session_tag is not None:
            conditions.append("session_tag = ?")
            parameters.append(session_tag)

        if not include_pruned:
            conditions.append("pruned = 0")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY id DESC LIMIT ?"
        parameters.append(limit)

        rows = self.connection.execute(
            query, parameters
        ).fetchall()

        return [self._row_to_turn(row) for row in reversed(rows)]

    def context_window(
        self,
        *,
        session_bucket: str = "week",
        limit: int = 50,
        now: datetime | None = None,
    ) -> list[SessionTurn]:
        """
        The turns nix_core may treat as context: only the current
        session, only un-pruned turns.
        """
        tz = self.timezone
        moment = now or datetime.now(tz)
        tag = session_tag_for(moment, bucket=session_bucket)

        return self.list_turns(
            session_tag=tag,
            include_pruned=False,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_turns_referencing(
        self,
        *,
        source_record_id: int,
        knowledge_type: str | None = None,
        reason: str = "knowledge updated",
    ) -> int:
        """
        Mark every turn that references an upstream knowledge record as
        pruned. Pruned turns stay on disk for audit purposes but are
        excluded from context_window() forever.
        """
        rows = self.connection.execute(
            "SELECT id, refs FROM turns WHERE pruned = 0"
        ).fetchall()

        affected: list[int] = []
        for row in rows:
            refs = json.loads(row["refs"] or "{}")
            record_ids = refs.get("knowledge_record_ids") or []

            if source_record_id not in record_ids:
                continue

            # A turn that listed the record id without naming its type
            # is pruned too: when in doubt, stale context loses.
            turn_type = refs.get("knowledge_type")
            if (
                knowledge_type is not None
                and turn_type is not None
                and turn_type != knowledge_type
            ):
                continue

            affected.append(row["id"])

        for turn_id in affected:
            self.connection.execute(
                """
                UPDATE turns
                SET pruned = 1,
                    pruned_reason = ?
                WHERE id = ?
                """,
                (reason, turn_id),
            )

        self.connection.commit()
        return len(affected)

    # ------------------------------------------------------------------
    # Session clearing
    # ------------------------------------------------------------------

    def clear_stale_sessions(
        self,
        *,
        now: datetime | None = None,
    ) -> int:
        """
        Delete sessions whose bucket has fully rolled over. The current
        day and week sessions are never touched. 1 week is a week
        session; every week gets cleared when the next one starts.
        """
        moment = now or datetime.now(self.timezone)
        current = self.current_tags(moment, timezone=self.timezone)

        rows = self.connection.execute(
            "SELECT DISTINCT session_tag FROM turns"
        ).fetchall()

        removed = 0
        for row in rows:
            tag = row["session_tag"]

            if tag in (current["day"], current["week"]):
                continue

            cursor = self.connection.execute(
                "DELETE FROM turns WHERE session_tag = ?",
                (tag,),
            )
            removed += cursor.rowcount

        self.connection.commit()
        return removed

    def session_stats(self) -> list[dict[str, Any]]:
        """Per-session counts, oldest first."""
        rows = self.connection.execute(
            """
            SELECT session_tag,
                   COUNT(*)                    AS total,
                   SUM(pruned = 0)             AS active,
                   SUM(pruned = 1)             AS pruned,
                   MIN(created_at)             AS first_turn,
                   MAX(created_at)             AS last_turn
            FROM turns
            GROUP BY session_tag
            ORDER BY session_tag ASC
            """
        ).fetchall()

        return [
            {
                "session_tag": row["session_tag"],
                "bucket": (
                    "week"
                    if row["session_tag"].startswith("week-")
                    else "day"
                ),
                "total": row["total"],
                "active": row["active"] or 0,
                "pruned": row["pruned"] or 0,
                "first_turn": row["first_turn"],
                "last_turn": row["last_turn"],
            }
            for row in rows
        ]

    @staticmethod
    def _row_to_turn(row) -> SessionTurn:
        from datetime import datetime as _dt

        return SessionTurn(
            id=row["id"],
            session_tag=row["session_tag"],
            role=row["role"],
            content=row["content"],
            created_at=_dt.fromisoformat(row["created_at"]),
            pruned=bool(row["pruned"]),
            pruned_reason=row["pruned_reason"],
            refs=json.loads(row["refs"] or "{}"),
        )
