from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from nix_actions.engine import ActionsEngine

from .knowledge import KnowledgeProvider, KnowledgeRecord, StaticKnowledgeProvider
from .sessions import SessionStore, SessionTurn


@dataclass(frozen=True)
class CoreResponse:
    """Everything one request produced, for the caller and the logs."""

    reply: str
    session_tag: str
    turn_id: int

    knowledge_used: list[KnowledgeRecord]
    actions_captured: list[int]

    # True when this request triggered pruning of stale session turns
    pruned_turns: int

    metadata: dict[str, Any]


class NixCore:
    """
    The conversational brain.

    - holds ONLY session history (day/week sessions, cleared on
      schedule); it never stores durable knowledge
    - answers requests by pulling from a KnowledgeProvider
    - schedules outcomes by capturing actions into nix_actions
    - when knowledge is updated upstream, prunes the stale turns out
      of its own history and cancels the linked actions
    """

    def __init__(
        self,
        *,
        knowledge: KnowledgeProvider | None = None,
        actions: ActionsEngine | None = None,
        sessions: SessionStore | None = None,
        database_path: str | Path = "nix_core.db",
        actions_database_path: str | Path = "actions.db",
        timezone: str = "America/Chicago",
        session_bucket: str = "week",
        event_log_path: str | Path | None = None,
    ):
        self.session_bucket = session_bucket

        self.sessions = sessions or SessionStore(
            database_path,
            timezone=timezone,
        )

        self.actions = actions or ActionsEngine(
            actions_database_path,
            timezone=timezone,
        )

        self.knowledge: KnowledgeProvider = (
            knowledge or StaticKnowledgeProvider()
        )

        self.event_log_path = (
            Path(event_log_path) if event_log_path else None
        )

    def close(self) -> None:
        self.sessions.close()
        self.actions.close()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, kind: str, **fields: Any) -> None:
        entry = {
            "kind": kind,
            "at": datetime.now(self.sessions.timezone).isoformat(),
            **fields,
        }
        if self.event_log_path:
            with self.event_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def handle_request(
        self,
        *,
        text: str,
        role: str = "user",
        session_bucket: str | None = None,
    ) -> CoreResponse:
        """
        One conversational turn.

        1. log the turn into the current session
        2. pull knowledge for it (nix_core holds no facts itself)
        3. check the pulled records are still current; if any record
           changed upstream, prune the stale turns out of session
           history and cancel the linked actions so the old record can
           never be mistaken for present context
        4. draft a reply from whatever knowledge remains
        5. capture scheduled actions into nix_actions
        """
        bucket = session_bucket or self.session_bucket

        turn = self.sessions.add_turn(
            role=role,
            content=text,
            refs={},
            session_bucket=bucket,
        )

        pulled = self.knowledge.lookup(query=text, limit=5)

        pruned = self._reconcile_knowledge(pulled)

        live = [
            record
            for record in pulled
            if record.version == self.knowledge.version_of(
                knowledge_type=record.knowledge_type,
                record_id=record.id,
            )
        ]

        self.sessions.connection.execute(
            """
            UPDATE turns
            SET refs = ?
            WHERE id = ?
            """,
            (
                json.dumps(
                    {
                        "knowledge_type": (
                            live[0].knowledge_type if live else None
                        ),
                        "knowledge_record_ids": [
                            record.id for record in live
                        ],
                        "knowledge_versions": {
                            f"{r.knowledge_type}#{r.id}": r.version
                            for r in live
                        },
                    }
                ),
                turn.id,
            ),
        )
        self.sessions.connection.commit()

        reply = self._draft_reply(text, live)
        reply_turn = self.sessions.add_turn(
            role="assistant",
            content=reply,
            refs={
                "in_reply_to": turn.id,
                "knowledge_type": (
                    live[0].knowledge_type if live else None
                ),
                "knowledge_record_ids": [r.id for r in live],
            },
            session_bucket=bucket,
        )

        action_ids = self._capture_actions(text, live, reply_turn)

        self._log(
            "request",
            turn_id=turn.id,
            session=reply_turn.session_tag,
            knowledge=[r.id for r in live],
            pruned_turns=pruned,
            actions=action_ids,
        )

        return CoreResponse(
            reply=reply,
            session_tag=reply_turn.session_tag,
            turn_id=reply_turn.id,
            knowledge_used=live,
            actions_captured=action_ids,
            pruned_turns=pruned,
            metadata={
                "bucket": bucket,
                "pruned": pruned,
            },
        )

    # ------------------------------------------------------------------
    # Knowledge reconciliation
    # ------------------------------------------------------------------

    def _reconcile_knowledge(
        self,
        pulled: list[KnowledgeRecord],
    ) -> int:
        """
        If any pulled record is stale (upstream updated it), prune the
        old turns out of session history AND cancel the actions tied
        to the old record version, then log the event.
        """
        pruned = 0

        for record in pulled:
            current_version = self.knowledge.version_of(
                knowledge_type=record.knowledge_type,
                record_id=record.id,
            )

            if current_version == record.version:
                continue

            pruned += self.sessions.prune_turns_referencing(
                source_record_id=record.id,
                knowledge_type=record.knowledge_type,
                reason=(
                    f"{record.knowledge_type}#{record.id} updated "
                    f"v{record.version} -> v{current_version}"
                ),
            )

            cancelled = self.actions.cancel(
                source_record_id=record.id,
                knowledge_type=record.knowledge_type,
                reason=(
                    f"knowledge_updated: {record.knowledge_type} "
                    f"changed version"
                ),
            )

            self._log(
                "knowledge_stale",
                knowledge_type=record.knowledge_type,
                record_id=record.id,
                old_version=record.version,
                new_version=current_version,
                pruned_turns=pruned,
                cancelled_actions=cancelled,
            )

        return pruned

    # ------------------------------------------------------------------
    # Drafting (deterministic stand-in for the future LLM/brain)
    # ------------------------------------------------------------------

    def _draft_reply(
        self,
        text: str,
        records: list[KnowledgeRecord],
    ) -> str:
        if not records:
            return (
                "I don't have knowledge that covers that yet. "
                "The request is saved to this session."
            )

        lines = [f"About \"{text}\":"]
        for record in records:
            lines.append(
                f"- [{record.knowledge_type}#{record.id} v"
                f"{record.version}] {record.content}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Action capture
    # ------------------------------------------------------------------

    def _capture_actions(
        self,
        text: str,
        records: list[KnowledgeRecord],
        turn: SessionTurn,
    ) -> list[int]:
        """
        Deterministic intent mapping: request text containing a time
        phrase becomes a reminder action in nix_actions, linked to the
        source record so upstream changes propagate.

        This is the seam the future brain replaces; nix_actions does
        not change when it does.
        """
        action_ids: list[int] = []

        source_record_id = records[0].id if records else None
        knowledge_type = records[0].knowledge_type if records else None

        if source_record_id is None:
            return action_ids

        lowered = text.lower()
        if "remind" not in lowered and "alarm" not in lowered:
            return action_ids

        action = self.actions.capture(
            action_type="reminder" if "remind" in lowered else "alarm",
            scheduled_for=datetime.now(self.actions.timezone),
            payload={"message": text, "via": "nix_core"},
            source="nix_core",
            source_record_id=source_record_id,
            knowledge_type=knowledge_type,
            session_bucket=self.session_bucket,
        )
        action_ids.append(action.id)

        return action_ids

    # ------------------------------------------------------------------
    # Upstream propagation entry point
    # ------------------------------------------------------------------

    def on_knowledge_updated(
        self,
        *,
        knowledge_type: str,
        source_record_id: int,
        reason: str = "knowledge updated",
    ) -> dict[str, int]:
        """
        Hook nix_knowledge calls when a record changes.

        nix_actions cancels the actions linked to the old record, and
        nix_core prunes the turns that referenced it out of session
        history - so a stale log entry can never be mistaken for
        current context.
        """
        cancelled = self.actions.on_knowledge_updated(
            knowledge_type=knowledge_type,
            source_record_id=source_record_id,
            reason=reason,
        )

        pruned = self.sessions.prune_turns_referencing(
            source_record_id=source_record_id,
            knowledge_type=knowledge_type,
            reason=reason,
        )

        self._log(
            "knowledge_updated",
            knowledge_type=knowledge_type,
            record_id=source_record_id,
            cancelled_actions=cancelled,
            pruned_turns=pruned,
        )

        return {"cancelled_actions": cancelled, "pruned_turns": pruned}

    def reschedule(
        self,
        *,
        source_record_id: int,
        scheduled_for: datetime,
        knowledge_type: str | None = None,
        reason: str | None = None,
    ) -> int:
        """Pass-through so upstream moves propagate from one place."""
        return self.actions.reschedule(
            source_record_id=source_record_id,
            scheduled_for=scheduled_for,
            knowledge_type=knowledge_type,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def maintain(self) -> dict[str, int]:
        """
        Run housekeeping: clear rolled-over sessions, purge old
        finished actions. Safe to call on every request or from cron.
        """
        cleared = self.sessions.clear_stale_sessions()
        purged = self.actions.purge_finished()

        if cleared or purged:
            self._log(
                "maintenance",
                cleared_turns=cleared,
                purged_actions=purged,
            )

        return {"cleared_turns": cleared, "purged_actions": purged}
