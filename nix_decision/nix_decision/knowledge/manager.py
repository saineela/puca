from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .database import KnowledgeDatabase
from .models import KnowledgeRecord
from .operations import KnowledgeOperation


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class KnowledgeManager:

    def __init__(self, database: KnowledgeDatabase):
        self.database = database

    # ---------------------------------------------------------
    # CREATE
    # ---------------------------------------------------------

    def create(
        self,
        knowledge_type: str,
        data: dict[str, Any],
        *,
        confidence: float = 1.0,
        certainty: str = "known",
        source: str = "system",
        source_id: str | None = None,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        reason: str | None = None,
    ) -> KnowledgeRecord:

        now = utc_now()

        cursor = self.database.execute(
            """
            INSERT INTO knowledge (
                knowledge_type,
                data,
                confidence,
                certainty,
                source,
                source_id,
                status,
                valid_from,
                valid_until,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                knowledge_type,
                self.database.encode(data),
                confidence,
                certainty,
                source,
                source_id,
                "active",
                valid_from.isoformat() if valid_from else None,
                valid_until.isoformat() if valid_until else None,
                now.isoformat(),
                now.isoformat(),
            ),
        )

        record_id = cursor.lastrowid

        self._audit(
            record_id=record_id,
            operation="create",
            knowledge_type=knowledge_type,
            before=None,
            after=data,
            source=source,
            reason=reason,
        )

        return self.get(record_id)

    # ---------------------------------------------------------
    # GET
    # ---------------------------------------------------------

    def get(self, record_id: int) -> KnowledgeRecord:
        row = self.database.fetchone(
            """
            SELECT *
            FROM knowledge
            WHERE id = ?
            """,
            (record_id,),
        )

        if row is None:
            raise KeyError(
                f"Knowledge record {record_id} does not exist"
            )

        return self._row_to_record(row)

    # ---------------------------------------------------------
    # SEARCH
    # ---------------------------------------------------------

    def search(
        self,
        knowledge_type: str | None = None,
        *,
        status: str = "active",
    ) -> list[KnowledgeRecord]:

        if knowledge_type:
            rows = self.database.fetchall(
                """
                SELECT *
                FROM knowledge
                WHERE knowledge_type = ?
                  AND status = ?
                ORDER BY updated_at DESC
                """,
                (knowledge_type, status),
            )
        else:
            rows = self.database.fetchall(
                """
                SELECT *
                FROM knowledge
                WHERE status = ?
                ORDER BY updated_at DESC
                """,
                (status,),
            )

        return [
            self._row_to_record(row)
            for row in rows
        ]

    # ---------------------------------------------------------
    # MATCH
    # ---------------------------------------------------------

    def find_matching(
        self,
        knowledge_type: str,
        match: dict[str, Any],
    ) -> list[KnowledgeRecord]:

        records = self.search(
            knowledge_type,
            status="active",
        )

        results = []

        for record in records:
            if self._matches(record.data, match):
                results.append(record)

        return results

    # ---------------------------------------------------------
    # UPDATE
    # ---------------------------------------------------------

    def update(
        self,
        record_id: int,
        changes: dict[str, Any],
        *,
        source: str = "system",
        reason: str | None = None,
    ) -> KnowledgeRecord:

        record = self.get(record_id)

        before = dict(record.data)
        after = dict(record.data)

        after.update(changes)

        now = utc_now()

        self.database.execute(
            """
            UPDATE knowledge
            SET data = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                self.database.encode(after),
                now.isoformat(),
                record_id,
            ),
        )

        self._audit(
            record_id=record_id,
            operation="update",
            knowledge_type=record.knowledge_type,
            before=before,
            after=after,
            source=source,
            reason=reason,
        )

        return self.get(record_id)

    # ---------------------------------------------------------
    # DELETE
    # ---------------------------------------------------------

    def delete(
        self,
        record_id: int,
        *,
        source: str = "system",
        reason: str | None = None,
    ) -> None:

        record = self.get(record_id)

        self.database.execute(
            """
            UPDATE knowledge
            SET status = 'deleted',
                updated_at = ?
            WHERE id = ?
            """,
            (
                utc_now().isoformat(),
                record_id,
            ),
        )

        self._audit(
            record_id=record_id,
            operation="delete",
            knowledge_type=record.knowledge_type,
            before=record.data,
            after=None,
            source=source,
            reason=reason,
        )

    # ---------------------------------------------------------
    # MODEL OPERATION INTERFACE
    # ---------------------------------------------------------

    def apply_operation(
        self,
        operation: KnowledgeOperation,
    ):

        operation.validate()

        if operation.operation == "create":

            return self.create(
                operation.knowledge_type,
                operation.data or {},
                confidence=operation.confidence,
                certainty=operation.certainty,
                source=operation.source,
                source_id=operation.source_id,
                reason=operation.reason,
            )

        if operation.operation == "update":

            matches = self.find_matching(
                operation.knowledge_type,
                operation.match or {},
            )

            if not matches:
                raise KeyError(
                    "No knowledge record matched update operation"
                )

            if len(matches) > 1:
                raise ValueError(
                    "Update operation matched multiple records"
                )

            return self.update(
                matches[0].id,
                operation.changes or {},
                source=operation.source,
                reason=operation.reason,
            )

        if operation.operation == "delete":

            matches = self.find_matching(
                operation.knowledge_type,
                operation.match or {},
            )

            if not matches:
                raise KeyError(
                    "No knowledge record matched delete operation"
                )

            if len(matches) > 1:
                raise ValueError(
                    "Delete operation matched multiple records"
                )

            self.delete(
                matches[0].id,
                source=operation.source,
                reason=operation.reason,
            )

            return None

        raise ValueError(
            f"Unsupported operation: {operation.operation}"
        )

    # ---------------------------------------------------------
    # AUDIT
    # ---------------------------------------------------------

    def _audit(
        self,
        *,
        record_id: int | None,
        operation: str,
        knowledge_type: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        source: str,
        reason: str | None,
    ):

        self.database.execute(
            """
            INSERT INTO knowledge_changes (
                record_id,
                operation,
                knowledge_type,
                before_data,
                after_data,
                source,
                reason,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                operation,
                knowledge_type,
                self.database.encode(before)
                if before is not None
                else None,
                self.database.encode(after)
                if after is not None
                else None,
                source,
                reason,
                utc_now().isoformat(),
            ),
        )

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _matches(
        data: dict[str, Any],
        match: dict[str, Any],
    ) -> bool:

        for key, expected in match.items():

            if data.get(key) != expected:
                return False

        return True

    @staticmethod
    def _row_to_record(row) -> KnowledgeRecord:

        from datetime import datetime

        return KnowledgeRecord(
            id=row["id"],
            knowledge_type=row["knowledge_type"],
            data=KnowledgeDatabase.decode(row["data"]),

            confidence=row["confidence"],
            certainty=row["certainty"],

            source=row["source"],
            source_id=row["source_id"],

            status=row["status"],

            valid_from=(
                datetime.fromisoformat(row["valid_from"])
                if row["valid_from"]
                else None
            ),

            valid_until=(
                datetime.fromisoformat(row["valid_until"])
                if row["valid_until"]
                else None
            ),

            created_at=datetime.fromisoformat(
                row["created_at"]
            ),

            updated_at=datetime.fromisoformat(
                row["updated_at"]
            ),
        )
