from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .database import KnowledgeDatabase
from .models import KnowledgeRecord


class KnowledgeRetrieval:

    def __init__(self, database: KnowledgeDatabase):
        self.database = database

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

    def matching(
        self,
        knowledge_type: str,
        match: dict[str, Any],
    ) -> list[KnowledgeRecord]:

        records = self.search(
            knowledge_type,
            status="active",
        )

        return [
            record
            for record in records
            if self._matches(record, match)
        ]

    def relevant_now(
        self,
        knowledge_type: str | None = None,
    ) -> list[KnowledgeRecord]:

        now = datetime.now(timezone.utc)

        records = self.search(
            knowledge_type,
            status="active",
        )

        results = []

        for record in records:

            if (
                record.valid_from
                and now < record.valid_from
            ):
                continue

            if (
                record.valid_until
                and now >= record.valid_until
            ):
                continue

            results.append(record)

        return results

    @staticmethod
    def _matches(
        record: KnowledgeRecord,
        match: dict[str, Any],
    ) -> bool:

        for key, expected in match.items():

            if key == "id":
                if record.id != expected:
                    return False
                continue

            if record.data.get(key) != expected:
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
