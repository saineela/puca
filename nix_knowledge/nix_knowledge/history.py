from __future__ import annotations

from .database import KnowledgeDatabase
from .models import KnowledgeChange


class KnowledgeHistory:

    def __init__(self, database: KnowledgeDatabase):
        self.database = database

    def list_changes(
        self,
        record_id: int | None = None,
    ) -> list[KnowledgeChange]:

        if record_id is None:
            rows = self.database.fetchall(
                """
                SELECT *
                FROM knowledge_changes
                ORDER BY created_at DESC
                """
            )
        else:
            rows = self.database.fetchall(
                """
                SELECT *
                FROM knowledge_changes
                WHERE record_id = ?
                ORDER BY created_at DESC
                """,
                (record_id,),
            )

        return [
            KnowledgeChange(
                id=row["id"],
                record_id=row["record_id"],
                operation=row["operation"],
                knowledge_type=row["knowledge_type"],
                before=self.database.decode(
                    row["before_data"]
                ),
                after=self.database.decode(
                    row["after_data"]
                ),
                source=row["source"],
                reason=row["reason"],
                created_at=__import__(
                    "datetime"
                ).datetime.fromisoformat(
                    row["created_at"]
                ),
            )
            for row in rows
        ]
