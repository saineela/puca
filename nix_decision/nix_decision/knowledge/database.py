from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class KnowledgeDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path)

        self.connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
        )

        self.connection.row_factory = sqlite3.Row

        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                knowledge_type TEXT NOT NULL,

                data TEXT NOT NULL,

                confidence REAL NOT NULL DEFAULT 1.0,
                certainty TEXT NOT NULL DEFAULT 'known',

                source TEXT NOT NULL,
                source_id TEXT,

                status TEXT NOT NULL DEFAULT 'active',

                valid_from TEXT,
                valid_until TEXT,

                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_knowledge_type
                ON knowledge(knowledge_type);

            CREATE INDEX IF NOT EXISTS idx_knowledge_status
                ON knowledge(status);

            CREATE TABLE IF NOT EXISTS knowledge_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                record_id INTEGER,

                operation TEXT NOT NULL,
                knowledge_type TEXT NOT NULL,

                before_data TEXT,
                after_data TEXT,

                source TEXT NOT NULL,
                reason TEXT,

                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_changes_record
                ON knowledge_changes(record_id);
            """
        )

        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    @staticmethod
    def encode(data: Any) -> str:
        return json.dumps(
            data,
            ensure_ascii=False,
            default=str,
        )

    @staticmethod
    def decode(data: str | None) -> Any:
        if data is None:
            return None

        return json.loads(data)

    def execute(self, query: str, parameters=()):
        cursor = self.connection.execute(query, parameters)
        self.connection.commit()
        return cursor

    def fetchone(self, query: str, parameters=()):
        return self.connection.execute(
            query,
            parameters,
        ).fetchone()

    def fetchall(self, query: str, parameters=()):
        return self.connection.execute(
            query,
            parameters,
        ).fetchall()
