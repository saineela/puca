from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator


class KnowledgeDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path)

        self.connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
        )

        self.connection.row_factory = sqlite3.Row
        self._in_transaction = False

        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;

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

                -- Turn-level provenance for a supersede: the record that
                -- replaced this one (D1 in the PUCA system model). Nullable
                -- because rows created before the link existed stay valid.
                superseded_by INTEGER,

                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_knowledge_type
                ON knowledge(knowledge_type);

            CREATE INDEX IF NOT EXISTS idx_knowledge_status
                ON knowledge(status);

            CREATE INDEX IF NOT EXISTS idx_knowledge_valid_from
                ON knowledge(valid_from);

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

            CREATE INDEX IF NOT EXISTS idx_changes_created
                ON knowledge_changes(created_at);
            """
        )

        self._ensure_columns()
        self.connection.commit()

    def _ensure_columns(self) -> None:
        """Additive migrations, so an existing knowledge.db keeps working."""
        existing = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(knowledge)").fetchall()
        }
        if "superseded_by" not in existing:
            self.connection.execute(
                "ALTER TABLE knowledge ADD COLUMN superseded_by INTEGER"
            )

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

    @contextmanager
    def transaction(self) -> Iterator["KnowledgeDatabase"]:
        """Group several writes so a supersede link can never dangle."""
        if self._in_transaction:
            yield self
            return
        self._in_transaction = True
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            self._in_transaction = False

    def execute(self, query: str, parameters=()):
        cursor = self.connection.execute(query, parameters)
        if not self._in_transaction:
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

    def close(self) -> None:
        self.connection.close()
