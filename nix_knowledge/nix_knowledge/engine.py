from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .database import KnowledgeDatabase
from .history import KnowledgeHistory
from .models import KnowledgeRecord
from .operations import KnowledgeOperation
from .retrieval import KnowledgeRetrieval
from .validation import KnowledgeValidator

try:
    from .semantic.service import SemanticService
except ImportError:  # optional dependency set (torch etc.)
    SemanticService = None


class KnowledgeEngine:

    def __init__(
        self,
        database_path: str | Path = "knowledge.db",
    ):
        self.database_path = str(database_path)
        self.database = KnowledgeDatabase(database_path)

        self.validator = KnowledgeValidator()
        self.retrieval = KnowledgeRetrieval(
            self.database
        )
        self.history = KnowledgeHistory(
            self.database
        )

        # Neural semantic layer (vector search) -- optional at import
        # time so the engine still works on a bare install. The vector
        # store MUST be scoped to this engine's database: a shared
        # global store would leak vectors across engines and make
        # duplicate detection match unrelated records. Vector blobs
        # stay OUT of the record DB (embedded blobs in SQLite slow
        # every query), so sibling files are used:
        #   knowledge.db -> knowledge_vectors.db + knowledge_entities.db
        # Construction is LAZY (first semantic access): instantiating
        # the embedder loads a model, which plain CRUD engines and
        # test fixtures should never pay for.
        self._semantic: SemanticService | None = None
        self._semantic_failed = False

    # ---------------------------------------------------------
    # SEMANTIC LAYER (lazy)
    # ---------------------------------------------------------

    @property
    def semantic(self) -> SemanticService | None:
        """Engine-scoped neural semantic layer, built on first use.

        Returns None when torch/sentence-transformers are missing or
        the model failed to load - callers already handle a None
        semantic layer by degrading to plain CRUD behavior.
        """
        if self._semantic is not None or self._semantic_failed:
            return self._semantic
        if SemanticService is None:
            self._semantic_failed = True
            return None
        try:
            db_str = str(self.database_path)
            if db_str.endswith(".db"):
                vec_path = db_str[:-3] + "_vectors.db"
                ent_path = db_str[:-3] + "_entities.db"
            else:
                vec_path = db_str + "_vectors.db"
                ent_path = db_str + "_entities.db"
            self._semantic = SemanticService(
                vec_path, entity_path=ent_path
            )
        except Exception:
            self._semantic = None
            self._semantic_failed = True
        return self._semantic

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

        operation = KnowledgeOperation(
            operation="create",
            knowledge_type=knowledge_type,
            data=data,
            confidence=confidence,
            certainty=certainty,
            source=source,
            source_id=source_id,
            reason=reason,
            valid_from=(
                valid_from.isoformat()
                if valid_from
                else None
            ),
            valid_until=(
                valid_until.isoformat()
                if valid_until
                else None
            ),
        )

        return self.apply(operation)

    # ---------------------------------------------------------
    # GET
    # ---------------------------------------------------------

    def get(self, record_id: int) -> KnowledgeRecord:
        return self.retrieval.get(record_id)

    # ---------------------------------------------------------
    # SEARCH
    # ---------------------------------------------------------

    def search(
        self,
        knowledge_type: str | None = None,
        *,
        status: str = "active",
    ) -> list[KnowledgeRecord]:

        return self.retrieval.search(
            knowledge_type,
            status=status,
        )

    # ---------------------------------------------------------
    # SEMANTIC (VECTOR) SEARCH -- primary search path
    # ---------------------------------------------------------

    def semantic_search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict | None = None,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        Hybrid neural search (vector + BM25 + temporal).

        Returns ranked hits with record data, fused score and the
        semantic classification of each chunk:
            [{"record_id", "text", "score", "metadata", "record"}]
        """
        if self.semantic is None:
            return []

        hits = self.semantic.search(
            query,
            limit=limit,
            filters=filters,
            min_score=min_score,
        )

        results = []
        for hit in hits:
            try:
                record = self.get(hit.record_id)
            except KeyError:
                # knowledge record was deleted; drop stale vector
                self.semantic.delete_record(hit.record_id)
                continue
            results.append(
                {
                    "record_id": hit.record_id,
                    "chunk_index": hit.chunk_index,
                    "text": hit.text,
                    "score": hit.score,
                    "components": hit.components,
                    "metadata": hit.metadata,
                    "record": record,
                }
            )
        return results

    def observe_and_store(self, text: str) -> dict:
        """
        Perception pass: classify, gate, and persist knowledge the
        user stated. Returns what was stored/rejected/escalated.
        """
        if self.semantic is None:
            return {"stored": [], "rejected": [], "escalated": [],
                    "indexed_chunks": 0, "contradictions": []}
        return self.semantic.observe_and_store(text)

    def evidence_pack(self, claim: str, **kwargs) -> dict:
        """
        Evidence for/against a claim (grounded argumentation support
        for Core): supporting + contradicting records and a verdict.
        """
        if self.semantic is None:
            return {"claim": claim, "verdict": "insufficient_evidence",
                    "confidence": 0.0, "supporting": [],
                    "contradicting": []}
        return self.semantic.evidence_pack(claim, **kwargs)

    # ---------------------------------------------------------
    # CONTEXT
    # ---------------------------------------------------------

    def relevant_now(
        self,
        knowledge_type: str | None = None,
    ) -> list[KnowledgeRecord]:

        return self.retrieval.relevant_now(
            knowledge_type
        )

    # ---------------------------------------------------------
    # APPLY MODEL OPERATION
    # ---------------------------------------------------------

    def apply(
        self,
        operation: KnowledgeOperation,
    ):

        self.validator.validate(operation)

        if operation.operation == "create":
            return self._create(operation)

        if operation.operation == "update":
            return self._update(operation)

        if operation.operation == "delete":
            return self._delete(operation)

        raise ValueError(
            f"Unsupported operation: {operation.operation}"
        )

    # ---------------------------------------------------------
    # INTERNAL CREATE
    # ---------------------------------------------------------

    def _create(
        self,
        operation: KnowledgeOperation,
    ):

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

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
                operation.knowledge_type,
                self.database.encode(
                    operation.data or {}
                ),
                operation.confidence,
                operation.certainty,
                operation.source,
                operation.source_id,
                "active",
                operation.valid_from,
                operation.valid_until,
                now.isoformat(),
                now.isoformat(),
            ),
        )

        record_id = cursor.lastrowid

        self._audit(
            record_id=record_id,
            operation="create",
            knowledge_type=operation.knowledge_type,
            before=None,
            after=operation.data,
            source=operation.source,
            reason=operation.reason,
        )

        self._sync_semantic(
            record_id=record_id,
            data=operation.data or {},
        )

        return self.get(record_id)

    # ---------------------------------------------------------
    # INTERNAL UPDATE
    # ---------------------------------------------------------

    def _update(
        self,
        operation: KnowledgeOperation,
    ):

        matches = self.retrieval.matching(
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

        record = matches[0]

        before = dict(record.data)
        after = dict(record.data)

        after.update(
            operation.changes or {}
        )

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        self.database.execute(
            """
            UPDATE knowledge
            SET data = ?,
                confidence = ?,
                certainty = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                self.database.encode(after),
                operation.confidence,
                operation.certainty,
                now.isoformat(),
                record.id,
            ),
        )

        self._audit(
            record_id=record.id,
            operation="update",
            knowledge_type=record.knowledge_type,
            before=before,
            after=after,
            source=operation.source,
            reason=operation.reason,
        )

        self._sync_semantic(
            record_id=record.id,
            data=after,
        )

        return self.get(record.id)

    # ---------------------------------------------------------
    # INTERNAL DELETE
    # ---------------------------------------------------------

    def _delete(
        self,
        operation: KnowledgeOperation,
    ):

        matches = self.retrieval.matching(
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

        record = matches[0]

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        self.database.execute(
            """
            UPDATE knowledge
            SET status = 'deleted',
                updated_at = ?
            WHERE id = ?
            """,
            (
                now.isoformat(),
                record.id,
            ),
        )

        self._audit(
            record_id=record.id,
            operation="delete",
            knowledge_type=record.knowledge_type,
            before=record.data,
            after=None,
            source=operation.source,
            reason=operation.reason,
        )

        if self.semantic is not None:
            self.semantic.delete_record(record.id)

        return None

    # ---------------------------------------------------------
    # SEMANTIC SYNC
    # ---------------------------------------------------------

    def _sync_semantic(self, *, record_id: int, data: dict) -> None:
        """Keep the vector index in step with the knowledge table."""
        if self.semantic is None:
            return
        try:
            text = semantic_record_text(data)
            if text:
                self.semantic.ingest(record_id, text)
        except Exception:
            # Semantic indexing must never break knowledge writes;
            # backfill repairs any gap later.
            pass

    # ---------------------------------------------------------
    # AUDIT
    # ---------------------------------------------------------

    def _audit(
        self,
        *,
        record_id: int | None,
        operation: str,
        knowledge_type: str,
        before,
        after,
        source: str,
        reason: str | None,
    ):

        from datetime import datetime, timezone

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
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def close(self):
        # use the private attribute: constructing the semantic layer
        # just to close it would defeat the lazy design
        if self._semantic is not None:
            try:
                self._semantic.close()
            except Exception:
                pass
        self.database.close()


def semantic_record_text(data: dict) -> str:
    """
    Flatten a knowledge record's data dict into indexable text.
    Keys are included so short values ("Austin") still embed well.
    """
    parts = []
    for key, value in data.items():
        if value is None:
            continue
        parts.append(f"{key.replace('_', ' ')}: {value}")
    return ". ".join(parts)


def main():
    engine = KnowledgeEngine()

    print("Nix Knowledge Engine v0.1")
    print(f"Database: {engine.database.path}")

    records = engine.search()

    print(f"Knowledge records: {len(records)}")

    engine.close()


if __name__ == "__main__":
    main()
