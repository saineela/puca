from __future__ import annotations

"""
SemanticService: the front door of the neural semantic layer.

Owns the pipeline (chunker -> embedder -> store -> search ->
classifier -> extractor) and exposes:

    ingest(record_id, text)     index + classify + extract
    search(query, filters)      hybrid vector/BM25/temporal search
    delete_record(record_id)    remove a record's vectors
    stats()                     index health

The classifier is optional at runtime: search works without a
trained head; ingestion simply stores chunks with empty metadata
until a classifier is available (backfill re-enriches later).
"""

import threading
from typing import Any

from .chunker import Chunk, Chunker
from .classifier import SemanticClassifier
from .embedder import Embedder
from .entities import EntityRegistry, extract_entities, find_referents
from .extraction import DUP_THRESHOLD, ExtractionReport, KnowledgeExtractor
from .gate import StorageGate
from .search import HybridSearch, SearchHit
from .vector_store import VectorStore


class SemanticService:
    def __init__(
        self,
        db_path: str | None = None,
        *,
        embedder: Embedder | None = None,
        load_classifier: bool = True,
        entity_path: str | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.store = VectorStore(db_path)
        # entity registry lives next to the vector store
        if entity_path is None:
            if db_path and db_path != ":memory:" and str(db_path).endswith(
                ".db"
            ):
                entity_path = str(db_path).replace(".db", "_entities.db")
            else:
                entity_path = None
        self.entities = EntityRegistry(
            entity_path or "semantic_entities.db"
        )
        self.searcher = HybridSearch(self.store)
        self.chunker = Chunker()

        self.classifier: SemanticClassifier | None = None
        if load_classifier:
            candidate = SemanticClassifier(self.embedder)
            if candidate.load():
                self.classifier = candidate

        self.gate = StorageGate()
        self.extractor = KnowledgeExtractor(
            classifier=self.classifier,
            store=self.store,
            gate=self.gate,
        )

        self._ingest_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, record_id: int, text: str) -> dict[str, Any]:
        """
        Chunk, embed, classify, and index one knowledge record.

        Returns a summary dict (chunk count, classification preview).
        """
        with self._ingest_lock:
            metadata: dict[str, Any] = {}
            if self.classifier is not None:
                result = self.classifier.classify(text)
                metadata = {
                    "emotion": [e[0] for e in result["emotion"][:3]],
                    "tone": result["tone"][0],
                    "importance": result["importance"],
                    "category": result["category"][0],
                }

            # register (role, name) entities found in the statement so
            # later referential queries ("my sister") resolve to names
            for entity in extract_entities(text):
                self.entities.register(
                    entity.role,
                    entity.name,
                    record_id=record_id,
                    evidence=text[:160],
                )

            chunks = self.chunker.chunk_record(
                record_id, text, metadata=metadata
            )
            if not chunks:
                return {"chunks": 0}

            vectors = self.embedder.encode(
                [c.text for c in chunks]
            )
            written = self.store.upsert(chunks, vectors)

            return {
                "chunks": written,
                "metadata": metadata,
            }

    def ingest_batch(
        self, items: list[tuple[int, str]]
    ) -> dict[str, Any]:
        """Efficient batched ingestion of (record_id, text) pairs."""
        all_chunks: list[Chunk] = []
        for record_id, text in items:
            metadata = {}
            if self.classifier is not None:
                result = self.classifier.classify(text)
                metadata = {
                    "emotion": [e[0] for e in result["emotion"][:3]],
                    "tone": result["tone"][0],
                    "importance": result["importance"],
                    "category": result["category"][0],
                }
            all_chunks.extend(
                self.chunker.chunk_record(
                    record_id, text, metadata=metadata
                )
            )

        if not all_chunks:
            return {"chunks": 0}

        vectors = self.embedder.encode(
            [c.text for c in all_chunks]
        )
        written = self.store.upsert(all_chunks, vectors)
        return {"chunks": written}

    def check_duplicate(self, text: str) -> tuple[int, str] | None:
        """
        Return (record_id, stored_text) when `text` is a near-exact
        duplicate of an already-indexed chunk (cosine >= threshold),
        else None. Used by the create_fact path so re-stating a fact
        never writes a second record.
        """
        if not text.strip():
            return None
        vectors = self.embedder.encode([text])
        hits = self.store.search(vectors[0], limit=1)
        if hits and hits[0].score >= DUP_THRESHOLD:
            return hits[0].record_id, hits[0].text
        return None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        """
        Hybrid semantic search with referent resolution.

        Referential phrases ("my sister") are resolved through the
        entity registry and the resolved names are folded into the
        query, so records containing the actual name rank first.

        filters: exact metadata filters, e.g.
            {"tone": "urgent"}
            {"emotion": ["joy", "gratitude"]}
            {"category": "personal_fact", "min_importance": 0.7}
        """
        # ---- referent resolution ------------------------------------
        resolved: list[tuple[str, str]] = []
        for role in find_referents(query):
            name = self.entities.get(role)
            if name:
                resolved.append((role, name))

        effective_query = query
        if resolved:
            augmentation = ". " + ", ".join(
                f"{role}: {name}" for role, name in resolved
            )
            effective_query = query + augmentation

        query_vector = self.embedder.encode_one(effective_query)

        min_importance = None
        if filters:
            filters = dict(filters)
            min_importance = filters.pop("min_importance", None)

        base_filter = self.searcher._build_filter(filters or {})

        def combined(uid: str, record_id: int, metadata: dict) -> bool:
            if base_filter is not None and not base_filter(
                uid, record_id, metadata
            ):
                return False
            if min_importance is not None:
                if float(metadata.get("importance", 0.5)) < float(
                    min_importance
                ):
                    return False
            return True

        return self.searcher.search(
            effective_query,
            query_vector,
            limit=limit,
            min_score=min_score,
            filters=None,
            filter_override=combined,
        )

    def extract(self, record_id: int, text: str) -> ExtractionReport:
        """Run extraction (facts/dedup/contradictions) without storing."""
        return self.extractor.extract(record_id, text)

    # ------------------------------------------------------------------
    # Observation pipeline (perception for Core)
    # ------------------------------------------------------------------

    def observe(self, text: str) -> dict[str, Any]:
        """
        Full perception pass over something the user said.

        Returns classification for EVERY chunk plus storage decisions,
        so Core always knows what was learned, what was ignored, what
        was flagged as figurative, and what needs confirmation.
        """
        report = self.extractor.extract(record_id=0, text=text)

        stored, rejected, escalated = [], [], []
        for candidate, decision in zip(
            report.candidates, report.decisions
        ):
            entry = {
                "text": candidate.text,
                "category": candidate.category,
                "tone": candidate.tone,
                "importance": candidate.importance,
                "emotion": candidate.emotion[:3],
                "sarcasm": candidate.sarcasm_probability,
                "person": (candidate.metadata or {}).get("person"),
                "fact": candidate.fact,
                "reason": decision.reason,
            }
            if decision.decision == "store":
                stored.append(entry)
            elif decision.decision == "escalate":
                escalated.append(entry)
            else:
                rejected.append(entry)

        return {
            "stored": stored,
            "rejected": rejected,
            "escalated": escalated,
            "contradictions": [
                {
                    "text": c.text,
                    "conflicts_with": c.contradiction_with,
                }
                for c in report.contradictions
            ],
        }

    # ------------------------------------------------------------------
    # Evidence pack (argumentation support for Core)
    # ------------------------------------------------------------------

    def observe_and_store(self, text: str) -> dict[str, Any]:
        """
        Perceive, gate, and PERSIST the approved knowledge.

        This is the loop Core uses in conversation: everything the
        gate allows is embedded and indexed immediately; everything
        else is reported back with its reason.
        """
        report = self.extractor.extract(record_id=0, text=text)

        stored, rejected, escalated = [], [], []
        to_index: list[tuple[str, dict]] = []

        for candidate, decision in zip(
            report.candidates, report.decisions
        ):
            entry = {
                "text": candidate.text,
                "category": candidate.category,
                "tone": candidate.tone,
                "importance": candidate.importance,
                "emotion": candidate.emotion[:3],
                "sarcasm": candidate.sarcasm_probability,
                "person": (candidate.metadata or {}).get("person"),
                "fact": candidate.fact,
                "reason": decision.reason,
            }
            if decision.decision == "store":
                stored.append(entry)
                to_index.append((
                    candidate.text,
                    {
                        "category": candidate.category,
                        "tone": candidate.tone,
                        "importance": candidate.importance,
                        "emotion": [e[0] for e in candidate.emotion[:3]],
                        "person": (candidate.metadata or {}).get("person"),
                    },
                ))
            elif decision.decision == "escalate":
                escalated.append(entry)
            else:
                rejected.append(entry)

        indexed = 0
        if to_index:
            next_id = self.store.count() + 1
            chunks = self.chunker.chunk_record(
                next_id,
                " ".join(t for t, _ in to_index),
            )
            # rebuild metadata per chunk (chunker merges text only)
            for chunk, (_, meta) in zip(chunks, to_index):
                chunk.metadata.update(meta)
            if chunks:
                vectors = self.embedder.encode(
                    [c.text for c in chunks]
                )
                indexed = self.store.upsert(chunks, vectors)

        return {
            "stored": stored,
            "rejected": rejected,
            "escalated": escalated,
            "indexed_chunks": indexed,
            "contradictions": [
                {
                    "text": c.text,
                    "conflicts_with": c.contradiction_with,
                }
                for c in report.contradictions
            ],
        }

    def evidence_pack(
        self,
        claim: str,
        *,
        limit: int = 5,
        min_score: float = 0.35,
    ) -> dict[str, Any]:
        """
        Evidence for/against a claim, for grounded argumentation.

        Uses TRUE cosine similarity (VectorHit.score) so evidence
        strength is meaningful, not rank-based.
        Verdict: supported | contradicted | insufficient_evidence
        """
        from .extraction import _CONTRADICTION_CUES

        query_vector = self.embedder.encode_one(claim)
        hits = self.store.search(
            query_vector, limit=max(limit * 2, 8), min_score=min_score
        )
        if not hits:
            return {
                "claim": claim,
                "verdict": "insufficient_evidence",
                "confidence": 0.0,
                "supporting": [],
                "contradicting": [],
            }

        claim_lower = claim.lower()
        supporting, contradicting = [], []
        for hit in hits:
            entry = {
                "text": hit.text,
                "record_id": hit.record_id,
                "similarity": round(float(hit.score), 4),
                "importance": (hit.metadata or {}).get("importance"),
            }
            is_contra = False
            for a, b in _CONTRADICTION_CUES:
                if (a in claim_lower and b in hit.text.lower()) or (
                    b in claim_lower and a in hit.text.lower()
                ):
                    is_contra = True
                    break
            (contradicting if is_contra else supporting).append(entry)

        support_strength = sum(e["similarity"] for e in supporting)
        contra_strength = sum(e["similarity"] for e in contradicting)

        if contradicting and contra_strength > support_strength:
            verdict = "contradicted"
            confidence = contra_strength / (
                contra_strength + support_strength + 1e-6
            )
        elif supporting and contradicting:
            verdict = "conflict"
            confidence = support_strength / (
                support_strength + contra_strength + 1e-6
            )
        elif supporting:
            verdict = "supported"
            confidence = support_strength / (
                support_strength + contra_strength + 1e-6
            )
        else:
            verdict = "insufficient_evidence"
            confidence = 0.0

        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": round(min(1.0, confidence), 4),
            "supporting": supporting[:limit],
            "contradicting": contradicting[:limit],
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def delete_record(self, record_id: int) -> int:
        return self.store.delete_record(record_id)

    def stats(self) -> dict[str, Any]:
        return {
            "chunks": self.store.count(),
            "embedder": self.embedder.info(),
            "classifier_version": (
                self.classifier.version
                if self.classifier else "untrained"
            ),
        }

    def close(self) -> None:
        self.store.close()
