from __future__ import annotations

"""
VectorStore: persistent embedding storage + exact cosine search.

Storage model:
  - SQLite table `semantic_chunks` (uid, record_id, chunk_index, text,
    metadata JSON, timestamps) -- authoritative chunk registry
  - float32 vector blobs stored alongside in the same row
  - numpy exact cosine search (deterministic, no ANN approximation).

Scaling note: exact search over N chunks costs O(N*d); at 384 dims
this comfortably handles hundreds of thousands of chunks. An ANN
index (FAISS/HNSW) can be added later without changing the API.
"""

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .chunker import Chunk


@dataclass(frozen=True)
class VectorHit:
    uid: str
    record_id: int
    chunk_index: int
    text: str
    score: float
    metadata: dict[str, Any]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_chunks (
    uid         TEXT PRIMARY KEY,
    record_id   INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_semantic_record
    ON semantic_chunks (record_id);
CREATE INDEX IF NOT EXISTS idx_semantic_metadata
    ON semantic_chunks (record_id, chunk_index);
"""


class VectorStore:
    """
    Persistent exact-search vector index backed by SQLite + numpy.

    All vectors for a query are loaded once per search and scored in a
    single vectorized pass; this is both simple and fast at the scale
    a personal knowledge base reaches.
    """

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(
                _default_db_path()
            )
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            db_path, check_same_thread=False
        )
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
    ) -> int:
        """
        Insert or update chunks with their vectors.
        Returns the number of rows written.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                "chunks and vectors length mismatch: "
                f"{len(chunks)} vs {len(vectors)}"
            )

        now = time.time()
        rows = []
        for chunk, vector in zip(chunks, vectors):
            vec = np.asarray(vector, dtype=np.float32)
            rows.append(
                (
                    chunk.uid,
                    chunk.record_id,
                    chunk.chunk_index,
                    chunk.text,
                    json.dumps(chunk.metadata or {}, ensure_ascii=False),
                    int(vec.shape[0]),
                    vec.tobytes(),
                    now,
                    now,
                )
            )

        with self._lock, self._conn:
            self._conn.executemany(
                """
                INSERT INTO semantic_chunks
                    (uid, record_id, chunk_index, text, metadata,
                     dim, vector, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(uid) DO UPDATE SET
                    text        = excluded.text,
                    metadata    = excluded.metadata,
                    dim         = excluded.dim,
                    vector      = excluded.vector,
                    updated_at  = excluded.updated_at
                """,
                rows,
            )
        return len(rows)

    def delete_record(self, record_id: int) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM semantic_chunks WHERE record_id = ?",
                (int(record_id),),
            )
            return cur.rowcount

    # ------------------------------------------------------------------
    # Search path
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        *,
        limit: int = 10,
        min_score: float = 0.0,
        filter_fn: Any = None,
    ) -> list[VectorHit]:
        """
        Exact cosine search over all stored vectors.

        filter_fn: optional callable(uid, record_id, metadata_dict)
                   -> bool for post-filtering.
        """
        q = np.asarray(query_vector, dtype=np.float32).ravel()
        if q.size == 0:
            return []

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT uid, record_id, chunk_index, text, metadata,
                       vector
                FROM semantic_chunks
                """
            ).fetchall()

        if not rows:
            return []

        uids: list[str] = []
        record_ids: list[int] = []
        chunk_indexes: list[int] = []
        texts: list[str] = []
        metas: list[dict] = []
        vecs: list[np.ndarray] = []

        for uid, record_id, chunk_index, text, meta_json, blob in rows:
            meta = {}
            try:
                metadata = json.loads(meta_json or "{}")
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            if filter_fn is not None:
                try:
                    ok = filter_fn(uid, record_id, metadata)
                except Exception:
                    ok = False
                if not ok:
                    continue
            uids.append(uid)
            record_ids.append(int(record_id))
            chunk_indexes.append(int(chunk_index))
            texts.append(text)
            metas.append(metadata)
            vecs.append(
                np.frombuffer(blob, dtype=np.float32)
            )

        if not vecs:
            return []

        matrix = np.vstack(vecs)
        if matrix.shape[1] != q.size:
            raise ValueError(
                f"dimension mismatch: store has {matrix.shape[1]}, "
                f"query has {q.size}"
            )

        # bge vectors are L2-normalized -> dot product == cosine
        scores = matrix @ q

        order = np.argsort(-scores)[: max(limit * 3, limit)]
        hits: list[VectorHit] = []
        for i in order:
            score = float(scores[i])
            if score < min_score:
                continue
            hits.append(
                VectorHit(
                    uid=uids[i],
                    record_id=record_ids[i],
                    chunk_index=chunk_indexes[i],
                    text=texts[i],
                    score=score,
                    metadata=metas[i],
                )
            )
            if len(hits) >= limit:
                break

        return hits

    def count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM semantic_chunks"
        ).fetchone()
        return int(row[0]) if row else 0

    def iter_all(self) -> list[tuple[str, str, dict]]:
        """Return (uid, text, metadata) for every stored chunk."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT uid, text, metadata FROM semantic_chunks"
            ).fetchall()
        out = []
        for uid, text, meta_json in rows:
            out.append((uid, text, _loads(meta_json)))
        return out

    def get_text_and_metadata(self, uid: str) -> tuple[str, dict]:
        row = self._conn.execute(
            "SELECT text, metadata FROM semantic_chunks WHERE uid = ?",
            (uid,),
        ).fetchone()
        if row is None:
            return "", {}
        return row[0], _loads(row[1])

    def get_created_at_map(
        self, uids: list[str]
    ) -> dict[str, float]:
        """created_at epoch seconds per uid (missing uids omitted)."""
        out: dict[str, float] = {}
        for uid in uids:
            row = self._conn.execute(
                "SELECT created_at FROM semantic_chunks WHERE uid = ?",
                (uid,),
            ).fetchone()
            if row is not None:
                out[uid] = float(row[0])
        return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------


def _loads(meta_json: str | None) -> dict:
    try:
        return json.loads(meta_json or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def _default_db_path() -> "Path":
    from pathlib import Path

    return (
        Path(__file__).resolve().parent.parent.parent
        / "semantic_vectors.db"
    )
