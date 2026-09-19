from __future__ import annotations

"""
Chunker: turns raw knowledge text into retrieval-ready chunks.

bge-small-en-v1.5 has a 512-token window, but semantic accuracy is
highest when a chunk is one self-contained statement. We therefore
split on sentence boundaries first and only merge short neighbors.

Every chunk carries provenance: which knowledge record it came from,
its position, and (optionally) a pre-computed classification payload
attached at ingestion time.
"""

import re
from dataclasses import dataclass, field
from typing import Any

_SENTENCE_SPLIT = re.compile(
    r"""
    (?<=[.!?])          # after sentence-ending punctuation
    \s+                 # whitespace
    (?=[A-Z0-9"'(])     # a plausible sentence start
    """,
    re.VERBOSE,
)

_WORD_RE = re.compile(r"\S+")

MIN_WORDS = 4
MAX_WORDS = 96
OVERLAP_WORDS = 12


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit derived from a knowledge record."""

    record_id: int
    chunk_index: int
    text: str
    word_count: int

    # Optional semantic payload attached after classification.
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        """Stable unique id used by the vector store."""
        return f"{self.record_id}:{self.chunk_index}"

    @staticmethod
    def split_uid(uid: str) -> tuple[int, int]:
        record_id_s, chunk_index_s = uid.split(":", 1)
        return int(record_id_s), int(chunk_index_s)


class Chunker:
    """
    Sentence-aware chunking with light merging and bounded overlap.

    Design goals:
      - one chunk ~= one statement (best embedding granularity)
      - never lose information between chunks (overlap on long runs)
      - deterministic output for the same input
    """

    def __init__(
        self,
        *,
        min_words: int = MIN_WORDS,
        max_words: int = MAX_WORDS,
        overlap_words: int = OVERLAP_WORDS,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.overlap_words = overlap_words

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_record(
        self,
        record_id: int,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        text = (text or "").strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        groups = self._group_sentences(sentences)

        chunks: list[Chunk] = []
        for i, group in enumerate(groups):
            joined = " ".join(group)
            chunks.append(
                Chunk(
                    record_id=record_id,
                    chunk_index=i,
                    text=joined,
                    word_count=len(_WORD_RE.findall(joined)),
                    metadata=dict(metadata or {}),
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = _SENTENCE_SPLIT.split(text)
        return [p.strip() for p in parts if p and p.strip()]

    def _group_sentences(
        self,
        sentences: list[str],
    ) -> list[list[str]]:
        """
        Greedy grouping:

          - a sentence is emitted alone if it already has >= min_words
          - consecutive short sentences are merged until min_words
          - a merged group never exceeds max_words; overflow starts a
            new group with a small lexical overlap so context is kept
        """
        groups: list[list[str]] = []
        current: list[str] = []
        current_words = 0

        for sentence in sentences:
            n = len(_WORD_RE.findall(sentence))

            if n > self.max_words:
                # Very long sentence: hard-split on words with overlap.
                if current:
                    groups.append(current)
                    current, current_words = [], 0
                for piece in self._hard_split(sentence):
                    groups.append([piece])
                continue

            if current and current_words + n > self.max_words:
                groups.append(current)
                tail = " ".join(current)[-1:]
                # keep a tiny overlap tail (last overlap_words words)
                tail_words = _WORD_RE.findall(" ".join(current))
                overlap = tail_words[-self.overlap_words:]
                current = [" ".join(overlap)] if overlap else []
                current_words = len(overlap)

            current.append(sentence)
            current_words += n

            if current_words >= self.min_words:
                groups.append(current)
                current, current_words = [], 0

        if current:
            if groups and current_words < self.min_words:
                # fold a tiny trailing fragment into the previous group
                groups[-1].extend(current)
            else:
                groups.append(current)

        # normalize: strip single-string groups made only of overlap
        return [g for g in groups if " ".join(g).strip()]

    def _hard_split(self, sentence: str) -> list[str]:
        words = _WORD_RE.findall(sentence)
        pieces: list[str] = []
        step = self.max_words - self.overlap_words
        for start in range(0, len(words), step):
            piece = words[start : start + self.max_words]
            if piece:
                pieces.append(" ".join(piece))
            if start + self.max_words >= len(words):
                break
        return pieces
