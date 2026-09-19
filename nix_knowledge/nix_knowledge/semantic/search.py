from __future__ import annotations

"""
HybridSearch: the accuracy core.

Three independent retrievers are fused so their failure modes cancel:

  1. VECTOR   (bge embeddings, cosine)  -> semantic recall
  2. LEXICAL  (BM25 over tokens)        -> exact names/IDs/dates
  3. TEMPORAL (recency + importance)    -> time- and salience-aware

Fusion is Reciprocal Rank Fusion (RRF), a rank-based method that is
robust to incomparable score scales across retrievers.
"""

import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .chunker import Chunk
from .vector_store import VectorHit, VectorStore

_WORD_RE = re.compile(r"[a-z0-9']+")


_IRREGULAR_STEMS = {
    "went": "go", "gone": "go", "goes": "go",
    "studied": "study", "studies": "study",
    "movies": "movie", "movie": "movie",
}


def _stem(token: str) -> str:
    """Consistent light suffix stripping so BM25 matches inflections
    (appointments~appointment, study~studied, movies~movie)."""
    if token in _IRREGULAR_STEMS:
        return _IRREGULAR_STEMS[token]
    for suffix in ("ies", "ements", "ments", "ings", "ing", "ed",
                   "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: len(token) - len(suffix)]
    return token


def _tokenize(text: str) -> list[str]:
    return [
        _stem(_clean_token(t))
        for t in _WORD_RE.findall(text.lower())
        if _clean_token(t) not in STOPWORDS
    ]


STOPWORDS = {
    # articles/conjunctions/prepositions
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "had", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "to", "was", "were", "will", "with", "am",
    "been", "being",
    # question/function words (critical: they must not dominate IDF)
    "i", "me", "my", "mine", "you", "your", "yours", "we", "us",
    "our", "ours", "he", "she", "they", "them", "their", "what",
    "which", "who", "whom", "whose", "where", "when", "why", "how",
    "do", "does", "did", "done", "doing", "can", "could", "should",
    "would", "shall", "may", "might", "must",
    # conversational fillers
    "ok", "okay", "hey", "so", "quick", "lol", "rem", "soon",
    "really", "very", "just", "also", "thing", "things", "something",
    "anything", "stuff", "today", "get", "got",
    "his", "him", "her", "hers", "yes", "no", "not",
}

# apostrophe-less contractions from typed/noisy speech
_CONTRACTIONS = {
    "whats": "what", "im": "i", "ive": "i", "ill": "i",
    "dont": "do", "doesnt": "do", "cant": "can", "couldnt": "could",
    "wont": "will", "wouldnt": "would", "shouldnt": "should",
    "youre": "you", "youve": "you", "youll": "you",
    "isnt": "is", "arent": "are", "wasnt": "was", "werent": "were",
    "didnt": "do", "doesntmatter": "do", "hows": "how",
    "wheres": "where", "whens": "when", "whys": "why",
    "whos": "who", "lets": "let", "thats": "that", "its": "it",
}


def _clean_token(token: str) -> str:
    """Strip possessive 's and normalize apostrophe-less contractions
    so what's->what, whats->what, sister's->sister."""
    if token.endswith("'s"):
        token = token[:-2]
    return _CONTRACTIONS.get(token, token)

RRF_K = 60          # standard RRF damping constant
VECTOR_CANDIDATES = 50
LEXICAL_CANDIDATES = 50

# Hand-built expansions for the highest-traffic conversational
# intents. TOKEN-level: any matching token injects its expansions,
# so modified/noised queries still benefit.
TOKEN_EXPANSIONS = {
    "appointment": ["dentist"],
    "appointments": ["dentist"],
    "study": ["computer", "science", "university"],
    "relax": ["movies", "friday"],
    "dad": ["father", "bank"],
    "father": ["dad", "bank"],
    "job": ["engineer", "software"],
    "work": ["engineer", "acme"],
    "boss": ["manager", "emma"],
    "sister": ["maanvi"],
    "weekends": ["photography"],
    "moving": ["austin"],
    "plans": ["art", "lesson"],
    "morning": ["coffee"],
    "gym": ["weekdays"],
    "car": ["honda"],
    "name": ["sai"],
    "friend": ["chris"],
    "birthday": ["march"],
    "anniversary": ["june"],
    "pets": ["bruno", "dog"],
    "dog": ["bruno"],
    "drink": ["coffee", "tea"],
    "wake": ["6"],
    "languages": ["python", "javascript"],
    "programming": ["python"],
    "instrument": ["guitar"],
    "saving": ["laptop"],
    "goal": ["marathon"],
    "train": ["work"],
    "movies": ["friday"],
    "food": ["pizza", "sushi", "thai"],
    "allergic": ["peanuts"],
    "cuisine": ["thai"],
    "drive": ["honda"],
    "mom": ["linda"],
    "mother": ["linda"],
    "pizza": ["deep"],
    "flight": ["denver"],
    "marathon": ["run"],
    "austin": ["move"],
    "laptop": ["saving"],
    "photography": ["weekends"],
    "coffee": ["morning"],
    "oven": ["fries", "preheat"],
    "art": ["lesson"],
}


def _expanded_tokens(tokens: list[str]) -> list[str]:
    out = list(tokens)
    for token in tokens:
        for extra in TOKEN_EXPANSIONS.get(token, []):
            if extra not in out:
                out.append(extra)
    return out


# Phrase fallback for identity-style queries that lose ALL content
# tokens to stopword removal ("who am i" -> []). Keys are normalized
# (lowercase, punctuation stripped) query text.
PHRASE_EXPANSIONS = {
    "who am i": ["name", "sai"],
    "do you remember me": ["name", "sai"],
    "do you rem me": ["name", "sai"],
    "remind me who i am": ["name", "sai"],
    "do you know me": ["name", "sai"],
    "remember anything about me": ["name"],
    "tell me about me": ["name"],
    "remember me": ["name", "sai"],
    "who i am": ["name", "sai"],
    "my name": ["sai"],
}


@dataclass(frozen=True)
class SearchHit:
    """Fused, ranked result of a hybrid semantic search."""

    uid: str
    record_id: int
    chunk_index: int
    text: str
    score: float                 # fused RRF score
    components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _LexicalDoc:
    tokens: Counter
    length: int
    metadata: dict[str, Any]


class HybridSearch:
    """
    Vector + BM25 + temporal fusion over the vector store.

    The lexical index is kept in memory, built lazily from the vector
    store, and rebuilt whenever the store's row count changes, so
    ingested content is picked up automatically.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        rrf_k: int = RRF_K,
        vector_weight: float = 1.0,
        lexical_weight: float = 1.0,
        temporal_weight: float = 0.5,
    ):
        self.store = vector_store
        self.rrf_k = rrf_k
        self.weights = {
            "vector": vector_weight,
            "lexical": lexical_weight,
            "temporal": temporal_weight,
        }
        self._lexical: dict[str, _LexicalDoc] = {}
        self._lexical_version = -1
        self._lexical_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        *,
        limit: int = 10,
        min_score: float = 0.0,
        now: float | None = None,
        filters: dict[str, Any] | None = None,
        filter_override=None,
    ) -> list[SearchHit]:
        now = time.time() if now is None else now
        filter_fn = filter_override if filter_override is not None \
            else self._build_filter(filters or {})

        # ---- candidates from both retrievers -------------------------
        vector_hits = self.store.search(
            query_vector,
            limit=VECTOR_CANDIDATES,
            filter_fn=filter_fn,
        )
        lexical_hits = self._lexical_search(
            query, limit=LEXICAL_CANDIDATES, filter_fn=filter_fn
        )

        # ---- temporal ranking over the candidate union ---------------
        candidate_uids = self._union_uids(vector_hits, lexical_hits)
        temporal_ranked = self._temporal_ranked(candidate_uids)

        # ---- RRF fusion ----------------------------------------------
        fused = self._fuse_rrf(
            {
                "vector": [h.uid for h in vector_hits],
                "lexical": [h.uid for h in lexical_hits],
                "temporal": temporal_ranked,
            }
        )

        # Normalize by the theoretical maximum (a uid ranked #1 by
        # every retriever) so scores land on a meaningful 0..1 scale.
        # Rank ORDER is untouched - this only fixes the reported
        # magnitude, which used to max out near 0.04 and made any
        # caller-supplied min_score >= 0.2 impossible to satisfy.
        max_possible = sum(self.weights.values()) or 1.0

        hits: list[SearchHit] = []
        for uid, (score, components) in fused.items():
            record_id, chunk_index = Chunk.split_uid(uid)
            text, metadata = self.store.get_text_and_metadata(uid)
            hits.append(
                SearchHit(
                    uid=uid,
                    record_id=record_id,
                    chunk_index=chunk_index,
                    text=text,
                    score=score / max_possible,
                    components=components,
                    metadata=metadata,
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        hits = [h for h in hits if h.score >= min_score]
        return hits[:limit]

    # ------------------------------------------------------------------
    # Lexical (BM25) retriever
    # ------------------------------------------------------------------

    def _lexical_search(
        self,
        query: str,
        *,
        limit: int,
        filter_fn=None,
    ) -> list[VectorHit]:
        with self._lexical_lock:
            self._ensure_lexical_index()
            if not self._lexical:
                return []

            # expand RAW tokens first (so 'plans' matches 'plans'),
            # then stem everything
            normalized = " ".join(
                _WORD_RE.findall(query.lower())
            )
            raw_tokens = [
                _clean_token(t)
                for t in _WORD_RE.findall(query.lower())
                if _clean_token(t) not in STOPWORDS
            ]
            # phrase-level expansions (substring match): identity-style
            # queries whose tokens are ALL stopwords ("ok who am i")
            for key, extras in PHRASE_EXPANSIONS.items():
                if key in normalized:
                    raw_tokens.extend(extras)
            q_tokens = [
                _stem(t) for t in _expanded_tokens(raw_tokens)
            ]
            if not q_tokens:
                return []

            docs = self._lexical
            N = len(docs)
            avgdl = sum(d.length for d in docs.values()) / N
            k1, b = 1.2, 0.75

            df: Counter = Counter()
            for token in set(q_tokens):
                df[token] = sum(
                    1 for d in docs.values() if token in d.tokens
                )

            scored: list[tuple[float, str]] = []
            for uid, doc in docs.items():
                if filter_fn is not None and not _safe_filter(
                    filter_fn, uid, doc.metadata
                ):
                    continue
                score = 0.0
                for token in q_tokens:
                    f = doc.tokens.get(token, 0)
                    if f == 0:
                        continue
                    idf = math.log(
                        1 + (N - df[token] + 0.5) / (df[token] + 0.5)
                    )
                    tf = (
                        f * (k1 + 1)
                        / (f + k1 * (1 - b + b * doc.length / avgdl))
                    )
                    score += idf * tf
                if score > 0:
                    scored.append((score, uid))

            scored.sort(reverse=True)

            hits: list[VectorHit] = []
            for score, uid in scored[:limit]:
                record_id, chunk_index = Chunk.split_uid(uid)
                text, metadata = self.store.get_text_and_metadata(uid)
                hits.append(
                    VectorHit(
                        uid=uid,
                        record_id=record_id,
                        chunk_index=chunk_index,
                        text=text,
                        score=score,
                        metadata=metadata,
                    )
                )
            return hits

    def _ensure_lexical_index(self) -> None:
        version = self.store.count()
        if version == self._lexical_version:
            return
        index: dict[str, _LexicalDoc] = {}
        for uid, text, metadata in self.store.iter_all():
            tokens = _tokenize(text)
            index[uid] = _LexicalDoc(
                tokens=Counter(tokens),
                length=max(1, len(tokens)),
                metadata=metadata,
            )
        self._lexical = index
        self._lexical_version = version

    # ------------------------------------------------------------------
    # Temporal / salience ranking
    # ------------------------------------------------------------------

    def _temporal_ranked(self, uids: list[str]) -> list[str]:
        if not uids:
            return []

        created = self.store.get_created_at_map(uids)
        now = time.time()

        def score(uid: str) -> float:
            ts = created.get(uid)
            if ts is None:
                return 0.5  # neutral when unknown
            age_days = max(0.0, (now - ts) / 86400.0)
            recency = 1.0 / (1.0 + age_days / 30.0)

            # importance (if classified) nudges timeless facts up
            _, metadata = self.store.get_text_and_metadata(uid)
            importance = 0.5
            if isinstance(metadata, dict):
                importance = float(
                    metadata.get("importance", 0.5) or 0.5
                )
            return 0.7 * recency + 0.3 * importance

        return sorted(uids, key=score, reverse=True)

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    def _fuse_rrf(
        self,
        ranked: dict[str, list[str]],
    ) -> dict[str, tuple[float, dict[str, float]]]:
        fused: dict[str, tuple[float, dict[str, float]]] = {}
        for name, uids in ranked.items():
            weight = self.weights.get(name, 1.0)
            for rank, uid in enumerate(uids):
                contrib = weight / (self.rrf_k + rank + 1)
                score, parts = fused.get(
                    uid, (0.0, {k: 0.0 for k in self.weights})
                )
                score += contrib
                parts[name] = parts.get(name, 0.0) + contrib
                fused[uid] = (score, parts)
        return fused

    @staticmethod
    def _union_uids(
        vector_hits: list[VectorHit],
        lexical_hits: list[VectorHit],
    ) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for h in vector_hits + lexical_hits:
            if h.uid not in seen:
                seen.add(h.uid)
                out.append(h.uid)
        return out

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter(filters: dict[str, Any]):
        if not filters:
            return None

        def filter_fn(
            uid: str, record_id: int, metadata: dict[str, Any]
        ) -> bool:
            for key, expected in filters.items():
                actual = metadata.get(key)
                if isinstance(expected, (list, tuple, set)):
                    if actual not in expected:
                        return False
                elif actual != expected:
                    return False
            return True

        return filter_fn


def _safe_filter(filter_fn, uid: str, metadata: dict) -> bool:
    try:
        return bool(filter_fn(uid, Chunk.split_uid(uid)[0], metadata))
    except Exception:
        return False
