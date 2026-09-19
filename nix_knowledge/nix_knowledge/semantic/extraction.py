from __future__ import annotations

"""
KnowledgeExtractor: text -> structured knowledge candidates.

Given raw text (conversation turns, notes, documents), the extractor:
  1. classifies every chunk (emotion/tone/importance/category)
  2. extracts simple first-person facts via patterns
  3. deduplicates against the vector store (cosine >= 0.92)
  4. flags potential contradictions (high overlap + opposing cues)
  5. emits KnowledgeCandidate objects ready for KnowledgeEngine.create
"""

import re
from dataclasses import dataclass, field
from typing import Any

from .chunker import Chunk, Chunker
from .classifier import SemanticClassifier

# cosine above this = near-duplicate
DUP_THRESHOLD = 0.92
# cosine above this + opposing cue = possible contradiction
# Candidate-pool gate for contradiction search. Measured: negation
# flips sit at LOW cosine ("i love hiking" vs "i hate hiking" = 0.556)
# because embeddings largely ignore polarity - so 0.75 never even
# surfaced the candidates. 0.45 admits same-topic pairs; the cue check
# below (requires an actual opposition pair across the two texts) is
# what decides, so this stays cheap and false positives stay rare.
CONTRADICTION_OVERLAP = 0.45
# candidate passed both gates -- safe to store
# (kept here so gate + service agree on one constant)
DUP_THRESHOLD_GATE = DUP_THRESHOLD

_CONTRADICTION_CUES = [
    ("no longer", "still"), ("anymore", "still"),
    ("stopped", "started"), ("quit", "began"),
    ("hate", "love"), ("never", "always"), ("moved out", "moved in"),
    ("divorced", "married"), ("sold", "bought"), ("left", "joined"),
    ("not", "still"), ("no longer", "now"),
    # first-person preference/status flips (measured miss:
    # love->hate was never flagged)
    ("dislike", "like"), ("dislike", "love"),
    ("healthy", "sick"), ("better", "worse"),
    ("broke up", "dating"), ("unemployed", "hired"),
    ("adopted", "gave away"),
]

_FACT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("name", re.compile(
        r"\bmy name is (\w+)", re.I)),
    ("lives_in", re.compile(
        r"\bi (?:live|am based) in ([a-z\s]+)", re.I)),
    ("works_as", re.compile(
        r"\bi work as an? ([a-z\s]+)", re.I)),
    ("works_at", re.compile(
        r"\bi work at ([\w\s&\.]+)", re.I)),
    ("studies_at", re.compile(
        r"\bi (?:study|go to|attend) ([\w\s&\.]+)", re.I)),
    ("likes", re.compile(
        r"\bi (?:really )?(?:love|like|enjoy) ([^.!?]+)", re.I)),
    ("dislikes", re.compile(
        r"\bi (?:really )?(?:hate|dislike|can't stand|cannot stand) "
        r"([^.!?]+)", re.I)),
    ("allergic_to", re.compile(
        r"\bi(?:'m| am) allergic to ([a-z\s]+)", re.I)),
    ("owns", re.compile(
        r"\bi (?:own|have) an? ([a-z\s]+)", re.I)),
]

# facts that are time-sensitive and should expire rather than dedup
_VOLATILE_FACTS = {"works_at", "works_as", "lives_in", "owns"}

_PERSON_MARKER = re.compile(
    r"\b(?:my|i)\b", re.I
)
_NAME_EXTRACTION = re.compile(
    r"\b(?:my name is|i am|i'm|call me)\s+([A-Z][a-z]+)\b"
)


def extract_person(text: str) -> str | None:
    """Best-effort speaker attribution from first-person text."""
    match = _NAME_EXTRACTION.search(text)
    if match:
        return match.group(1)
    if _PERSON_MARKER.search(text):
        return "user"
    return None


# Lexical irony/sarcasm cues: a strong cue raises suspicion regardless
# of the neural score (belt-and-suspenders for the storage gate).
_IRONY_CUES = re.compile(
    r"\b(yeah right|just what i needed|oh great|oh wonderful|"
    r"sure thing|as if|totally|perfect,? just perfect|"
    r"could(n't| not) be happier|best day ever|living the dream)\b",
    re.I,
)

# Non-knowledge idioms and filler the neural heads misread as facts
# (e.g. "idk what you mean" matches no first-person marker but carries
# 'i am/you mean' shapes; "that reminds me of something" trips the
# preference patterns). Caught BEFORE classification so they never
# reach the gate as would-be knowledge.
_NON_KNOWLEDGE_IDIOMS = re.compile(
    r"\b(idk|i dunno|i don't know|what you mean|what do you mean|"
    r"reminds me of something|not sure really|who knows|wait what|"
    r"huh interesting|hmm+|whatever works|fine fine|ok sure|haha yeah)\b",
    re.I,
)


@dataclass
class KnowledgeCandidate:
    """One extracted, classified, deduplicated knowledge unit."""

    text: str
    record_id: int | None
    chunk_index: int
    emotion: list[tuple[str, float]]
    tone: tuple[str, float]
    importance: float
    category: str
    category_confidence: float
    confidence: float
    sarcasm_flagged: bool = False
    sarcasm_probability: float = 0.0
    fact: dict[str, Any] | None = None
    duplicate_of: str | None = None
    contradiction_with: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ExtractionReport:
    """Aggregates extraction results for a batch of texts."""

    candidates: list[KnowledgeCandidate] = field(default_factory=list)
    duplicates: list[KnowledgeCandidate] = field(default_factory=list)
    contradictions: list[KnowledgeCandidate] = field(
        default_factory=list
    )
    decisions: list = field(default_factory=list)

    @property
    def novel(self) -> list[KnowledgeCandidate]:
        return [
            c for c in self.candidates
            if c.duplicate_of is None
            and c.contradiction_with is None
        ]


class KnowledgeExtractor:
    def __init__(
        self,
        classifier: SemanticClassifier | None = None,
        store=None,
        gate=None,
    ):
        self.classifier = classifier or SemanticClassifier()
        self.store = store  # optional VectorStore for dedup
        self.gate = gate    # optional StorageGate for policy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, record_id: int, text: str) -> ExtractionReport:
        """
        Classify + fact-extract + dedup + contradiction-check.

        When a StorageGate is attached, candidates also carry a
        GateDecision in ``report.decisions`` in the same order as
        ``report.candidates``.
        """
        report = ExtractionReport()
        chunks = Chunker().chunk_record(record_id, text)
        if not chunks:
            return report

        texts = [c.text for c in chunks]
        results = []
        for t in texts:
            if _NON_KNOWLEDGE_IDIOMS.search(t):
                results.append({
                    "emotion": [("confusion", 0.6)],
                    "tone": ("neutral", 0.5),
                    "importance": 0.05,
                    "category": ("general", 0.9),
                    "sarcasm": (False, 0.0),
                    "literal": True,
                    "storage_eligible": False,
                    "confidence": 0.2,
                })
            else:
                results.append(self.classifier.classify(t))

        vectors = None
        if self.store is not None:
            vectors = self.classifier.embedder.encode(texts)

        for i, (chunk, result) in enumerate(zip(chunks, results)):
            vector = vectors[i] if vectors is not None else None
            similarity = None
            if vector is not None and self.store is not None:
                nearest = self.store.search(vector, limit=1)
                if nearest:
                    similarity = nearest[0].score

            candidate = self._build_candidate(
                chunk, result, vector
            )
            # layered sarcasm decision: neural score OR explicit cue
            if candidate.sarcasm_probability < 0.8 and _IRONY_CUES.search(
                chunk.text
            ):
                candidate.sarcasm_probability = max(
                    0.8, candidate.sarcasm_probability
                )
                candidate.sarcasm_flagged = True
            report.candidates.append(candidate)
            if candidate.duplicate_of:
                report.duplicates.append(candidate)
            elif candidate.contradiction_with:
                report.contradictions.append(candidate)

            if self.gate is not None:
                report.decisions.append(
                    self.gate.evaluate(
                        candidate, similarity_score=similarity
                    )
                )

        return report

    def dedupe_only(self, text: str, vector) -> tuple[bool, str]:
        """Return (is_duplicate, uid_of_duplicate)."""
        if self.store is None:
            return False, ""
        hits = self.store.search(vector, limit=1)
        if hits and hits[0].score >= DUP_THRESHOLD:
            return True, hits[0].uid
        return False, ""

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_candidate(self, chunk, result, vector):
        fact = self._extract_fact(chunk.text)
        duplicate_of = None
        contradiction_with = None

        if vector is not None and self.store is not None:
            is_dup, uid = self.dedupe_only(chunk.text, vector)
            if is_dup:
                duplicate_of = uid
            else:
                contradiction_with = self._find_contradiction(
                    chunk.text, vector
                )

        emotion_pairs = [
            (label, round(p, 4)) for label, p in result["emotion"]
        ]
        tone, tone_conf = result["tone"]
        category, category_conf = result["category"]
        sarcasm_flagged, sarcasm_prob = result.get(
            "sarcasm", (False, 0.0)
        )
        # normalize: gate policy operates on the calibrated threshold
        sarcasm_flagged = bool(
            sarcasm_prob >= self.classifier.thresholds.get(
                "sarcasm_gate", 0.8
            )
        )

        return KnowledgeCandidate(
            text=chunk.text,
            record_id=chunk.record_id,
            chunk_index=chunk.chunk_index,
            emotion=emotion_pairs,
            tone=tone,
            importance=float(result["importance"]),
            category=category,
            category_confidence=category_conf,
            confidence=result["confidence"],
            sarcasm_flagged=sarcasm_flagged,
            sarcasm_probability=sarcasm_prob,
            fact=fact,
            duplicate_of=duplicate_of,
            contradiction_with=contradiction_with,
            metadata={
                "sarcasm_probability": sarcasm_prob,
                "literal": not sarcasm_flagged,
                "person": extract_person(chunk.text),
            },
        )

    def _find_contradiction(
        self, text: str, vector
    ) -> str | None:
        if self.store is None:
            return None
        hits = self.store.search(
            vector, limit=5, min_score=CONTRADICTION_OVERLAP
        )
        lower = text.lower()
        for hit in hits:
            other = hit.text.lower()
            for a, b in _CONTRADICTION_CUES:
                if (a in lower and b in other) or (
                    b in lower and a in other
                ):
                    return hit.uid
        return None

    @staticmethod
    def _extract_fact(text: str) -> dict | None:
        for name, pattern in _FACT_PATTERNS:
            match = pattern.search(text)
            if match:
                value = match.group(1).strip(" .")
                return {"type": name.strip(), "value": value}
        return None
