"""Tests for the neural semantic layer (chunker, store, gate, service)."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
pytest.importorskip("sentence_transformers")

from nix_knowledge.semantic.chunker import Chunker
from nix_knowledge.semantic.gate import StorageGate
from nix_knowledge.semantic.vector_store import VectorStore
from nix_knowledge.semantic.service import SemanticService
from nix_knowledge.semantic.classifier import SemanticClassifier


@pytest.fixture(scope="module")
def service():
    db = os.path.join(tempfile.mkdtemp(), "semantic_test.db")
    svc = SemanticService(db_path=db)
    yield svc
    svc.close()


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


def test_chunker_sentence_split_and_provenance():
    chunks = Chunker().chunk_record(
        7, "I love coffee. I hate tea! Do you like soda?"
    )
    assert all(c.record_id == 7 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.text for c in chunks)


def test_chunker_empty_input():
    assert Chunker().chunk_record(1, "   ") == []


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


def test_vector_store_roundtrip():
    db = os.path.join(tempfile.mkdtemp(), "store_test.db")
    store = VectorStore(db)
    # 3+ word sentences stay separate chunks (min_words = 4)
    chunks = Chunker().chunk_record(
        1,
        "Alpha sentence traveling happily. Beta sentence wandering briskly. "
        "Gamma sentence drifting lazily.",
    )
    assert len(chunks) >= 2, "expected sentence-level chunks"
    vectors = [[1.0] * 8 for _ in chunks]
    store.upsert(chunks, vectors)
    assert store.count() == len(chunks)
    hits = store.search([1.0] * 8, limit=2)
    assert len(hits) == 2
    assert hits[0].record_id == 1
    store.close()


def test_vector_store_upsert_is_idempotent():
    db = os.path.join(tempfile.mkdtemp(), "store_idem.db")
    store = VectorStore(db)
    chunks = Chunker().chunk_record(3, "Only one statement.")
    vectors = [[0.5] * 8]
    store.upsert(chunks, vectors)
    store.upsert(chunks, vectors)
    assert store.count() == 1
    store.close()


def test_vector_store_delete_record():
    db = os.path.join(tempfile.mkdtemp(), "store_del.db")
    store = VectorStore(db)
    chunks = Chunker().chunk_record(9, "One. Two. Three. Four.")
    store.upsert(chunks, [[1.0] * 8] * len(chunks))
    assert store.count() == len(chunks)
    assert store.delete_record(9) == len(chunks)
    assert store.count() == 0
    store.close()


# ---------------------------------------------------------------------------
# Classifier (requires trained artifact)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def classifier():
    clf = SemanticClassifier()
    clf.load()
    return clf


def test_classifier_loads_trained_heads(classifier):
    assert classifier.net is not None
    assert classifier.version.startswith("supervised")


def test_classifier_output_contract(classifier):
    result = classifier.classify("I have a dentist appointment tomorrow at 3")
    assert isinstance(result["emotion"], list)
    assert result["tone"][0] in ("formal", "casual", "urgent", "neutral")
    assert 0.0 <= result["importance"] <= 1.0
    assert isinstance(result["sarcasm"], tuple)
    assert result["category"][0] in (
        "personal_fact", "preference", "event", "relationship",
        "skill", "opinion", "goal", "general",
    )


def test_classifier_events_score_high_importance(classifier):
    result = classifier.classify("My wedding is on June 3rd")
    assert result["category"][0] == "event"
    assert result["importance"] >= 0.6


# ---------------------------------------------------------------------------
# Storage gate policy
# ---------------------------------------------------------------------------


def _candidate(**overrides):
    from nix_knowledge.semantic.extraction import KnowledgeCandidate

    defaults = dict(
        text="I live in Chicago",
        record_id=0,
        chunk_index=0,
        emotion=[("neutral", 0.5)],
        tone=("neutral", 0.8),
        importance=0.8,
        category="personal_fact",
        category_confidence=0.9,
        confidence=0.8,
    )
    defaults.update(overrides)
    return KnowledgeCandidate(**defaults)


def test_gate_rejects_duplicates():
    decision = StorageGate().evaluate(
        _candidate(), similarity_score=0.95
    )
    assert decision.decision == "reject"
    assert decision.reason == "duplicate"


def test_gate_rejects_figurative():
    decision = StorageGate().evaluate(
        _candidate(sarcasm_flagged=True, sarcasm_probability=0.9)
    )
    assert decision.decision == "reject"
    assert decision.reason == "figurative_language"


def test_gate_rejects_low_importance():
    decision = StorageGate().evaluate(
        _candidate(importance=0.2, fact=None)
    )
    assert decision.decision == "reject"
    assert decision.reason == "low_importance"


def test_gate_stores_normal_fact():
    decision = StorageGate().evaluate(_candidate())
    assert decision.decision == "store"


def test_gate_escalates_uncertain_key_fact():
    decision = StorageGate().evaluate(
        _candidate(confidence=0.2, fact={"type": "name", "value": "Bo"})
    )
    assert decision.decision == "escalate"


# ---------------------------------------------------------------------------
# End-to-end service behavior
# ---------------------------------------------------------------------------


def test_observe_stores_facts_and_blocks_sarcasm(service):
    result = service.observe_and_store(
        "My name is Alex and I live in Chicago. I love deep dish pizza."
    )
    assert result["indexed_chunks"] >= 2
    assert all(
        s["category"] in ("personal_fact", "preference")
        for s in result["stored"]
    )

    sarc = service.observe_and_store(
        "Oh great, another Monday, just what I needed"
    )
    assert sarc["indexed_chunks"] == 0
    assert any(
        r["reason"] == "figurative_language" for r in sarc["rejected"]
    )


def test_exact_duplicate_is_rejected(service):
    first = service.observe_and_store(
        "I work as an engineer at Acme Corp"
    )
    assert first["indexed_chunks"] >= 1

    again = service.observe_and_store(
        "I work as an engineer at Acme Corp"
    )
    assert again["indexed_chunks"] == 0
    assert any(
        r["reason"] == "duplicate" for r in again["rejected"]
    )


def test_hybrid_search_finds_stored_fact(service):
    hits = service.search("Where does the user live?", limit=3)
    assert hits, "expected at least one hit for stored location fact"
    assert any("live" in h.text.lower() for h in hits)


def test_evidence_pack_supported_verdict(service):
    pack = service.evidence_pack("Where does the user live?")
    assert pack["verdict"] in ("supported", "conflict")
    assert pack["supporting"], "expected supporting evidence"


def test_evidence_pack_detects_location_change_conflict(service):
    service.observe_and_store(
        "I now live in Denver, not Chicago anymore"
    )
    pack = service.evidence_pack("Does the user still live in Chicago?")
    assert pack["verdict"] in ("conflict", "contradicted")
    assert pack["contradicting"], "expected contradicting evidence"


def test_low_importance_chatter_not_stored(service):
    before = service.store.count()
    result = service.observe_and_store(
        "The weather is kind of nice today I guess"
    )
    after = service.store.count()
    assert after == before
    assert result["indexed_chunks"] == 0


# ---------------------------------------------------------------------------
# Defect fixes verified 2026-09-12 (gate questions, score scale, cues)
# ---------------------------------------------------------------------------


def test_gate_rejects_wh_question(service):
    before = service.store.count()
    result = service.observe_and_store(
        "what is my sister's favorite food?"
    )
    assert service.store.count() == before
    reasons = [e["reason"] for e in result["rejected"]]
    assert "question_not_knowledge" in reasons


def test_gate_rejects_aux_inversion_question(service):
    before = service.store.count()
    result = service.observe_and_store("do you remember my birthday")
    assert service.store.count() == before
    reasons = [e["reason"] for e in result["rejected"]]
    assert "question_not_knowledge" in reasons


def test_gate_rejects_trailing_question_mark(service):
    before = service.store.count()
    result = service.observe_and_store("my sister is coming over?")
    assert service.store.count() == before
    reasons = [e["reason"] for e in result["rejected"]]
    assert "question_not_knowledge" in reasons


def test_gate_allows_aux_with_content_word(service):
    # "can speak spanish" (aux + CONTENT, post-strip fragment shape)
    # must NOT be rejected by the question rule. It may still be
    # rejected for other reasons (or stored) - just not as a question.
    result = service.observe_and_store("can speak spanish fluently")
    reasons = [e["reason"] for e in result["rejected"]]
    assert "question_not_knowledge" not in reasons


def test_hybrid_scores_on_unit_scale(service):
    hits = service.search("user's sister", limit=5)
    assert hits, "expected candidates for the query"
    # scores are normalized to 0..1 (theoretical max = unanimous #1);
    # pre-fix they maxed near 0.04 with no interpretable ceiling
    assert all(0.0 < h.score <= 1.0 for h in hits)
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


def test_contradiction_flip_flagged(service):
    service.observe_and_store("i love running at sunrise")
    result = service.observe_and_store(
        "i hate running now, i quit completely"
    )
    assert result["contradictions"], "love->hate flip must be flagged"


def test_contradiction_no_false_positive_unrelated(service):
    service.observe_and_store("i love rainy days in autumn")
    result = service.observe_and_store("i love rainy days in autumn a lot")
    # restating the same opinion must not be flagged as a conflict
    assert result["contradictions"] == []
