from __future__ import annotations

import re

from ..engine import KnowledgeEngine
from ..operations import KnowledgeOperation

#: One request may remove this many facts at most. A description broad enough
#: to match more is a bulk delete, which Core already answers with a refusal:
#: the tool must not become the loop hole for it.
DELETE_MAX_MATCHES = 3


def _normalized(value: str) -> str:
    """The comparison form of a fact: case, spacing and end punctuation gone."""
    text = re.sub(r"\s+", " ", str(value or "")).strip().casefold()
    return text.rstrip(" .!?")


def _exact_duplicate(engine: KnowledgeEngine, value: str) -> tuple[int, str] | None:
    """An active fact with the same words, found without the semantic layer.

    The near-duplicate gate below needs the embedding model, and when that
    model cannot load (measured live: the GPU was busy with the console, so
    the semantic layer was silently unavailable) every restatement of the
    same sentence became a new record - six identical "my dog's name is
    mochi" rows. Exact sameness does not need a model.
    """
    wanted = _normalized(value)
    if not wanted:
        return None
    try:
        records = engine.search("fact")
    except Exception:  # noqa: BLE001 - a lookup must never block the write
        return None
    for record in records:
        stored = str(record.data.get("value", ""))
        if _normalized(stored) == wanted:
            return record.id, stored
    return None


def create_fact(
    engine: KnowledgeEngine,
    value: str,
    source: str = "knowledge_needle",
    reason: str | None = None,
):
    # -------------------------------------------------------------
    # Duplicate gate: a restatement of an already-stored fact must
    # never write a second record. The same words are caught first,
    # deterministically, so the gate holds even when the semantic layer
    # is unavailable; the neural layer then catches near-duplicates
    # (cosine >= 0.92), and a failure there degrades to the write path.
    # -------------------------------------------------------------
    same = _exact_duplicate(engine, value)
    if same is not None:
        record_id, stored = same
        return {
            "ok": True,
            "operation": "DUPLICATE",
            "record_type": "fact",
            "duplicate": True,
            "duplicate_of": record_id,
            "duplicate_text": stored,
            "record_id": record_id,
            "data": {"value": stored},
        }

    if engine.semantic is not None:
        try:
            duplicate = engine.semantic.check_duplicate(value)
        except Exception:  # noqa: BLE001 - dedup must never block
            duplicate = None
        if duplicate is not None:
            dup_record_id, dup_text = duplicate
            # stale-vector guard: the matched record must still exist
            # (vectors can outlive a record deleted outside the engine)
            try:
                engine.get(dup_record_id)
            except Exception:  # noqa: BLE001
                duplicate = None
        if duplicate is not None:
            return {
                "ok": True,
                "operation": "DUPLICATE",
                "record_type": "fact",
                "duplicate": True,
                "duplicate_of": dup_record_id,
                "duplicate_text": dup_text,
                "record_id": dup_record_id,
                "data": {"value": dup_text},
            }

    record = engine.create(
        "fact",
        {"value": value},
        source=source,
        reason=reason,
    )

    return {
        "ok": True,
        "operation": "CREATE",
        "record_type": "fact",
        "record_id": record.id,
        "data": record.data,
    }


def matching_facts(engine: KnowledgeEngine, query: str) -> list:
    """Active facts whose sentence matches a description of the fact.

    The same content-word comparison the finder uses, so "the note about my
    old address" reaches "My old address was 14 Willow Lane." without the
    description having to be the stored sentence.
    """
    from ..query import content_words, matches_words, words

    wanted = content_words(query)
    literal = _normalized(query)
    if not wanted and not literal:
        return []
    matched = []
    for record in engine.search("fact"):
        value = str(record.data.get("value", ""))
        if not value:
            continue
        if literal and literal in _normalized(value):
            matched.append(record)
            continue
        if wanted and matches_words(wanted, value):
            matched.append(record)
            continue
        # A description can be one distinctive word the stored sentence
        # carries in a different form ("licence" vs "license"): compare the
        # stems as a last, still deterministic, check.
        if wanted and set(wanted) & set(words(value)):
            matched.append(record)
    return matched


def delete_facts(
    engine: KnowledgeEngine,
    query: str,
    *,
    confirm: bool = False,
    source: str = "knowledge_needle",
    reason: str | None = None,
) -> dict:
    """Delete stored facts by description, behind an explicit confirmation.

    Deletion is destructive, so the first call never writes: it returns the
    exact records the description reaches and asks for a confirmation. Only
    ``confirm=True`` soft-deletes them (the row stays with ``status='deleted'``,
    which keeps the audit trail). A description that reaches nothing, or more
    records than one request may remove, deletes nothing.
    """
    query = str(query or "").strip()
    if not query:
        return {
            "ok": False,
            "status": "refused_empty_query",
            "operation": "DELETE_FACT",
            "detail": "Name the fact to delete; nothing was removed.",
        }

    matched = matching_facts(engine, query)
    if not matched:
        return {
            "ok": False,
            "status": "not_found",
            "operation": "DELETE_FACT",
            "query": query,
            "deleted": [],
            "detail": "No stored fact matches that description; nothing was removed.",
        }

    if len(matched) > DELETE_MAX_MATCHES:
        return {
            "ok": False,
            "status": "refused_too_broad",
            "operation": "DELETE_FACT",
            "query": query,
            "match_count": len(matched),
            "deleted": [],
            "detail": (
                "That description matches more facts than one request may "
                "delete. Name the single fact to remove; nothing was removed."
            ),
        }

    candidates = [
        {
            "record_id": record.id,
            "value": str(record.data.get("value", "")),
        }
        for record in matched
    ]

    if not confirm:
        return {
            "ok": True,
            "status": "needs_confirmation",
            "operation": "DELETE_FACT_PENDING",
            "query": query,
            "matches": candidates,
            "deleted": [],
            "detail": "Nothing was removed yet; the user must confirm.",
        }

    deleted = []
    for record in matched:
        engine.apply(
            KnowledgeOperation(
                operation="delete",
                knowledge_type="fact",
                match={"id": record.id},
                source=source,
                reason=reason or "delete by description",
            )
        )
        deleted.append(
            {
                "record_id": record.id,
                "value": str(record.data.get("value", "")),
            }
        )

    return {
        "ok": True,
        "status": "deleted",
        "operation": "DELETE_FACT",
        "query": query,
        "count": len(deleted),
        "deleted": deleted,
    }
