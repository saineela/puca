from __future__ import annotations

from ..engine import KnowledgeEngine


def create_fact(
    engine: KnowledgeEngine,
    value: str,
    source: str = "knowledge_needle",
    reason: str | None = None,
):
    # -------------------------------------------------------------
    # Duplicate gate: a near-exact restatement of an already-stored
    # fact must never write a second record. The neural semantic
    # layer (cosine >= 0.92) decides; a failure there degrades to the
    # old write path rather than blocking storage entirely.
    # -------------------------------------------------------------
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
        "record_id": record.id,
        "data": record.data,
    }
