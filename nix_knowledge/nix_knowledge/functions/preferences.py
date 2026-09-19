from __future__ import annotations

from typing import Any

from ..engine import KnowledgeEngine
from ..operations import KnowledgeOperation
from ..resolver import (
    KnowledgeResolver,
    ResolutionAction,
)


def create_preference(
    engine: KnowledgeEngine,
    *,
    subject: str,
    value: str,
    source: str = "knowledge_needle",
    reason: str | None = None,
):
    """
    Store a preference intelligently.

    The resolver decides whether the preference should:
      - create a new record
      - update an existing record
      - be ignored as already known
    """

    resolver = KnowledgeResolver(engine)

    resolution = resolver.resolve_preference(
        subject=subject,
        value=value,
    )

    if resolution.action == ResolutionAction.IGNORE:
        return {
            "ok": True,
            "action": "ignored",
            "relation": resolution.relation.value,
            "record_id": (
                resolution.record.id
                if resolution.record
                else None
            ),
            "reason": resolution.reason,
            "data": (
                resolution.record.data
                if resolution.record
                else None
            ),
        }

    if resolution.action == ResolutionAction.UPDATE:
        if resolution.record is None:
            raise RuntimeError(
                "Resolver returned UPDATE without an existing record."
            )

        record = engine.apply(
            KnowledgeOperation(
                operation="update",
                knowledge_type="preference",
                match={
                    "id": resolution.record.id,
                },
                changes={
                    "subject": subject,
                    "value": value,
                },
                source=source,
                reason=reason or resolution.reason,
            )
        )

        return {
            "ok": True,
            "action": "updated",
            "relation": resolution.relation.value,
            "record_id": record.id,
            "reason": resolution.reason,
            "data": record.data,
        }

    record = engine.create(
        "preference",
        {
            "subject": subject,
            "value": value,
        },
        source=source,
        reason=reason or resolution.reason,
    )

    return {
        "ok": True,
        "action": "created",
        "relation": resolution.relation.value,
        "record_id": record.id,
        "reason": resolution.reason,
        "data": record.data,
    }


def upsert_preference(
    engine: KnowledgeEngine,
    *,
    subject: str,
    value: str,
    source: str = "knowledge_needle",
    reason: str | None = None,
):
    """
    Backwards-compatible alias for create_preference().

    Despite the name, the resolver determines whether the operation
    actually creates, updates, or ignores the preference.
    """

    return create_preference(
        engine,
        subject=subject,
        value=value,
        source=source,
        reason=reason,
    )


def find_preferences(
    engine: KnowledgeEngine,
    *,
    subject: str | None = None,
    value: str | None = None,
) -> list:
    """
    Find active preference records.

    If subject and/or value are supplied, matching is performed
    against the preference data.
    """

    records = engine.search(
        "preference",
        status="active",
    )

    if subject is None and value is None:
        return records

    results = []

    for record in records:
        data = record.data

        if subject is not None:
            if str(data.get("subject", "")).lower().strip() != (
                subject.lower().strip()
            ):
                continue

        if value is not None:
            if str(data.get("value", "")).lower().strip() != (
                value.lower().strip()
            ):
                continue

        results.append(record)

    return results


def update_preference(
    engine: KnowledgeEngine,
    *,
    record_id: int,
    subject: str | None = None,
    value: str | None = None,
    source: str = "knowledge_needle",
    reason: str | None = None,
):
    """
    Explicitly update an existing preference by record ID.

    This bypasses resolver decision-making because the caller has
    explicitly requested an update.
    """

    changes: dict[str, Any] = {}

    if subject is not None:
        changes["subject"] = subject

    if value is not None:
        changes["value"] = value

    if not changes:
        raise ValueError(
            "update_preference requires subject and/or value"
        )

    return engine.apply(
        KnowledgeOperation(
            operation="update",
            knowledge_type="preference",
            match={
                "id": record_id,
            },
            changes=changes,
            source=source,
            reason=reason,
        )
    )
