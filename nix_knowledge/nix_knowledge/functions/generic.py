from __future__ import annotations

from typing import Any

from nix_knowledge.operations import KnowledgeOperation


def create_knowledge(
    engine,
    knowledge_type: str,
    data: dict[str, Any],
    source: str = "knowledge_needle",
    source_id: str | None = None,
    confidence: float = 1.0,
    certainty: str = "certain",
    reason: str | None = None,
):
    operation = KnowledgeOperation(
        operation="create",
        knowledge_type=knowledge_type,
        data=data,
        confidence=confidence,
        certainty=certainty,
        source=source,
        source_id=source_id,
        reason=reason or "Created by Knowledge Needle",
    )

    return engine.apply(operation)
