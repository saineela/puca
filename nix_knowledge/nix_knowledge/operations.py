from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_OPERATIONS = {
    "create",
    "update",
    "delete",
}


VALID_KNOWLEDGE_TYPES = {
    "fact",
    "key",
    "person",
    "place",
    "relationship",
    "routine",
    "calendar_event",
    "preference",
    "project",
    "device",
}


@dataclass(frozen=True)
class KnowledgeOperation:
    operation: str
    knowledge_type: str

    data: dict[str, Any] | None = None
    match: dict[str, Any] | None = None
    changes: dict[str, Any] | None = None

    confidence: float = 1.0
    certainty: str = "known"

    source: str = "system"
    source_id: str | None = None

    reason: str | None = None

    valid_from: str | None = None
    valid_until: str | None = None

    def validate(self) -> None:
        if self.operation not in VALID_OPERATIONS:
            raise ValueError(
                f"Invalid knowledge operation: {self.operation}"
            )

        if self.knowledge_type not in VALID_KNOWLEDGE_TYPES:
            raise ValueError(
                f"Invalid knowledge type: {self.knowledge_type}"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "Knowledge confidence must be between 0 and 1"
            )

        if not self.source:
            raise ValueError("Knowledge source cannot be empty")

        if self.operation == "create":
            if not self.data:
                raise ValueError(
                    "Create operation requires data"
                )

        elif self.operation == "update":
            if not self.match:
                raise ValueError(
                    "Update operation requires match"
                )

            if not self.changes:
                raise ValueError(
                    "Update operation requires changes"
                )

        elif self.operation == "delete":
            if not self.match:
                raise ValueError(
                    "Delete operation requires match"
                )
