from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class KnowledgeRecord:
    id: int
    knowledge_type: str
    data: dict[str, Any]

    confidence: float
    certainty: str

    source: str
    source_id: str | None

    status: str

    valid_from: datetime | None
    valid_until: datetime | None

    created_at: datetime
    updated_at: datetime

    # D1 provenance: id of the record that replaced this one, for a
    # supersede chain. None for live records and for pre-D1 rows.
    superseded_by: int | None = None

    @property
    def provenance(self) -> dict[str, Any]:
        """Where the record came from. Not a trust signal."""
        return {
            "source": self.source,
            "source_id": self.source_id,
            "superseded_by": self.superseded_by,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
        }


@dataclass(frozen=True)
class KnowledgeChange:
    id: int
    record_id: int | None

    operation: str
    knowledge_type: str

    before: dict[str, Any] | None
    after: dict[str, Any] | None

    source: str
    reason: str | None

    created_at: datetime
