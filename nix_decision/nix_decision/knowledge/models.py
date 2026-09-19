from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
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


@dataclass
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
