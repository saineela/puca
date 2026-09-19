from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class KnowledgeRecord:
    """What nix_core receives when it asks for knowledge."""

    id: int
    knowledge_type: str
    content: str
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class KnowledgeProvider(Protocol):
    """
    The contract nix_knowledge implements.

    nix_core holds no facts of its own - whenever a request needs
    durable knowledge it asks a provider. Records are addressed by
    (knowledge_type, id) which is exactly the linkage nix_actions
    tracks on captured actions.
    """

    def lookup(
        self,
        *,
        query: str,
        knowledge_type: str | None = None,
        limit: int = 5,
    ) -> list[KnowledgeRecord]:
        """Return records relevant to a user request."""
        ...

    def version_of(
        self,
        *,
        knowledge_type: str,
        record_id: int,
    ) -> int:
        """Current version of a record; -1 if unknown/gone."""
        ...

    def is_current(
        self,
        *,
        knowledge_type: str,
        record_id: int,
        version: int,
    ) -> bool:
        """Whether a record still carries the version nix_core saw."""
        ...


class StaticKnowledgeProvider:
    """
    Minimal in-memory provider so nix_core runs standalone today.
    nix_knowledge replaces this by implementing KnowledgeProvider.
    """

    def __init__(
        self,
        records: list[KnowledgeRecord] | None = None,
    ):
        # (knowledge_type, id) -> KnowledgeRecord
        self._records: dict[tuple[str, int], KnowledgeRecord] = {}
        for record in records or []:
            self.put(record)

    def put(self, record: KnowledgeRecord) -> None:
        self._records[(record.knowledge_type, record.id)] = record

    def lookup(
        self,
        *,
        query: str,
        knowledge_type: str | None = None,
        limit: int = 5,
    ) -> list[KnowledgeRecord]:
        query_terms = set(query.lower().split())

        scored: list[tuple[int, KnowledgeRecord]] = []
        for record in self._records.values():
            if knowledge_type and record.knowledge_type != knowledge_type:
                continue

            haystack = (
                f"{record.content} {record.knowledge_type}".lower()
            )
            score = sum(
                1 for term in query_terms if term in haystack
            )
            if score:
                scored.append((score, record))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def version_of(
        self,
        *,
        knowledge_type: str,
        record_id: int,
    ) -> int:
        record = self._records.get((knowledge_type, record_id))
        return record.version if record else -1

    def is_current(
        self,
        *,
        knowledge_type: str,
        record_id: int,
        version: int,
    ) -> bool:
        return (
            self.version_of(
                knowledge_type=knowledge_type,
                record_id=record_id,
            )
            == version
        )
