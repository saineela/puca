from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .engine import KnowledgeEngine
from .models import KnowledgeRecord


class Relation(str, Enum):
    SAME = "same"
    RELATED = "related"
    NEW = "new"


class ResolutionAction(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    IGNORE = "ignore"


@dataclass
class Resolution:
    action: ResolutionAction
    relation: Relation
    record: KnowledgeRecord | None
    reason: str


class KnowledgeResolver:
    """
    Determines how a newly extracted knowledge claim relates
    to existing knowledge.

    The resolver is authoritative for deciding whether a claim
    should create a new record, update an existing one, or be
    ignored as redundant.
    """

    def __init__(self, engine: KnowledgeEngine):
        self.engine = engine

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def resolve_preference(
        self,
        *,
        subject: str,
        value: str,
    ) -> Resolution:

        preferences = self.engine.search("preference")

        normalized_subject = self._normalize(subject)
        normalized_value = self._normalize(value)

        best_related: KnowledgeRecord | None = None

        for record in preferences:
            data = record.data

            existing_subject = self._normalize(
                str(data.get("subject", ""))
            )

            existing_value = self._normalize(
                str(data.get("value", ""))
            )

            # Exact same concept.
            if existing_subject == normalized_subject:

                # Same information already exists.
                if existing_value == normalized_value:
                    return Resolution(
                        action=ResolutionAction.IGNORE,
                        relation=Relation.SAME,
                        record=record,
                        reason="Preference already exists unchanged.",
                    )

                # Same subject, new value.
                return Resolution(
                    action=ResolutionAction.UPDATE,
                    relation=Relation.SAME,
                    record=record,
                    reason=(
                        "Existing preference refers to the same "
                        "subject but has a different value."
                    ),
                )

            # Keep track of related concepts, but do NOT overwrite them.
            if self._is_related(
                normalized_subject,
                existing_subject,
            ):
                best_related = record

        # Related knowledge should remain separate.
        if best_related:
            return Resolution(
                action=ResolutionAction.CREATE,
                relation=Relation.RELATED,
                record=best_related,
                reason=(
                    "Existing knowledge is related but represents "
                    "a different subject."
                ),
            )

        return Resolution(
            action=ResolutionAction.CREATE,
            relation=Relation.NEW,
            record=None,
            reason="No existing preference represents this subject.",
        )

    # ---------------------------------------------------------
    # NORMALIZATION
    # ---------------------------------------------------------

    @staticmethod
    def _normalize(value: str) -> str:
        value = value.lower().strip()

        value = re.sub(
            r"[^\w\s]",
            "",
            value,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value

    # ---------------------------------------------------------
    # RELATION DETECTION
    # ---------------------------------------------------------

    @staticmethod
    def _is_related(
        new_subject: str,
        existing_subject: str,
    ) -> bool:

        if not new_subject or not existing_subject:
            return False

        # One concept contains the other.
        #
        # Example:
        # "robotics"
        # "robotics electrical work"
        if (
            new_subject in existing_subject
            or existing_subject in new_subject
        ):
            return True

        new_words = set(new_subject.split())
        existing_words = set(existing_subject.split())

        overlap = new_words & existing_words

        return bool(overlap)
