from __future__ import annotations

from pathlib import Path

import pytest

from nix_knowledge import (
    KnowledgeEngine,
    KnowledgeOperation,
)


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "test.db"

    engine = KnowledgeEngine(db)

    yield engine

    engine.close()


def test_create(engine):

    record = engine.apply(
        KnowledgeOperation(
            operation="create",
            knowledge_type="calendar_event",
            data={
                "title": "CyberPatriot",
                "start": "2026-09-06T18:00:00-05:00",
                "end": "2026-09-06T20:00:00-05:00",
                "status": "scheduled",
            },
            confidence=0.95,
            certainty="probable",
            source="user_statement",
            reason="User stated that CyberPatriot is tomorrow.",
        )
    )

    assert record.id > 0
    assert record.knowledge_type == "calendar_event"
    assert record.data["title"] == "CyberPatriot"
    assert record.confidence == 0.95


def test_update(engine):

    created = engine.apply(
        KnowledgeOperation(
            operation="create",
            knowledge_type="calendar_event",
            data={
                "title": "CyberPatriot",
                "status": "scheduled",
            },
            source="user_statement",
        )
    )

    updated = engine.apply(
        KnowledgeOperation(
            operation="update",
            knowledge_type="calendar_event",
            match={
                "title": "CyberPatriot",
            },
            changes={
                "status": "cancelled",
            },
            source="user_statement",
            reason="User cancelled CyberPatriot.",
        )
    )

    assert updated.id == created.id
    assert updated.data["status"] == "cancelled"


def test_delete(engine):

    created = engine.apply(
        KnowledgeOperation(
            operation="create",
            knowledge_type="fact",
            data={
                "text": "Test fact",
            },
        )
    )

    engine.apply(
        KnowledgeOperation(
            operation="delete",
            knowledge_type="fact",
            match={
                "text": "Test fact",
            },
        )
    )

    assert engine.search("fact") == []


def test_multiple_matches_are_rejected(engine):

    engine.apply(
        KnowledgeOperation(
            operation="create",
            knowledge_type="person",
            data={"name": "Alex"},
        )
    )

    engine.apply(
        KnowledgeOperation(
            operation="create",
            knowledge_type="person",
            data={"name": "Alex"},
        )
    )

    with pytest.raises(ValueError):
        engine.apply(
            KnowledgeOperation(
                operation="update",
                knowledge_type="person",
                match={"name": "Alex"},
                changes={"relationship": "friend"},
            )
        )


def test_invalid_operation_is_rejected(engine):

    with pytest.raises(ValueError):
        engine.apply(
            KnowledgeOperation(
                operation="destroy",
                knowledge_type="fact",
                data={"x": 1},
            )
        )


def test_history_is_recorded(engine):

    record = engine.apply(
        KnowledgeOperation(
            operation="create",
            knowledge_type="fact",
            data={
                "text": "Nix knows this.",
            },
            source="needle",
            reason="Needle extracted this fact.",
        )
    )

    changes = engine.history.list_changes(
        record.id
    )

    assert len(changes) == 1
    assert changes[0].operation == "create"
    assert changes[0].source == "needle"


def test_temporal_relevance(engine):

    record = engine.apply(
        KnowledgeOperation(
            operation="create",
            knowledge_type="calendar_event",
            data={
                "title": "Future Event",
            },
            valid_from="2999-01-01T00:00:00+00:00",
        )
    )

    relevant = engine.relevant_now(
        "calendar_event"
    )

    assert record not in relevant
