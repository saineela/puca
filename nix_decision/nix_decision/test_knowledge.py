from __future__ import annotations

from pathlib import Path

from .knowledge import (
    KnowledgeDatabase,
    KnowledgeManager,
    KnowledgeOperation,
)


def main():

    db_path = Path("/tmp/nix_knowledge_test.db")

    if db_path.exists():
        db_path.unlink()

    db = KnowledgeDatabase(db_path)
    kb = KnowledgeManager(db)

    print()
    print("=== KNOWLEDGE BASE TEST ===")
    print()

    # ---------------------------------------------------------
    # CREATE
    # ---------------------------------------------------------

    event = kb.apply_operation(
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
            reason="User stated that CyberPatriot is scheduled.",
        )
    )

    print("CREATED:")
    print(event)
    print()

    # ---------------------------------------------------------
    # SEARCH
    # ---------------------------------------------------------

    events = kb.search("calendar_event")

    print("CALENDAR EVENTS:")

    for item in events:
        print(item.data)

    print()

    # ---------------------------------------------------------
    # UPDATE
    # ---------------------------------------------------------

    updated = kb.apply_operation(
        KnowledgeOperation(
            operation="update",
            knowledge_type="calendar_event",
            match={
                "title": "CyberPatriot",
            },
            changes={
                "status": "cancelled",
            },
            source="needle",
            reason="User said CyberPatriot was cancelled.",
        )
    )

    print("UPDATED:")
    print(updated.data)
    print()

    # ---------------------------------------------------------
    # VERIFY
    # ---------------------------------------------------------

    print("FINAL KNOWLEDGE:")

    for item in kb.search():
        print(
            item.id,
            item.knowledge_type,
            item.data,
        )

    print()
    print("Knowledge base test successful.")

    db.close()


if __name__ == "__main__":
    main()
