from __future__ import annotations

import json
import re

from nix_knowledge.engine import KnowledgeEngine
from nix_knowledge.needle import KnowledgeNeedle


# Arrow-key history recall embeds terminal escape sequences in the
# input string; strip them before anything touches the Knowledge Base.
_ANSI_ESCAPE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-Z\\-_])"
)


def print_base(engine):
    records = engine.search()

    print()
    print("=" * 80)
    print("NIX KNOWLEDGE BASE")
    print("=" * 80)

    if not records:
        print()
        print("Knowledge Base is empty.")
        print()
        return

    print()

    for record in records:
        print(f"ID:         {record.id}")
        print(f"Type:       {record.knowledge_type}")
        print(f"Confidence: {record.confidence}")
        print(f"Certainty:  {record.certainty}")
        print(f"Source:     {record.source}")
        print(f"Status:     {record.status}")
        print(f"Created:    {record.created_at}")
        print(f"Updated:    {record.updated_at}")
        print("Data:")
        print(
            json.dumps(
                record.data,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
        print()
        print("-" * 80)

    print()


def main():
    engine = KnowledgeEngine("knowledge.db")

    try:
        needle = KnowledgeNeedle(
            engine,
            timezone="America/Chicago",
        )

        print()
        print("=" * 80)
        print("NIX KNOWLEDGE ENGINE")
        print("=" * 80)
        print()
        print("Raw user request mode.")
        print()
        print("Commands:")
        print("  /base   Show complete Knowledge Base")
        print("  /quit   Exit")
        print()

        while True:
            try:
                request = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            request = _ANSI_ESCAPE.sub("", request).strip()

            if not request:
                continue

            if request in {"/quit", "/exit"}:
                break

            if request == "/base":
                print_base(engine)
                continue

            try:
                result = needle.process(request)

                print("RESULT:")
                print(
                    json.dumps(
                        result,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                )
                print()

            except Exception as exc:
                print()
                print("ERROR:")
                print(f"{type(exc).__name__}: {exc}")
                print()

    finally:
        engine.close()


def base():
    engine = KnowledgeEngine("knowledge.db")

    try:
        print_base(engine)
    finally:
        engine.close()
