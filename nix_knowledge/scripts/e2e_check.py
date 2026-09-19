from __future__ import annotations

import json
import sys

from nix_knowledge.engine import KnowledgeEngine
from nix_knowledge.needle import KnowledgeNeedle


REQUESTS = [
    "I have a dentist appointment tomorrow",
    "I have a meeting with Bob tomorrow at 3pm",
    "what is on my calendar",
    "remember that I like robotics",
    "what do you remember about robotics",
    "cancel my dentist appointment",
    "what is on my calendar",
    "move my meeting with Bob to friday at 6pm",
    "what is on my calendar",
    "do you remember I like robotics",
    "remember that my favorite language is Python",
]


def main() -> int:
    engine = KnowledgeEngine("e2e_test.db")
    failures = 0

    try:
        needle = KnowledgeNeedle(engine, timezone="America/Chicago")

        for request in REQUESTS:
            result = needle.process(request)

            function = result["function"]
            payload = result["result"]

            print("=" * 70)
            print("REQUEST: ", request)
            print("FUNCTION:", json.dumps(function))
            print("RESULT:  ", json.dumps(payload, default=str)[:400])

            if function is None:
                failures += 1
                print(">>> FAIL: no function selected")

    finally:
        engine.close()

    print("=" * 70)
    print(f"FAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
