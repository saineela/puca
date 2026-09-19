"""
Router accuracy report.

Runs the labeled corpus through the deterministic classifier and
prints per-route accuracy plus a confusion breakdown. The model
fallback (hosted by nix_knowledge) is exercised separately by the
live smoke test; here we require the rules layer to hit 100% on its
corpus and report how much traffic the rules resolve without a model
call at all.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import Counter  # noqa: E402

from router import classify  # noqa: E402
from router_corpus import CORPUS  # noqa: E402


def main() -> int:
    total = len(CORPUS)
    correct = 0
    per_route: dict[str, Counter] = {}
    failures: list[str] = []

    for request, expected in CORPUS:
        route, features = classify(request)

        stats = per_route.setdefault(
            expected, Counter(total=0, correct=0)
        )
        stats["total"] += 1

        if route == expected:
            correct += 1
            stats["correct"] += 1
        else:
            failures.append(
                f"  MISS {request!r}\n"
                f"       expected={expected} got={route} "
                f"rule={features.get('rule')}"
            )

    print(f"Corpus size: {total}")
    print(f"Correct:     {correct}")
    print(f"Accuracy:    {correct / total:.1%}")
    print()

    for expected, stats in sorted(per_route.items()):
        pct = stats["correct"] / stats["total"]
        print(f"  {expected:<10} {stats['correct']}/{stats['total']} ({pct:.0%})")

    if failures:
        print("\nFailures:")
        print("\n".join(failures))

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
