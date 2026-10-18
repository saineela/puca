"""Seed the D4 routing-replay store from eval/data/routing_replay_keys.json.

The replay needs a knowledge arm that can answer the held-out memory cases.
The facts live in a dedicated evaluation database, never in the owner's
canonical memory; the same protected-path guard as the D2 seeder applies.

    python3 benchmarks/seed_routing_replay.py --db /tmp/d4_eval/data/knowledge.db
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nix_knowledge.engine import KnowledgeEngine  # noqa: E402
from nix_knowledge.functions.facts import create_fact  # noqa: E402

DEFAULT_KEYS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "eval", "data", "routing_replay_keys.json",
)
PROTECTED = (
    Path("/root/data/knowledge.db"),
    Path("/root/nix_knowledge/knowledge.db"),
    Path("/root/data").resolve(),
)


def _is_protected(db: Path) -> bool:
    resolved = db.resolve()
    return any(
        resolved == path or (path.is_dir() and resolved.is_relative_to(path))
        for path in PROTECTED
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--keys", default=DEFAULT_KEYS)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    db = Path(args.db)
    if _is_protected(db):
        print(f"refusing to seed {db}: it is the owner's canonical memory", file=sys.stderr)
        return 2
    db.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.keys).read_text(encoding="utf-8"))
    engine = KnowledgeEngine(db)
    seeded = []
    for case_id, spec in (payload.get("cases") or {}).items():
        for fact in spec.get("seed") or []:
            result = create_fact(engine, fact, source="eval", reason=f"routing replay {case_id}")
            seeded.append({"case": case_id, "fact": fact, "record_id": result.get("record_id")})
    report = {
        "db": str(db),
        "keys": args.keys,
        "seeded": seeded,
        "active_facts": len(engine.search("fact")),
    }
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
