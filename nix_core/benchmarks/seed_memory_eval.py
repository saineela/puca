"""Seed a dedicated D2 evaluation store, never the owner's canonical memory.

The evaluation needs representative facts in a store of its own. This script
refuses to touch a store that is not explicitly named, and refuses the known
canonical paths, so a run cannot seed the assistant's real memory by accident.

    python3 benchmarks/seed_memory_eval.py \
        --db /tmp/d2_eval/data/knowledge.db \
        --scenarios eval/data/memory_scenarios_fresh.json

Each cluster in the scenario file's ``seed`` block has a current fact and,
usually, a superseded one. The old fact is written first and then superseded,
so the store exercises the same provenance path as a real update.
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

DEFAULT_SCENARIOS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "eval", "data", "memory_scenarios_fresh.json",
)

#: Paths that hold the owner's real memory. The evaluation must never seed
#: these; a dedicated instance uses a path under its own directory.
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


def seed(engine: KnowledgeEngine, clusters: list[dict]) -> dict:
    created = 0
    superseded = 0
    for cluster in clusters:
        current = str(cluster.get("current") or "").strip()
        old = str(cluster.get("old") or "").strip()
        if not current:
            continue
        if old:
            old_result = create_fact(engine, old, source="eval", reason="fresh-eval seed")
            old_id = old_result.get("record_id")
            if old_id:
                engine.supersede(
                    int(old_id),
                    data={"value": current},
                    source="eval",
                    reason="fresh-eval update",
                )
                superseded += 1
            else:
                create_fact(engine, current, source="eval", reason="fresh-eval seed")
                created += 1
        else:
            create_fact(engine, current, source="eval", reason="fresh-eval seed")
            created += 1
    return {
        "clusters": len(clusters),
        "created_without_history": created,
        "superseded_updates": superseded,
        "active_facts": len(engine.search("fact")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="a dedicated evaluation database")
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    db = Path(args.db)
    if _is_protected(db):
        print(
            f"refusing to seed {db}: it is the owner's canonical memory, not an "
            "evaluation instance",
            file=sys.stderr,
        )
        return 2
    if not db.parent.exists():
        print(f"parent directory does not exist: {db.parent}", file=sys.stderr)
        return 2

    payload = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    clusters = payload.get("seed") or []
    if not clusters:
        print(f"{args.scenarios} has no seed block", file=sys.stderr)
        return 2

    engine = KnowledgeEngine(db)
    result = seed(engine, clusters)
    result.update({"db": str(db), "set": payload.get("tag"), "scenarios": args.scenarios})
    print(json.dumps(result, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
