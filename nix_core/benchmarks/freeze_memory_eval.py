"""Freeze a D2 set *before* the run, and verify a result against that freeze.

The evaluation writes the same manifest into its result file, but WP2 requires
the hash to be recorded before the first scenario is answered: a hash written
only after the run could describe a set that changed during it. This script
writes the freeze artifact first; ``memory_evaluation.py --json`` records the
identical sha256, and ``--verify`` proves the two match.

    python3 benchmarks/freeze_memory_eval.py \\
        --scenarios eval/data/memory_scenarios_confirm.json \\
        --tag confirm-2026-10-18 \\
        --json benchmarks/results/d2_confirm_freeze_2026-10-18.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import memory_sets  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--sets", nargs="*", default=["development", "heldout", "adversarial"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", default="luna-v6-contextual-v1")
    parser.add_argument("--baseline-mode", choices=("console", "ollama"), default="ollama")
    parser.add_argument(
        "--baseline-model",
        default=os.environ.get("NIX_BASELINE_MODEL", "qwen3.5:4b"),
        help="the ollama model behind the console's exposed assistant model id",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="samples per case for the repeat protocol (the approved design uses 5)",
    )
    parser.add_argument("--prior-exposure", default="none recorded")
    parser.add_argument("--retired", nargs="*", default=[])
    parser.add_argument("--json", required=True, help="where to write the freeze artifact")
    parser.add_argument(
        "--verify",
        default="",
        help="a memory_evaluation result whose recorded freeze must match this one",
    )
    args = parser.parse_args()

    sets = memory_sets.load_sets(args.scenarios)
    missing = [name for name in args.sets if name not in sets]
    if missing:
        print(f"{args.scenarios} has no set(s): {', '.join(missing)}", file=sys.stderr)
        return 2
    note = memory_sets.frozen_manifest(
        {name: sets[name] for name in args.sets},
        tag=args.tag,
        model=args.model,
        hardware=os.uname().machine,
        prior_exposure=args.prior_exposure,
        retired=args.retired,
        baseline_mode=args.baseline_mode,
        baseline_model=args.baseline_model if args.baseline_mode == "ollama" else "",
        samples=args.samples,
    )
    note["scenarios_file"] = args.scenarios
    note["counts"] = {
        name: {
            "total": len(sets[name]),
            "unknown": sum(1 for row in sets[name] if row.unknown),
        }
        for name in args.sets
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
    print(f"froze {args.tag}: {note['sha256']} -> {args.json}")

    if args.verify:
        result = json.loads(Path(args.verify).read_text(encoding="utf-8"))
        recorded = (result.get("freeze") or {}).get("sha256")
        if recorded != note["sha256"]:
            print(
                f"freeze mismatch: result recorded {recorded}, this set hashes "
                f"{note['sha256']}",
                file=sys.stderr,
            )
            return 1
        print(f"verified: {args.verify} used the same frozen set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
