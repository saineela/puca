"""Dual-score recorded D2 replies with the frozen rule and the v2 proposal.

The frozen WP2 result stores metrics, not answers, so it cannot be re-scored
without a re-run. This runner works on any artifact that does store the replies
(the WP6 smoke result does), applies both rules to the same text, and writes a
comparison file that keeps every per-case score and reports a paired interval
on the mean delta over cases.

    python3 benchmarks/rescore_scoring_v2.py \\
        --cases benchmarks/results/d2_wp6_smoke_2026-10-18.json \\
        --set eval/data/memory_scenarios_confirm.json \\
        --out benchmarks/results/d2_scoring_v2_comparison_2026-10-18.json

It never modifies the frozen result file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import scoring_v2  # noqa: E402

DEFAULT_SET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "eval", "data", "memory_scenarios_confirm.json",
)


def _scenario_index(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    sets = payload.get("sets") if isinstance(payload, dict) else payload
    index: dict[str, dict] = {}
    if isinstance(sets, dict):
        for rows in sets.values():
            for row in rows:
                index[str(row.get("id"))] = row
    elif isinstance(sets, list):
        for row in sets:
            index[str(row.get("id"))] = row
    return index


def _cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return payload["cases"]
    if isinstance(payload, list):
        return payload
    raise SystemExit(f"{path} does not contain a cases list")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--set", default=DEFAULT_SET)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    scored = scoring_v2.rescore_cases(_cases(args.cases), _scenario_index(args.set))
    report = {
        "kind": "scoring-rule-comparison",
        "tag": "d2-scoring-v2-comparison-2026-10-18",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_cases": os.path.relpath(args.cases),
        "frozen_set": os.path.relpath(args.set),
        "rules": {
            scoring_v2.BASELINE_VERSION: "eval/memory_sets.py score_answer (frozen; unchanged)",
            scoring_v2.VERSION: "eval/scoring_v2.py score_answer_v2 (proposal)",
        },
        "summary": scoring_v2.summarize(scored),
        "cases": scored["rows"],
    }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, ensure_ascii=False)
        print(f"wrote {args.out}")

    summary = report["summary"]
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
