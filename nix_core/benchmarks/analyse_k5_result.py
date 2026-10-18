"""Report the k=5 confirmation result the way the pre-registration commits to.

Reads the run artifact written by ``benchmarks/memory_evaluation.py --samples 5``
and prints/records, per set and pooled:

* denominators: cases, unknown cases, planned samples, returned samples;
* fact recall, temporal correctness and stale count on the case means;
* abstention honesty on the 25 case-level outcomes with the pre-registered
  Wilson rule, plus a case-level cluster bootstrap *sensitivity* that never
  replaces the rule;
* paired memory comparisons against both baselines, pooled and per set;
* repeated-sampling variability: how many cases varied across their five
  samples and the mean within-case standard deviation;
* the missing-run check the runner recorded.

    python3 benchmarks/analyse_k5_result.py \
        --result benchmarks/results/d2_k5_confirm_2026-10-18.json \
        --json benchmarks/results/d2_k5_analysis_2026-10-18.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import memory_sets  # noqa: E402
from eval.acceptance import wilson_interval  # noqa: E402

SEED = 20261018


def _case_rows(result: dict) -> list[dict]:
    return list(result.get("scenarios") or [])


def _set_summary(name: str, result: dict) -> dict:
    rows = _case_rows(result)
    known = [row for row in rows if row.get("abstention_honesty") is None]
    unknown = [row for row in rows if row.get("abstention_honesty") is not None]
    honest = sum(1 for row in unknown if float(row["abstention_honesty"]) >= 0.5)
    low, high = wilson_interval(honest, len(unknown))
    spreads = [float(row.get("fact_recall_stdev") or 0.0) for row in rows]
    return {
        "set": name,
        "cases": len(rows),
        "known_cases": len(known),
        "unknown_cases": len(unknown),
        "planned_samples": result.get("planned_samples"),
        "failed_samples": result.get("failed_samples"),
        "failure_share": result.get("failure_share"),
        "baseline_failed_samples": result.get("baseline_failed_samples"),
        "fact_recall_mean_of_case_means": (
            round(sum(row["fact_recall"] for row in known) / len(known), 4) if known else None
        ),
        "fact_recall_all_cases": result.get("accuracy"),
        "temporal_correctness_all_cases": result.get("temporal_correctness"),
        "stale_fact_rate": result.get("stale_fact_rate"),
        "cases_with_full_recall": sum(1 for row in rows if row["fact_recall"] == 1.0),
        "cases_with_zero_recall": sum(1 for row in rows if row["fact_recall"] == 0.0),
        "abstention": {
            "honest": honest,
            "n": len(unknown),
            "rate": round(honest / len(unknown), 4) if unknown else None,
            "wilson_95": [round(low, 4), round(high, 4)],
        },
        "variability": {
            "cases_whose_recall_varied": sum(1 for value in spreads if value > 0),
            "cases": len(rows),
            "mean_within_case_stdev": round(sum(spreads) / len(spreads), 4) if spreads else None,
            "max_within_case_stdev": round(max(spreads), 4) if spreads else None,
        },
        "puca_mean_latency_ms": round(sum(row["latency_ms"] for row in rows) / len(rows), 1) if rows else None,
    }


def cluster_bootstrap_abstention(rows: list[dict], iterations: int = 10000) -> dict:
    """Sensitivity only: resample the 25 cases, never the 125 samples.

    The gate decision stays the pre-registered Wilson rule on the 25 case-level
    outcomes. This reports how often the same rule would still pass under a
    case-level resample, which is what a clustered uncertainty calculation buys
    over a point estimate.
    """
    cases = [float(row["abstention_honesty"]) for row in rows if row.get("abstention_honesty") is not None]
    if not cases:
        return {"cases": 0, "note": "no unknown cases in this result"}
    rng = random.Random(SEED)
    passed = 0
    for _ in range(iterations):
        draw = rng.choices(cases, k=len(cases))
        honest = sum(1 for value in draw if value >= 0.5)
        low, _ = wilson_interval(honest, len(draw))
        if honest >= memory_sets.ABSTENTION_MIN_HONEST and low >= memory_sets.ABSTENTION_MIN_LOWER:
            passed += 1
    return {
        "unit": "case",
        "cases": len(cases),
        "samples_never_used_as_cases": True,
        "resamples": iterations,
        "seed": SEED,
        "share_meeting_rule": round(passed / iterations, 4),
        "note": "sensitivity beside the pre-registered rule; the rule itself is the Wilson interval on cases",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    payload = json.loads(Path(args.result).read_text(encoding="utf-8"))
    results = payload["results"]
    freeze = payload.get("freeze") or {}

    per_set = {name: _set_summary(name, result) for name, result in results.items()}
    all_rows = [row for result in results.values() for row in result.get("scenarios") or []]
    pooled_unknown = [row for row in all_rows if row.get("abstention_honesty") is not None]
    honest = sum(1 for row in pooled_unknown if float(row["abstention_honesty"]) >= 0.5)
    low, high = wilson_interval(honest, len(pooled_unknown))
    pooled_abstention = {
        "honest": honest,
        "n": len(pooled_unknown),
        "rate": round(honest / len(pooled_unknown), 4) if pooled_unknown else None,
        "wilson_95": [round(low, 4), round(high, 4)],
        "rule_pass": bool(
            len(pooled_unknown) >= memory_sets.ABSTENTION_MIN_N
            and honest >= memory_sets.ABSTENTION_MIN_HONEST
            and low >= memory_sets.ABSTENTION_MIN_LOWER
        ),
        "cluster_bootstrap_sensitivity": cluster_bootstrap_abstention(pooled_unknown),
    }

    rows_by_set = {name: result.get("scenarios") or [] for name, result in results.items()}
    pooled_advantage = memory_sets.memory_advantage_report(rows_by_set)
    per_set_advantage = {
        name: memory_sets.memory_advantage_report({name: result.get("scenarios") or []})
        if all(arm in (result.get("scenarios") or [{}])[0] for arm in ("transcript", "filesystem"))
        else None
        for name, result in results.items()
    }

    spreads = [float(row.get("fact_recall_stdev") or 0.0) for row in all_rows]
    summary = {
        "kind": "d2-k5-analysis",
        "source_result": args.result,
        "freeze": {
            "tag": freeze.get("tag"),
            "sha256": freeze.get("sha256"),
            "frozen_at": freeze.get("frozen_at"),
            "prior_exposure": freeze.get("prior_exposure"),
            "samples_per_case": freeze.get("samples_per_case"),
            "retired": freeze.get("retired"),
        },
        "denominators": {
            "cases": len(all_rows),
            "unknown_cases": len(pooled_unknown),
            "planned_samples": sum(int(r.get("planned_samples") or 0) for r in results.values()),
            "failed_samples": sum(int(r.get("failed_samples") or 0) for r in results.values()),
        },
        "sample_failures": payload.get("sample_failures"),
        "per_set": per_set,
        "pooled": {
            "fact_recall_mean_of_case_means": round(
                sum(row["fact_recall"] for row in all_rows) / len(all_rows), 4
            ) if all_rows else None,
            "temporal_correctness": round(
                sum(row["temporal_correctness"] for row in all_rows) / len(all_rows), 4
            ) if all_rows else None,
            "stale_cases": sum(1 for row in all_rows if row["stale_fact"]),
            "cases_with_full_recall": sum(1 for row in all_rows if row["fact_recall"] == 1.0),
            "cases_with_zero_recall": sum(1 for row in all_rows if row["fact_recall"] == 0.0),
            "variability": {
                "cases_whose_recall_varied": sum(1 for value in spreads if value > 0),
                "cases": len(all_rows),
                "mean_within_case_stdev": round(statistics.fmean(spreads), 4) if spreads else None,
                "max_within_case_stdev": round(max(spreads), 4) if spreads else None,
            },
        },
        "abstention": pooled_abstention,
        "memory_advantage_pooled": pooled_advantage,
        "memory_advantage_per_set": per_set_advantage,
        "units": {
            "experimental_unit": "case",
            "interval_procedure": "paired percentile bootstrap over cases (memory) / Wilson on case outcomes (abstention)",
        },
    }

    print(json.dumps(summary, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
