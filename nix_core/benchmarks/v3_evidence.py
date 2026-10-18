"""Real-log evidence for the PUCA system model v3 decisions.

This reads the request log the console actually wrote
(``nix_core/logs/requests-YYYY-MM-DD.jsonl``) and reports what it can prove
about the running system, with the denominator next to every number:

* **D3** - how many requests carry all five context-section sizes;
* **D4** - the route mix, the counterfactual status, and the share of turns
  whose correct route is still unlabelled (reported, never hidden);
* **D5** - malformed plan envelopes, and the validate-retry rate that decides
  whether grammar-constrained decoding is worth building;
* **D10** - model calls with the model, backend and timing fields recorded.

It changes nothing. Run it against the frozen set with ``--tag`` once the
labelled routing set exists; until then the numbers are descriptive, not a
gate.

    python3 benchmarks/v3_evidence.py
    python3 benchmarks/v3_evidence.py --days 2 --json benchmarks/results/v3_evidence.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from datetime import date, timedelta
from typing import Any

DEFAULT_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)
SECTIONS = ("policy", "memory", "records", "history", "tools")


def read_entries(log_dir: str, days: int = 0) -> list[dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(log_dir, "requests-*.jsonl")))
    if days:
        cutoff = date.today() - timedelta(days=days - 1)
        paths = [
            path for path in paths
            if _day_of(path) is None or _day_of(path) >= cutoff
        ]
    entries: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def _day_of(path: str) -> date | None:
    stem = os.path.basename(path).removeprefix("requests-").removesuffix(".jsonl")
    try:
        return date.fromisoformat(stem)
    except ValueError:
        return None


def report(entries: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(entries)
    routes: Counter[str] = Counter()
    counterfactuals: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    sections_complete = 0
    model_calls = 0
    envelope_failures = 0
    envelopes_checked = 0
    planner_runs = 0
    grounded_retries = 0

    for entry in entries:
        routes[str(entry.get("route") or "unknown")] += 1
        ledger = entry.get("routing_ledger") or {}
        counterfactuals[str(ledger.get("counterfactual") or "unmeasured")] += 1
        outcomes[str(ledger.get("outcome") or "unlabelled")] += 1
        sizes = entry.get("context_sections")
        if isinstance(sizes, dict) and all(name in sizes for name in SECTIONS):
            sections_complete += 1
        details = entry.get("details") or {}
        if details.get("model"):
            model_calls += 1
        if details.get("skill_planner_called"):
            planner_runs += 1
            if details.get("skill_planner_grounded_retry"):
                grounded_retries += 1
        validation = details.get("plan_envelope_validation")
        if isinstance(validation, dict):
            envelopes_checked += 1
            if validation.get("schema"):
                envelope_failures += 1

    def share(part: int, whole: int) -> float:
        return round(part / whole, 4) if whole else 0.0

    return {
        "requests": total,
        "D3_context_sections": {
            "with_all_five": sections_complete,
            "share": share(sections_complete, total),
            "sections": list(SECTIONS),
            "note": "logging-only in Wave 1; budgets are enforced after the D2 curve",
        },
        "D4_routing": {
            "route_mix": dict(routes),
            "counterfactual": dict(counterfactuals),
            "outcomes": dict(outcomes),
            "undetermined_or_unlabelled": outcomes.get("unlabelled", 0),
            "undetermined_share": share(outcomes.get("unlabelled", 0), total),
            "note": (
                "false escalation and false de-escalation need the frozen "
                "labelled routing set and its controlled replays"
            ),
        },
        "D5_plan_envelopes": {
            "plans_checked": envelopes_checked,
            "malformed": envelope_failures,
            "malformed_rate": share(envelope_failures, envelopes_checked),
            "planner_runs": planner_runs,
            "validate_retry_rate": share(grounded_retries, planner_runs),
            "note": (
                "a validate-retry rate at zero drops the grammar work rather "
                "than building it"
            ),
        },
        "D10_inference": {
            "calls_with_model_recorded": model_calls,
            "share": share(model_calls, total),
        },
    }


def markdown(data: dict[str, Any], tag: str) -> str:
    lines = [
        f"# PUCA system model v3 — real-log evidence ({tag})",
        "",
        f"Requests read: **{data['requests']}**",
        "",
        "## D3 — context sections",
        "",
        f"All five section sizes logged: {data['D3_context_sections']['with_all_five']} "
        f"({data['D3_context_sections']['share']:.1%})",
        "",
        "## D4 — routing ledger",
        "",
        f"Route mix: {data['D4_routing']['route_mix']}",
        "",
        f"Counterfactual: {data['D4_routing']['counterfactual']}",
        "",
        f"Outcomes: {data['D4_routing']['outcomes']} "
        f"(unlabelled share {data['D4_routing']['undetermined_share']:.1%})",
        "",
        "## D5 — plan envelopes",
        "",
        f"Plans checked: {data['D5_plan_envelopes']['plans_checked']}; "
        f"malformed: {data['D5_plan_envelopes']['malformed']} "
        f"({data['D5_plan_envelopes']['malformed_rate']:.1%})",
        "",
        f"Planner runs: {data['D5_plan_envelopes']['planner_runs']}; "
        f"validate-retry rate: {data['D5_plan_envelopes']['validate_retry_rate']:.1%}",
        "",
        "## D10 — inference records",
        "",
        f"Calls with the model recorded: {data['D10_inference']['calls_with_model_recorded']} "
        f"({data['D10_inference']['share']:.1%})",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--days", type=int, default=0, help="0 means every log file")
    parser.add_argument("--tag", default="descriptive-run")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    data = report(read_entries(args.log_dir, args.days))
    print(markdown(data, args.tag))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"tag": args.tag, **data}, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
