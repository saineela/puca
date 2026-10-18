"""D4 — the labelled routing pilot, run without touching thresholds.

This is deliberately small and explicitly a pilot. It reports:

* route-selection correctness against a hand label, separately per class;
* the undetermined share (unscored cases are excluded from the rates, not
  counted as failures);
* answer quality as *unmeasured* — no reply was judged here;
* the counterfactual as *not replayed* — running both routes on the same
  request is the next step, and is not implied by this file.

The routing thresholds are unchanged by running it.

    python3 benchmarks/routing_pilot.py
    python3 benchmarks/routing_pilot.py --json benchmarks/results/d4_routing_pilot.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import routing_ledger  # noqa: E402
from routing_engine import CoreRoutingEngine  # noqa: E402

DEFAULT_SET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "eval", "data", "routing_pilot.json",
)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run(payload: dict) -> dict:
    engine = CoreRoutingEngine()
    rows = []
    for case in payload["cases"]:
        decision = engine.decide(case["text"])
        label = case["label"]
        if label == "undetermined" or decision.route == "unknown":
            # An abstention is not a wrong route: the caller resolves it. It is
            # reported as undetermined and counted in the abstention rate, so
            # it cannot silently improve the accuracy number.
            correct = None
        elif label == "escalation_required":
            correct = decision.route != "chat"
        else:
            correct = decision.route == "chat"
        rows.append(
            routing_ledger.RoutedOutcome(
                request_id=case["id"],
                chosen_route=decision.route,
                label=label,
                route_correct=correct,
                answer_quality=None,          # not judged in this run
                counterfactual="not_replayed",  # both routes not run
                replayed_route="",
                counterfactual_correct=None,
                notes=f"{case['class']} | rule={decision.rule}",
            )
        )

    report = routing_ledger.ledger_report(rows)
    by_class: dict[str, dict[str, int]] = {}
    for case, row in zip(payload["cases"], rows):
        bucket = by_class.setdefault(
            case["class"], {"n": 0, "correct": 0, "scored": 0}
        )
        bucket["n"] += 1
        if row.route_correct is not None:
            bucket["scored"] += 1
            bucket["correct"] += int(row.route_correct)
    report["by_class"] = by_class
    scored = [row for row in rows if row.route_correct is not None]
    report["scored_accuracy"] = (
        round(sum(1 for row in scored if row.route_correct) / len(scored), 4)
        if scored else None
    )
    report["scored"] = len(scored)
    abstained = sum(1 for case, row in zip(payload["cases"], rows) if row.chosen_route == "unknown")
    report["abstained"] = abstained
    report["abstention_share"] = round(abstained / len(rows), 4) if rows else None
    report["answers_judged"] = 0
    report["counterfactual_replay"] = "not run"
    report["thresholds_changed"] = False
    report["set"] = {"tag": payload.get("tag"), "n": len(rows), "kind": payload.get("kind")}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default=DEFAULT_SET)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    report = run(load(args.set))
    print(f"D4 routing pilot — {report['set']['tag']} (n={report['set']['n']})")
    print(f"route selection (scored only): {report['scored_accuracy']} "
          f"({report['scored']}/{report['total']} scored)")
    print(f"false escalation: {report['false_escalation']}")
    print(f"false de-escalation: {report['false_de_escalation']}")
    print(f"undetermined share: {report['undetermined_share']}")
    print(f"router abstentions: {report['abstained']} ({report['abstention_share']})")
    print("answer quality: unmeasured (0 replies judged)")
    print("counterfactual: not replayed")
    for name, row in sorted(report["by_class"].items()):
        print(f"  {name:<20} {row['correct']}/{row['scored']} scored (n={row['n']})")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
