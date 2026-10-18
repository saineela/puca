"""Run the D6 reliability gate: pass^5 quick signal plus the k=20 paired report.

    python3 benchmarks/reliability_gate.py --trials 20 --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import reliability  # noqa: E402
from eval.reliability_flows import paired_runs, run_pair, scenarios  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--min-scenarios",
        type=int,
        default=20,
        help="distinct scenarios per flow required before any claim (scenario is the unit)",
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    baseline_trials: dict[str, list[reliability.Trial]] = {}
    treatment_trials: dict[str, list[reliability.Trial]] = {}
    for scenario in scenarios():
        for index in range(5):
            baseline, treatment = run_pair(scenario, seed=index)
            baseline_trials.setdefault(scenario.flow, []).append(
                reliability.Trial(scenario.id, scenario.flow, baseline.passed)
            )
            treatment_trials.setdefault(scenario.flow, []).append(
                reliability.Trial(scenario.id, scenario.flow, treatment.passed)
            )
    print("pass^5 quick signal (pre-merge):")
    for flow in sorted(baseline_trials):
        k = min(5, len(baseline_trials[flow]))
        print(
            f"  {flow:<14} baseline={reliability.pass_power(baseline_trials[flow], k=k)} "
            f"treatment={reliability.pass_power(treatment_trials[flow], k=k)}"
        )

    pairs = paired_runs(args.trials)
    quick = reliability.gate(pairs, iterations=args.iterations)
    print(f"\nreplay-counted gate (k={args.trials} replays per scenario): {quick['decision']}")
    print("  note: repeated deterministic replays are not independent observations;")
    print("        this is a pre-merge signal only.")
    for flow, row in quick.get("flows", {}).items():
        print(
            f"  {flow:<14} baseline={row['baseline_passes']}/{row['trials']} "
            f"treatment={row['candidate_passes']}/{row['trials']} "
            f"wins={row['mcnemar']['candidate_only']} "
            f"losses={row['mcnemar']['baseline_only']} "
            f"ties={row['mcnemar']['ties']} "
            f"mcnemar_p={row['mcnemar']['p_value']}"
        )

    report = reliability.scenario_gate(
        pairs, min_scenarios=args.min_scenarios, iterations=args.iterations
    )
    print(f"\nscenario gate (independent unit, min {args.min_scenarios}/flow): {report['decision']}")
    print(f"reason: {report.get('reason')}")
    for flow, row in report.get("flows", {}).items():
        print(
            f"  {flow:<14} scenarios={row['scenarios']} "
            f"baseline={row['baseline_passes']}/{row['scenarios']} "
            f"treatment={row['candidate_passes']}/{row['scenarios']} "
            f"wins={row['mcnemar']['candidate_only']} "
            f"losses={row['mcnemar']['baseline_only']} "
            f"ties={row['mcnemar']['ties']} "
            f"diff={row['difference']['diff']} "
            f"[{row['difference']['low']}, {row['difference']['high']}] "
            f"mcnemar_p={row['mcnemar']['p_value']} "
            f"safety b/t={row['outcomes']['baseline_safety_violations']}/{row['outcomes']['candidate_safety_violations']}"
        )
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(
                {"quick": quick, "scenario_gate": report, "trials": args.trials},
                handle,
                indent=2,
            )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
