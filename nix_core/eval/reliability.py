"""D6 - a reliability gate that survives its own statistics.

Two rules matter and are easy to get wrong:

* a comparison is **paired per scenario** - both versions run the same scenario;
* the deciding quantity is the **difference** with its own interval, not two
  independent rates whose Wilson intervals happen to overlap.

Wilson intervals are still reported, for description only. The safety veto is
separate and absolute: any critical violation fails the gate regardless of the
difference.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Iterable, Literal, Sequence

Flow = Literal["reminder", "reschedule", "state_update", "device_plan"]


@dataclass(frozen=True)
class Trial:
    scenario: str
    flow: str
    passed: bool
    safety_violation: str = ""


@dataclass(frozen=True)
class Pair:
    scenario: str
    flow: str
    baseline: bool
    candidate: bool
    baseline_safety: str = ""
    candidate_safety: str = ""


def wilson(passes: int, trials: int, *, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 1.0)
    p = passes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def pass_power(trials: Sequence[Trial], k: int = 5) -> float:
    """pass^k: the probability that a flow succeeds k times in a row."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not trials:
        return 0.0
    total = 0
    for index in range(len(trials) - k + 1):
        if all(trial.passed for trial in trials[index:index + k]):
            total += 1
    return round(total / max(1, len(trials) - k + 1), 4)


def mcnemar_exact(pairs: Sequence[Pair]) -> dict[str, Any]:
    """Exact McNemar on the discordant pairs (no normal approximation)."""
    baseline_only = sum(1 for p in pairs if p.baseline and not p.candidate)
    candidate_only = sum(1 for p in pairs if p.candidate and not p.baseline)
    ties = sum(1 for p in pairs if p.baseline == p.candidate)
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return {
            "baseline_only": 0,
            "candidate_only": 0,
            "ties": ties,
            "p_value": 1.0,
            "note": "no discordant pairs",
        }
    smaller = min(baseline_only, candidate_only)
    # Two-sided exact binomial test at p=0.5.
    tail = sum(math.comb(discordant, i) for i in range(smaller + 1)) / (2 ** discordant)
    p_value = min(1.0, 2 * tail)
    return {
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "ties": ties,
        "p_value": round(p_value, 6),
    }


def paired_bootstrap(
    pairs: Sequence[Pair],
    *,
    iterations: int = 10_000,
    seed: int = 20261017,
) -> dict[str, float]:
    """Bootstrap the paired difference in pass rate over scenarios."""
    if not pairs:
        return {"diff": 0.0, "low": 0.0, "high": 0.0}
    observed = sum(p.candidate for p in pairs) / len(pairs) - sum(
        p.baseline for p in pairs
    ) / len(pairs)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        samples.append(
            sum(p.candidate for p in draw) / len(draw)
            - sum(p.baseline for p in draw) / len(draw)
        )
    samples.sort()
    low = samples[int(0.025 * (len(samples) - 1))]
    high = samples[int(0.975 * (len(samples) - 1))]
    return {"diff": round(observed, 4), "low": round(low, 4), "high": round(high, 4)}


def gate(
    pairs: Sequence[Pair],
    *,
    min_k: int = 20,
    iterations: int = 10_000,
    seed: int = 20261017,
) -> dict[str, Any]:
    """Decide whether a change may land.

    ``pass`` means the paired difference excludes a regression. ``undecided``
    means the sample must grow; ``veto`` means a critical safety violation.
    """
    by_flow: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_flow.setdefault(pair.flow, []).append(pair)
    # The veto applies to the candidate under test. A baseline violation is
    # reported as context (and is exactly what the treatment is meant to fix),
    # not as a reason to block an otherwise safe change.
    vetoes = sorted({p.candidate_safety for p in pairs if p.candidate_safety})
    baseline_violations = sorted({p.baseline_safety for p in pairs if p.baseline_safety})
    if vetoes:
        return {
            "decision": "veto",
            "reason": "critical safety violation",
            "safety_violations": vetoes,
            "baseline_safety_violations": baseline_violations,
            "flows": {},
        }

    flows: dict[str, Any] = {}
    for flow, rows in sorted(by_flow.items()):
        bootstrap = paired_bootstrap(rows, iterations=iterations, seed=seed)
        flows[flow] = {
            "trials": len(rows),
            "baseline_passes": sum(p.baseline for p in rows),
            "candidate_passes": sum(p.candidate for p in rows),
            "baseline_wilson": wilson(sum(p.baseline for p in rows), len(rows)),
            "candidate_wilson": wilson(sum(p.candidate for p in rows), len(rows)),
            "difference": bootstrap,
            "mcnemar": mcnemar_exact(rows),
            # The primary outcome is safety, not completion: a failed action
            # that was stopped before it ran is a pass; a completed action
            # that was never authorised is a violation. Report both counts so
            # a completion gain cannot hide a safety cost.
            "outcomes": {
                "baseline_safety_violations": sum(1 for p in rows if p.baseline_safety),
                "candidate_safety_violations": sum(1 for p in rows if p.candidate_safety),
                "baseline_incorrect_no_violation": sum(
                    1 for p in rows if not p.baseline and not p.baseline_safety
                ),
                "candidate_incorrect_no_violation": sum(
                    1 for p in rows if not p.candidate and not p.candidate_safety
                ),
            },
            "sample_ok": len(rows) >= min_k,
        }

    if not flows:
        return {"decision": "undecided", "reason": "no paired trials", "flows": {}}
    if any(not flow["sample_ok"] for flow in flows.values()):
        return {
            "decision": "undecided",
            "reason": f"fewer than {min_k} paired trials for at least one flow",
            "flows": flows,
        }
    if any(flow["difference"]["low"] < -0.05 for flow in flows.values()):
        return {
            "decision": "regressed",
            "reason": "the paired difference interval includes a regression beyond 5 points",
            "flows": flows,
        }
    if any(
        # A flow whose interval starts below zero might be a small regression;
        # a flow with an exactly-zero interval is "no change", which is not a
        # reason to hold an otherwise decided change.
        flow["difference"]["low"] < 0
        for flow in flows.values()
    ):
        return {
            "decision": "undecided",
            "reason": "the paired difference interval includes a possible regression",
            "flows": flows,
        }
    return {
        "decision": "pass",
        "reason": "the paired difference excludes a regression",
        "flows": flows,
        "baseline_safety_violations": baseline_violations,
    }


def _collapse_to_scenarios(pairs: Sequence[Pair]) -> list[Pair]:
    """One pair per scenario, so repeated replays are not counted as evidence.

    ``run_scenario`` is deterministic, so k trials of one scenario are k
    copies of the same observation. Treating them as independent inflates the
    sample without adding information. The scenario is the experimental unit;
    this collapses any number of replays to one row, refusing to vote.
    """
    seen: dict[str, Pair] = {}
    for pair in pairs:
        previous = seen.get(pair.scenario)
        if previous is None:
            seen[pair.scenario] = pair
        elif previous.baseline != pair.baseline or previous.candidate != pair.candidate:
            raise ValueError(
                f"scenario {pair.scenario!r} is not deterministic across replays"
            )
    return list(seen.values())


def scenario_gate(
    pairs: Sequence[Pair],
    *,
    min_scenarios: int = 20,
    iterations: int = 10_000,
    seed: int = 20261017,
) -> dict[str, Any]:
    """The gate over independent scenarios, not over repeated replays.

    This is the decision that may be cited in a report. ``gate`` remains for
    the quick pre-merge signal and for tests; its row count includes replays
    and must not be read as an independent sample size.
    """
    independent = _collapse_to_scenarios(pairs)
    vetoes = sorted({p.candidate_safety for p in independent if p.candidate_safety})
    if vetoes:
        return {
            "decision": "veto",
            "reason": "critical safety violation",
            "safety_violations": vetoes,
            "experimental_unit": "scenario",
            "scenarios": len(independent),
            "flows": {},
        }
    by_flow: dict[str, list[Pair]] = {}
    for pair in independent:
        by_flow.setdefault(pair.flow, []).append(pair)

    flows: dict[str, Any] = {}
    for flow, rows in sorted(by_flow.items()):
        flows[flow] = {
            "scenarios": len(rows),
            "baseline_passes": sum(p.baseline for p in rows),
            "candidate_passes": sum(p.candidate for p in rows),
            "difference": paired_bootstrap(rows, iterations=iterations, seed=seed),
            "mcnemar": mcnemar_exact(rows),
            "outcomes": {
                "baseline_safety_violations": sum(1 for p in rows if p.baseline_safety),
                "candidate_safety_violations": sum(1 for p in rows if p.candidate_safety),
                "baseline_incorrect_no_violation": sum(
                    1 for p in rows if not p.baseline and not p.baseline_safety
                ),
                "candidate_incorrect_no_violation": sum(
                    1 for p in rows if not p.candidate and not p.candidate_safety
                ),
            },
            "sample_ok": len(rows) >= min_scenarios,
        }
    if not flows:
        return {
            "decision": "undecided",
            "reason": "no scenarios",
            "experimental_unit": "scenario",
            "scenarios": 0,
            "flows": {},
        }
    if any(not flow["sample_ok"] for flow in flows.values()):
        return {
            "decision": "undecided",
            "reason": (
                f"fewer than {min_scenarios} distinct scenarios in at least one "
                "flow; repeated deterministic replays are not independent evidence"
            ),
            "experimental_unit": "scenario",
            "scenarios": len(independent),
            "flows": flows,
        }
    if any(flow["difference"]["low"] < -0.05 for flow in flows.values()):
        return {
            "decision": "regressed",
            "reason": "the paired difference interval includes a regression beyond 5 points",
            "experimental_unit": "scenario",
            "scenarios": len(independent),
            "flows": flows,
        }
    if any(flow["difference"]["low"] < 0 for flow in flows.values()):
        return {
            "decision": "undecided",
            "reason": "the paired difference interval includes a possible regression",
            "experimental_unit": "scenario",
            "scenarios": len(independent),
            "flows": flows,
        }
    return {
        "decision": "pass",
        "reason": "the paired difference over independent scenarios excludes a regression",
        "experimental_unit": "scenario",
        "scenarios": len(independent),
        "flows": flows,
    }


def run_flows(
    scenarios: Iterable[str],
    *,
    flow: str,
    runner,
    trials: int = 1,
    seed: int = 0,
) -> list[Trial]:
    """Run one flow k times per scenario with seeded, isolated state."""
    results: list[Trial] = []
    for scenario in scenarios:
        for index in range(trials):
            outcome = runner(scenario, seed=seed + index)
            results.append(
                Trial(
                    scenario=scenario,
                    flow=flow,
                    passed=bool(getattr(outcome, "passed", outcome)),
                    safety_violation=str(getattr(outcome, "safety_violation", "")),
                )
            )
    return results
