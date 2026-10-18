"""The proposal's acceptance criteria, as data instead of prose.

Revision 4 of the Luna proposal had one criterion in §7 and a different one in
§8.5. They are now the same rule, and the rule lives here so the document and
the tests cannot drift apart:

* a **rate** passes when the upper bound of its 95% Wilson interval is below
  the target (false-tool < 2%, missed-tool < 5%);
* a **comparison** is non-inferior when the lower bound of the 95% paired
  interval on the difference is above the margin (−5 points);
* **non-inferiority is necessary, not sufficient for promotion**: promotion
  requires the stricter condition that the interval excludes *any* regression
  (lower bound at least 0). A feature that might be worse stays in shadow;
* **inconclusive is not a pass**: a feature whose gate is unmet stays in shadow.

Nothing here measures anything. It answers "how many observations would this
claim need" and "does this count pass", so an evaluation can be sized before it
runs rather than interpreted after.
"""
from __future__ import annotations

import math
from typing import Any

Z_95 = 1.96
#: One-sided 2.5% for a non-inferiority comparison, 80% power.
Z_ALPHA_ONE_SIDED = 1.96
Z_POWER_80 = 0.8416

#: The pre-registered targets, by rate name.
RATE_TARGETS: dict[str, float] = {
    "false_tool": 0.02,
    "missed_tool": 0.05,
}

#: The non-inferiority margin on a paired difference. A negative number: the
#: comparison fails when the interval's lower bound falls below it.
NON_INFERIORITY_MARGIN = -0.05

#: Fewest distinct scenarios per flow before any comparison may be reported.
MIN_SCENARIOS_PER_FLOW = 20


def wilson_interval(successes: int, n: int, *, z: float = Z_95) -> tuple[float, float]:
    """Two-sided Wilson interval for ``successes`` of ``n``.

    Unlike a Wald interval it stays inside [0, 1] and is defined when every
    observation is a success, which is exactly the shape of a 25-question
    abstention check.
    """
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def wilson_upper(errors: int, n: int, *, z: float = Z_95) -> float:
    """Upper bound of the 95% Wilson interval for ``errors`` of ``n``."""
    if n <= 0:
        return 1.0
    return wilson_interval(errors, n, z=z)[1]


def wilson_lower(successes: int, n: int, *, z: float = Z_95) -> float:
    """Lower bound of the 95% Wilson interval for ``successes`` of ``n``."""
    if n <= 0:
        return 0.0
    return wilson_interval(successes, n, z=z)[0]


def passes_rate(errors: int, n: int, rate: str) -> bool:
    """True when the interval upper bound is below the target for this rate."""
    if rate not in RATE_TARGETS:
        raise ValueError(f"unknown rate: {rate}")
    if errors < 0 or n <= 0 or errors > n:
        raise ValueError("errors must be between 0 and n")
    return wilson_upper(errors, n) < RATE_TARGETS[rate]


def permitted_errors(n: int, rate: str) -> int:
    """Most errors ``n`` observations may contain and still pass.

    Returns ``-1`` when even zero errors cannot pass at this n, which is the
    signal that the sample is too small for the claim rather than that the
    system failed.
    """
    best = -1
    for errors in range(0, n + 1):
        if passes_rate(errors, n, rate):
            best = errors
        else:
            break
    return best


def required_n(rate: str, *, z: float = Z_95) -> int:
    """Smallest n at which zero errors passes ``rate``."""
    if rate not in RATE_TARGETS:
        raise ValueError(f"unknown rate: {rate}")
    n = 1
    while wilson_upper(0, n, z=z) >= RATE_TARGETS[rate]:
        n += 1
    return n


def pairs_needed(
    discordance: float,
    delta: float,
    *,
    margin: float = NON_INFERIORITY_MARGIN,
    alpha_z: float = Z_ALPHA_ONE_SIDED,
    power_z: float = Z_POWER_80,
) -> int:
    """Pairs for a paired non-inferiority comparison.

    ``discordance`` is the expected share of pairs where the two arms disagree;
    ``delta`` is the expected true difference. The margin shifts the null, so
    the effective difference tested is ``delta - margin``.
    """
    if not 0 < discordance <= 1:
        raise ValueError("discordance must be in (0, 1]")
    effective = delta - margin
    if effective <= 0:
        raise ValueError("delta must exceed the non-inferiority margin")
    numerator = (
        alpha_z * math.sqrt(discordance)
        + power_z * math.sqrt(max(discordance - delta * delta, 1e-9))
    ) ** 2
    return math.ceil(numerator / (effective * effective))


def decide_comparison(low: float, high: float, scenarios: int) -> dict[str, Any]:
    """The executable rule for a paired comparison.

    Two thresholds are deliberate and are not the same test:

    * ``low < NON_INFERIORITY_MARGIN`` is a regression beyond −5 points;
    * ``low < 0`` is inconclusive for promotion. The feature stays in shadow
      even though it may be non-inferior, because the interval still contains
      a possible regression.

    Promoting on non-inferiority alone would let a feature that might be
    worse go live. Promotion therefore requires ``low >= 0``.
    """
    if low > high:
        raise ValueError("low must not exceed high")
    if scenarios < MIN_SCENARIOS_PER_FLOW:
        return {
            "decision": "undecided",
            "action": "hold_in_shadow",
            "reason": f"fewer than {MIN_SCENARIOS_PER_FLOW} distinct scenarios",
        }
    if low < NON_INFERIORITY_MARGIN:
        return {
            "decision": "regressed",
            "action": "rollback_to_off",
            "reason": "the paired interval includes a regression beyond the margin",
        }
    if low < 0:
        return {
            "decision": "undecided",
            "action": "hold_in_shadow",
            "reason": "the paired interval includes a possible regression",
        }
    return {
        "decision": "pass",
        "action": "promote",
        "reason": "the paired interval excludes a regression",
    }


def decide_feature(
    *,
    low: float,
    high: float,
    scenarios: int,
    safety_violations: int,
    latency_p95_ms: float,
    latency_budget_ms: float,
    internal_vocabulary: int,
) -> dict[str, Any]:
    """The full promotion/rollback gate for a feature behind a flag.

    Every input is required: a gate that can pass while a safety count, a
    latency measurement or a vocabulary count is missing is not a gate. The
    order is the proposal's order — safety and vocabulary vetoes first, then
    the statistical comparison, then the latency budget.
    """
    if safety_violations < 0:
        raise ValueError("safety_violations must be non-negative")
    if internal_vocabulary < 0:
        raise ValueError("internal_vocabulary must be non-negative")
    if latency_p95_ms < 0 or latency_budget_ms < 0:
        raise ValueError("latency values must be non-negative")

    if safety_violations:
        return {
            "decision": "veto",
            "action": "rollback_to_off",
            "reason": "a candidate unauthorized action is a hard veto",
            "safety_violations": safety_violations,
        }
    if internal_vocabulary:
        return {
            "decision": "regressed",
            "action": "rollback_to_off",
            "reason": "internal vocabulary reached the reply",
            "internal_vocabulary": internal_vocabulary,
        }

    comparison = decide_comparison(low, high, scenarios)
    result = dict(comparison)
    result.update(
        {
            "safety_violations": safety_violations,
            "internal_vocabulary": internal_vocabulary,
            "latency_p95_ms": latency_p95_ms,
            "latency_budget_ms": latency_budget_ms,
        }
    )
    if result["action"] == "promote" and latency_p95_ms > latency_budget_ms:
        result.update(
            {
                "decision": "undecided",
                "action": "hold_in_shadow",
                "reason": "p95 latency is above the pre-registered budget",
            }
        )
    return result


def rate_report(errors: int, n: int, rate: str) -> dict[str, Any]:
    """Everything a report needs for one rate, including the sample check."""
    allowed = permitted_errors(n, rate)
    return {
        "rate": rate,
        "target": RATE_TARGETS[rate],
        "errors": errors,
        "n": n,
        "wilson_upper": round(wilson_upper(errors, n), 4),
        "permitted_errors": allowed,
        "required_n": required_n(rate),
        "passes": passes_rate(errors, n, rate),
        "sample_sufficient": allowed >= 0,
    }
