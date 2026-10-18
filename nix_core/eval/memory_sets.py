"""D2 - the memory evaluation: development, held-out and adversarial sets.

The point is not a leaderboard. It is a check that PUCA's memory beats two
honest baselines on PUCA's own tasks, at the *same* token budget and with the
same model and framing:

* the last N turns of transcript stuffed into context;
* a plain text dump of the records.

The final set is frozen before a reported run. ``frozen_manifest`` hashes the
scenarios, answer keys, scoring rules and baseline implementations; a result note
names that tag and states whether the set was looked at before.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from eval.acceptance import wilson_interval

SetName = Literal["development", "heldout", "adversarial"]
SCORING_RULES = {
    "fact_recall": "expected facts present in the answer, one point each, divided by expected count",
    "temporal_correctness": "the answer states the expected timeframe; 1.0 or 0.0",
    "stale_fact_rate": "a superseded value is reported as current; lower is better",
    "abstention_honesty": "for an unknown question, saying 'I don't know' scores 1.0; inventing scores 0.0",
}

#: The owner-approved abstention rule: of the 25 unknown questions pooled across
#: sets, at least 23 must be honest and the 95% Wilson lower bound must be at
#: least 0.75. Per-set rows are descriptive and cannot override the pooled rule.
ABSTENTION_MIN_HONEST = 23
ABSTENTION_MIN_N = 25
ABSTENTION_MIN_LOWER = 0.75

#: The repeat protocol's missing-run rule: an unreadable sample is scored as a
#: failure, and a run whose failures exceed this share of the planned samples in
#: any arm is invalid and must be repeated in full. Fixed in advance so a bad
#: run cannot be reinterpreted after the numbers are seen.
FAILURE_SHARE_LIMIT = 0.05


@dataclass(frozen=True)
class MemoryScenario:
    id: str
    question: str
    expected_facts: tuple[str, ...]
    expected_timeframe: str = ""
    current_facts: tuple[str, ...] = ()
    superseded_facts: tuple[str, ...] = ()
    permitted_record_ids: tuple[str, ...] = ()
    unknown: bool = False
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "expected_facts": list(self.expected_facts),
            "expected_timeframe": self.expected_timeframe,
            "current_facts": list(self.current_facts),
            "superseded_facts": list(self.superseded_facts),
            "permitted_record_ids": list(self.permitted_record_ids),
            "unknown": self.unknown,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Score:
    fact_recall: float
    temporal_correctness: float
    stale_fact: bool
    abstention_honesty: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fact_recall": round(self.fact_recall, 3),
            "temporal_correctness": round(self.temporal_correctness, 3),
            "stale_fact": self.stale_fact,
            "abstention_honesty": None if self.abstention_honesty is None else round(self.abstention_honesty, 3),
        }


def estimate_tokens(text: str) -> int:
    return max(0, max(len(re.findall(r"\S+", text or "")), (len(text or "") + 3) // 4))


def fit_to_budget(text: str, budget: int) -> str:
    """Cut a baseline's context to the same budget, keeping the newest text."""
    if estimate_tokens(text) <= budget:
        return text
    # Keep the tail: a transcript's newest turns are the useful part. The
    # estimator uses the larger of word count and characters/4, so trim on the
    # estimator itself rather than assuming one token per word.
    candidate = (text or "")[-(budget * 4):]
    while candidate and estimate_tokens(candidate) > budget:
        candidate = candidate[1:]
    return candidate.lstrip()


def transcript_baseline(history: Sequence[Mapping[str, Any]], *, budget: int) -> str:
    lines = [
        f"{turn.get('role', 'user')}: {turn.get('content', '')}"
        for turn in history
        if turn.get("role") in {"user", "assistant"} and turn.get("content")
    ]
    return fit_to_budget("\n".join(lines), budget)


def filesystem_baseline(records: Iterable[Mapping[str, Any]], *, budget: int) -> str:
    lines = [
        f"{record.get('id', '?')}: {json.dumps(record.get('data', record), ensure_ascii=False, sort_keys=True)}"
        for record in records
    ]
    return fit_to_budget("\n".join(lines), budget)


def score_answer(scenario: MemoryScenario, answer: str) -> Score:
    text = " ".join((answer or "").split())
    folded = text.casefold()
    facts = [fact for fact in scenario.expected_facts if fact and fact.casefold() in folded]
    recall = len(facts) / len(scenario.expected_facts) if scenario.expected_facts else 1.0
    timeframe = (
        scenario.expected_timeframe.casefold() in folded
        if scenario.expected_timeframe else True
    )
    stale = any(fact.casefold() in folded for fact in scenario.superseded_facts)
    if scenario.unknown:
        honesty = 1.0 if re.search(
            r"\b(?:i\s+don'?t\s+know|i\s+do\s+not\s+know|"
            r"i\s+don'?t\s+have|i\s+do\s+not\s+have|"
            r"don'?t\s+have\s+(?:that|any|a|it)|do\s+not\s+have\s+(?:that|any|a|it)|"
            r"haven'?t\s+(?:got|stored|learned|heard)|"
            r"not\s+sure|no\s+(?:record|information|memory|data)|"
            r"couldn'?t\s+find|could\s+not\s+find|unknown)\b", folded
        ) else 0.0
    else:
        honesty = None
    return Score(
        fact_recall=recall,
        temporal_correctness=1.0 if timeframe else 0.0,
        stale_fact=stale,
        abstention_honesty=honesty,
    )


def frozen_manifest(
    scenarios: Mapping[str, Sequence[MemoryScenario]],
    *,
    tag: str,
    model: str,
    hardware: str,
    prior_exposure: str = "none",
    retired: Sequence[str] = (),
    baseline_mode: str = "console",
    baseline_model: str = "",
    samples: int = 1,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tag": tag,
        "model": model,
        "hardware": hardware,
        "baseline_mode": baseline_mode,
        "scoring": SCORING_RULES,
        "sets": {
            name: [scenario.as_dict() for scenario in rows]
            for name, rows in scenarios.items()
        },
        "baselines": ["full_recent_transcript", "filesystem_dump"],
    }
    # Only recorded when it differs from the console's model id, so an older
    # freeze that did not need it keeps the hash it was reported with.
    if baseline_model:
        payload["baseline_model"] = baseline_model
    if samples > 1:
        # The repeat protocol is part of the frozen design: five samples per
        # case is a pre-registered decision, not a run-time choice.
        payload["samples_per_case"] = samples
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        **payload,
        "sha256": digest,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "prior_exposure": prior_exposure,
        "retired": list(retired),
    }


def _attempt(call) -> tuple[str, str]:
    """Run one arm's call, turning a transport failure into a scored failure.

    The repeat protocol is explicit about this: a sample that cannot be read is
    not dropped, not imputed and not retried. It scores as a failure and the
    case is flagged degraded, because a selectively repeated sample would be a
    different (easier) sample.
    """
    try:
        return call(), ""
    except Exception as exc:  # noqa: BLE001 - the error text is recorded
        return "", f"{type(exc).__name__}: {exc}"


def run_set(
    scenarios: Sequence[MemoryScenario],
    *,
    answer,
    budget: int,
    transcripts: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    records: Sequence[Mapping[str, Any]] | Callable[[MemoryScenario], Sequence[Mapping[str, Any]]] = (),
    baseline_answer=None,
    samples: int = 1,
    workers: int = 1,
) -> dict[str, Any]:
    """Run one set and return per-scenario and aggregate rows.

    Every arm is timed, so the table reports latency next to accuracy: a memory
    design that wins on facts but costs seconds per turn is not obviously the
    one to keep.

    ``samples`` asks each question that many independent times. The per-case
    row keeps every sample and its spread, and the aggregate averages the
    *cases* — five samples of one question are one cluster, not five cases, so
    they must not inflate n or narrow any interval (see the repeat protocol in
    the approved proposal).

    ``workers`` runs that many *cases* concurrently. It is a wall-clock setting
    only: a case's own samples stay sequential and independent either way, so
    the scores must not depend on it (demonstrated by
    ``test_repeat_protocol.py``).
    """
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    def score_case(scenario: MemoryScenario) -> dict[str, Any]:
        sample_rows = []
        latencies = []
        failures: list[dict[str, Any]] = []
        tokens = 0
        for sample_index in range(samples):
            started = time.perf_counter()
            answer_text, error = _attempt(lambda: answer(scenario, budget))
            latencies.append((time.perf_counter() - started) * 1000)
            tokens += estimate_tokens(answer_text)
            score = score_answer(scenario, answer_text).as_dict()
            score["failed"] = bool(error)
            if error:
                score["error"] = error
                failures.append({"sample": sample_index, "error": error})
            sample_rows.append(score)
        row = {
            "id": scenario.id,
            "samples": samples,
            "tokens": round(tokens / samples),
            "latency_ms": round(sum(latencies) / len(latencies), 1),
            "failed_samples": len(failures),
            "sample_failures": failures,
            **_mean_scores(sample_rows),
        }
        if samples > 1:
            row["per_sample"] = sample_rows
        if baseline_answer is not None:
            transcript = transcript_baseline(
                (transcripts or {}).get(scenario.id, ()), budget=budget
            )
            scenario_records = records(scenario) if callable(records) else records
            dump = filesystem_baseline(scenario_records, budget=budget)
            baseline_rows = {}
            for arm, context in (("transcript", transcript), ("filesystem", dump)):
                arm_scores = []
                arm_latencies = []
                arm_failures: list[dict[str, Any]] = []
                for sample_index in range(samples):
                    baseline_started = time.perf_counter()
                    baseline_text, error = _attempt(
                        lambda: baseline_answer(scenario, context, budget)
                    )
                    arm_latencies.append(
                        (time.perf_counter() - baseline_started) * 1000
                    )
                    score = score_answer(scenario, baseline_text).as_dict()
                    score["failed"] = bool(error)
                    if error:
                        score["error"] = error
                        arm_failures.append({"sample": sample_index, "error": error})
                    arm_scores.append(score)
                baseline_rows[arm] = {
                    **_mean_scores(arm_scores),
                    "latency_ms": round(sum(arm_latencies) / len(arm_latencies), 1),
                    "failed_samples": len(arm_failures),
                    "sample_failures": arm_failures,
                }
                if samples > 1:
                    baseline_rows[arm]["per_sample"] = arm_scores
            row.update(baseline_rows)
        return row

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(score_case, scenarios))
    else:
        rows = [score_case(scenario) for scenario in scenarios]
    count = len(rows)
    planned = count * samples
    failed = sum(int(row.get("failed_samples") or 0) for row in rows)
    baseline_failed = {
        arm: sum(int(row.get(arm, {}).get("failed_samples") or 0) for row in rows)
        for arm in ("transcript", "filesystem")
        if count and arm in rows[0]
    }
    return {
        "scenarios": rows,
        "planned_samples": planned,
        "failed_samples": failed,
        "failure_share": round(failed / planned, 4) if planned else 0.0,
        "baseline_failed_samples": baseline_failed,
        "baseline_failure_share": {
            arm: round(value / planned, 4) if planned else 0.0
            for arm, value in baseline_failed.items()
        },
        "accuracy": round(sum(row["fact_recall"] for row in rows) / count, 3) if count else 0.0,
        "temporal_correctness": round(sum(row["temporal_correctness"] for row in rows) / count, 3) if count else 0.0,
        "stale_fact_rate": round(sum(1 for row in rows if row["stale_fact"]) / count, 3) if count else 0.0,
        "abstention_honesty": round(
            sum(row["abstention_honesty"] for row in rows if row["abstention_honesty"] is not None)
            / max(1, sum(1 for row in rows if row["abstention_honesty"] is not None)),
            3,
        ),
        "tokens": sum(row["tokens"] for row in rows),
        "budget": budget,
        "scenarios_count": count,
        "samples_per_case": max((row.get("samples") or 1) for row in rows) if rows else 1,
        "latency_ms_p50": _percentile([row["latency_ms"] for row in rows], 0.50),
        "baselines": {
            arm: {
                "accuracy": round(
                    sum(row[arm]["fact_recall"] for row in rows) / count, 3
                ) if count else 0.0,
                "temporal_correctness": round(
                    sum(row[arm]["temporal_correctness"] for row in rows) / count, 3
                ) if count else 0.0,
            }
            for arm in ("transcript", "filesystem")
            if rows and arm in rows[0]
        },
    }


def sample_failure_report(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The pre-registered missing-run check, per arm and pooled.

    Denominators are planned samples, not returned ones: a failed call stays in
    the denominator, so a run that lost calls cannot look cleaner than it was.
    Any arm above ``FAILURE_SHARE_LIMIT`` makes the whole run invalid; there is
    no partial-run substitute and no selective retry.
    """
    planned = sum(int(result.get("planned_samples") or 0) for result in results.values())
    failed = {"puca": sum(int(result.get("failed_samples") or 0) for result in results.values())}
    for result in results.values():
        for arm, count in (result.get("baseline_failed_samples") or {}).items():
            failed[arm] = failed.get(arm, 0) + int(count)
    shares = {
        arm: round(count / planned, 4) if planned else 0.0
        for arm, count in failed.items()
    }
    degraded_cases = [
        row["id"]
        for result in results.values()
        for row in result.get("scenarios") or []
        if int(row.get("failed_samples") or 0) > 0
    ]
    return {
        "rule": {
            "failed_sample_scores_as_failure": True,
            "retries": "none; a full rerun is the only remedy",
            "max_failure_share_per_arm": FAILURE_SHARE_LIMIT,
        },
        "planned_samples": planned,
        "failed_samples": failed,
        "failure_share": shares,
        "degraded_cases": degraded_cases,
        "valid": bool(planned) and all(share <= FAILURE_SHARE_LIMIT for share in shares.values()),
    }


def abstention_report(rows_by_set: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Pooled abstention honesty, with the pre-registered rule applied.

    Only rows that carry an abstention score (unknown questions) count. The
    decision is computed on the pooled 25-question population; the per-set
    rows sit next to it and are descriptive.
    """
    pooled = [
        row
        for rows in rows_by_set.values()
        for row in rows
        if row.get("abstention_honesty") is not None
    ]
    honest = sum(1 for row in pooled if float(row["abstention_honesty"]) >= 0.5)
    n = len(pooled)
    low, high = wilson_interval(honest, n)
    passed = n >= ABSTENTION_MIN_N and honest >= ABSTENTION_MIN_HONEST and low >= ABSTENTION_MIN_LOWER
    per_set = {
        name: {
            "n": sum(1 for row in rows if row.get("abstention_honesty") is not None),
            "honest": sum(
                1
                for row in rows
                if row.get("abstention_honesty") is not None
                and float(row["abstention_honesty"]) >= 0.5
            ),
        }
        for name, rows in rows_by_set.items()
    }
    for name, block in per_set.items():
        block["rate"] = round(block["honest"] / block["n"], 3) if block["n"] else None
    return {
        "rule": {
            "min_honest": ABSTENTION_MIN_HONEST,
            "min_n": ABSTENTION_MIN_N,
            "min_wilson_lower": ABSTENTION_MIN_LOWER,
        },
        "pooled": {
            "n": n,
            "honest": honest,
            "rate": round(honest / n, 3) if n else None,
            "wilson_low": round(low, 4),
            "wilson_high": round(high, 4),
        },
        "per_set": per_set,
        "passed": bool(passed),
    }


def paired_difference_interval(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 20261018,
) -> dict[str, Any]:
    """Percentile bootstrap interval on the paired mean difference.

    The comparison is paired on identical scenarios, as the gate requires. The
    seed is fixed so the same result file cannot produce two different
    intervals on a re-read.
    """
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("paired arms need the same, non-zero number of rows")
    differences = [float(a) - float(b) for a, b in zip(candidate, baseline)]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = rng.choices(differences, k=len(differences))
        means.append(sum(draw) / len(draw))
    means.sort()
    low = means[int(0.025 * samples)]
    high = means[min(samples - 1, int(0.975 * samples))]
    return {
        "n": len(differences),
        "difference": round(sum(differences) / len(differences), 4),
        "low": round(low, 4),
        "high": round(high, 4),
    }


def memory_advantage_report(rows_by_set: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """PUCA's fact recall against each baseline, paired per scenario.

    A positive memory claim requires the interval to exclude zero against
    *both* baselines, per the approved proposal.
    """
    rows = [row for rows in rows_by_set.values() for row in rows]
    report: dict[str, Any] = {}
    for arm in ("transcript", "filesystem"):
        paired = [row for row in rows if arm in row]
        interval = paired_difference_interval(
            [row["fact_recall"] for row in paired],
            [row[arm]["fact_recall"] for row in paired],
        )
        report[arm] = {**interval, "demonstrated": interval["low"] > 0}
    report["demonstrated"] = all(report[arm]["demonstrated"] for arm in ("transcript", "filesystem"))
    return report


def _mean_scores(sample_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average one case's samples into one case-level row.

    The abstention flag keeps the majority rule: a case whose samples disagree
    is honest only when most of them are, and the case still contributes one
    observation to the interval.
    """
    count = max(1, len(sample_rows))

    def mean(key: str) -> float:
        return round(sum(float(row.get(key) or 0.0) for row in sample_rows) / count, 3)

    recalls = [float(row["fact_recall"]) for row in sample_rows]
    average_recall = sum(recalls) / len(recalls)
    stdev = (sum((value - average_recall) ** 2 for value in recalls) / len(recalls)) ** 0.5

    honesty_values = [
        float(row["abstention_honesty"])
        for row in sample_rows
        if row.get("abstention_honesty") is not None
    ]
    honesty = (
        1.0 if honesty_values and sum(1 for value in honesty_values if value >= 0.5) * 2 > len(honesty_values)
        else (0.0 if honesty_values else None)
    )
    return {
        "fact_recall": mean("fact_recall"),
        "temporal_correctness": mean("temporal_correctness"),
        "stale_fact": any(bool(row.get("stale_fact")) for row in sample_rows),
        "abstention_honesty": honesty,
        "fact_recall_stdev": round(stdev, 3),
        "fact_recall_samples": [round(value, 3) for value in recalls],
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return round(ordered[index], 1)


def format_table(results: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        f"{'set':<14}{'n':>4}{'accuracy':>10}{'temporal':>10}{'stale':>8}"
        f"{'abstain':>9}{'tokens':>9}{'p50 ms':>9}{'transcript':>12}{'filesystem':>12}",
    ]
    for name, result in results.items():
        baselines = result.get("baselines") or {}
        lines.append(
            f"{name:<14}{result['scenarios_count']:>4}"
            f"{result['accuracy']:>10.3f}{result['temporal_correctness']:>10.3f}"
            f"{result['stale_fact_rate']:>8.3f}{result['abstention_honesty']:>9.3f}"
            f"{result['tokens']:>9}"
            f"{(result.get('latency_ms_p50') or 0):>9.1f}"
            f"{baselines.get('transcript', {}).get('accuracy', 0.0):>12.3f}"
            f"{baselines.get('filesystem', {}).get('accuracy', 0.0):>12.3f}"
        )
    return "\n".join(lines)


def load_sets(path: str | Path) -> dict[str, list[MemoryScenario]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sets: dict[str, list[MemoryScenario]] = {}
    for name, rows in payload.get("sets", {}).items():
        sets[name] = [
            MemoryScenario(
                id=str(row["id"]),
                question=str(row["question"]),
                expected_facts=tuple(row.get("expected_facts") or ()),
                expected_timeframe=str(row.get("expected_timeframe") or ""),
                current_facts=tuple(row.get("current_facts") or ()),
                superseded_facts=tuple(row.get("superseded_facts") or ()),
                permitted_record_ids=tuple(row.get("permitted_record_ids") or ()),
                unknown=bool(row.get("unknown")),
                notes=str(row.get("notes") or ""),
            )
            for row in rows
        ]
    return sets
