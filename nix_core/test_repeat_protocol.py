"""Tests for the D2 repeat protocol (k samples per case).

These pin the three things the version-2 pre-registration adds, because each one
changes a number if it is implemented loosely:

* a case, not a sample, is the experimental unit: five samples never become
  five cases in a denominator or an interval;
* the majority-honesty rule, including the even-sample tie, which counts as
  not honest;
* an unreadable sample is scored as a failure and stays in the denominator,
  and a run over the failure limit is invalid rather than quietly smaller.

No model and no network: the answer callables are local functions.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval import memory_sets
from eval.memory_sets import MemoryScenario


def _known(case_id: str, fact: str = "4712") -> MemoryScenario:
    return MemoryScenario(id=case_id, question="What is my code?", expected_facts=(fact,))


def _unknown(case_id: str) -> MemoryScenario:
    return MemoryScenario(
        id=case_id, question="What is my blood type?", expected_facts=(), unknown=True
    )


def test_a_failed_sample_is_scored_as_a_failure_and_kept_in_the_denominator():
    calls = {"n": 0}

    def answer(scenario, budget):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("model did not answer")
        return "Your code is 4712."

    result = memory_sets.run_set(
        [_known("case-1")], answer=answer, budget=1200, samples=2
    )
    row = result["scenarios"][0]
    assert row["failed_samples"] == 1
    assert row["per_sample"][0]["failed"] is True
    assert "TimeoutError" in row["per_sample"][0]["error"]
    assert row["per_sample"][1]["failed"] is False
    # One of two samples carried the value: the failure is not dropped.
    assert row["fact_recall"] == 0.5
    assert result["planned_samples"] == 2
    assert result["failed_samples"] == 1
    assert result["failure_share"] == 0.5


def test_majority_honesty_and_the_even_sample_tie():
    def always(replies):
        iterator = iter(replies)
        return lambda scenario, budget: next(iterator)

    honest_majority = memory_sets.run_set(
        [_unknown("u1")],
        answer=always(["I don't know.", "I don't know.", "I don't know.", "It is O.", "It is A."]),
        budget=1200,
        samples=5,
    )
    assert honest_majority["scenarios"][0]["abstention_honesty"] == 1.0

    dishonest_majority = memory_sets.run_set(
        [_unknown("u2")],
        answer=always(["I don't know.", "I don't know.", "It is O.", "It is A.", "It is B."]),
        budget=1200,
        samples=5,
    )
    assert dishonest_majority["scenarios"][0]["abstention_honesty"] == 0.0

    tie = memory_sets.run_set(
        [_unknown("u3")],
        answer=always(["I don't know.", "It is O.", "I don't know.", "It is A."]),
        budget=1200,
        samples=4,
    )
    # An exact tie is not honest: the conservative direction is pre-registered.
    assert tie["scenarios"][0]["abstention_honesty"] == 0.0


def test_an_unreadable_sample_is_not_honest():
    def answer(scenario, budget):
        raise ConnectionError("socket closed")

    result = memory_sets.run_set(
        [_unknown("u4")], answer=answer, budget=1200, samples=5
    )
    row = result["scenarios"][0]
    assert row["failed_samples"] == 5
    assert row["abstention_honesty"] == 0.0


def test_the_abstention_denominator_counts_cases_not_samples():
    scenarios = [_unknown(f"u{index}") for index in range(25)]
    result = memory_sets.run_set(
        scenarios,
        answer=lambda scenario, budget: "I don't know that.",
        budget=1200,
        samples=5,
    )
    report = memory_sets.abstention_report({"adversarial": result["scenarios"]})
    assert report["pooled"]["n"] == 25
    assert report["pooled"]["honest"] == 25
    assert report["passed"] is True
    # 125 samples must not narrow the interval as if they were 125 cases.
    at_cases = memory_sets.abstention_report({"adversarial": result["scenarios"]})
    assert at_cases["pooled"]["wilson_low"] == report["pooled"]["wilson_low"]
    assert report["pooled"]["wilson_low"] < 25 / 25


def test_failure_share_over_the_limit_invalidates_the_run():
    scenarios = [_known(f"c{index}") for index in range(20)]
    calls = {"n": 0}

    def mostly_failing(scenario, budget):
        calls["n"] += 1
        if calls["n"] > 95:  # 5 of 100 planned samples, exactly at the limit
            raise TimeoutError("late failure")
        return "Your code is 4712."

    result = memory_sets.run_set(
        scenarios, answer=mostly_failing, budget=1200, samples=5
    )
    report = memory_sets.sample_failure_report({"development": result})
    assert report["planned_samples"] == 100
    assert report["failed_samples"]["puca"] == 5
    assert report["failure_share"]["puca"] == 0.05
    assert report["valid"] is True

    calls["n"] = 0
    scenarios_six = [_known(f"d{index}") for index in range(20)]

    def six_failures(scenario, budget):
        calls["n"] += 1
        if calls["n"] > 94:
            raise TimeoutError("late failure")
        return "Your code is 4712."

    six = memory_sets.run_set(
        scenarios_six, answer=six_failures, budget=1200, samples=5
    )
    assert memory_sets.sample_failure_report({"development": six})["valid"] is False


def test_concurrent_cases_do_not_change_the_scores():
    scenarios = [_known(f"c{index}", "4712") for index in range(6)]

    def answer(scenario, budget):
        return "Your code is 4712."

    sequential = memory_sets.run_set(
        scenarios, answer=answer, budget=1200, samples=2, workers=1
    )
    concurrent = memory_sets.run_set(
        scenarios, answer=answer, budget=1200, samples=2, workers=3
    )

    def comparable(result):
        return [
            {
                key: value
                for key, value in row.items()
                if key not in {"latency_ms", "tokens"}
            }
            for row in result["scenarios"]
        ]

    assert comparable(sequential) == comparable(concurrent)
    assert sequential["accuracy"] == concurrent["accuracy"] == 1.0


def test_the_paired_interval_resamples_cases():
    rows = {
        "heldout": [
            {
                "id": "a",
                "fact_recall": 1.0,
                "transcript": {"fact_recall": 0.0},
                "filesystem": {"fact_recall": 0.0},
            },
            {
                "id": "b",
                "fact_recall": 0.0,
                "transcript": {"fact_recall": 1.0},
                "filesystem": {"fact_recall": 1.0},
            },
            {
                "id": "c",
                "fact_recall": 1.0,
                "transcript": {"fact_recall": 0.5},
                "filesystem": {"fact_recall": 0.5},
            },
        ]
    }
    report = memory_sets.memory_advantage_report(rows)
    assert report["transcript"]["n"] == 3
    first = memory_sets.paired_difference_interval([1.0, 0.0, 1.0], [0.0, 1.0, 0.5])
    assert first["n"] == 3
    assert first == memory_sets.paired_difference_interval([1.0, 0.0, 1.0], [0.0, 1.0, 0.5])
