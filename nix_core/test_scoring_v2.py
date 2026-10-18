"""Tests for the scoring v2 proposal.

Two things must stay true for the proposal to be usable as evidence:

* the frozen v1 rule is untouched, so the published WP2 numbers keep their
  meaning;
* v2 never loses a value v1 counted, so a comparison between the two is
  one-directional and the movement is attributable to the rule.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval import scoring_v2
from eval.memory_sets import MemoryScenario, score_answer

HERE = os.path.dirname(os.path.abspath(__file__))


def _scenario(**overrides) -> MemoryScenario:
    base = {
        "id": "t",
        "question": "Where do I work?",
        "expected_facts": ("Cedar & Co",),
    }
    base.update(overrides)
    return MemoryScenario(**base)


def test_the_alias_case_that_zeroed_a_correct_answer_now_counts():
    """The measured WP6 smoke rows, both arms of the reported issue."""
    scenario = _scenario()
    assert score_answer(scenario, "I work at Cedars & Co.").fact_recall == 0.0
    assert scoring_v2.score_answer_v2(scenario, "I work at Cedars & Co.").fact_recall == 1.0

    spelled_out = "Yes, you live in Tacoma, and yes, you work at Cedar and Co."
    multi = _scenario(expected_facts=("Tacoma", "Cedar & Co"))
    assert score_answer(multi, spelled_out).fact_recall == 0.5
    assert scoring_v2.score_answer_v2(multi, spelled_out).fact_recall == 1.0


def test_a_rendering_of_the_same_value_still_counts():
    """A number said as an ordinal, or zero-padded, is the same number."""
    scenario = _scenario(expected_facts=("October 30",))
    assert scoring_v2.score_answer_v2(scenario, "Your deadline is October 30th.").fact_recall == 1.0

    padded = _scenario(expected_facts=("November 3",))
    assert scoring_v2.score_answer_v2(padded, "It renews on November 03.").fact_recall == 1.0


def test_literal_substring_matches_that_v1_made_are_kept():
    """v2 must not drop a value the frozen rule counted: the bank row where the
    reply runs the branch name into the previous word."""
    scenario = _scenario(expected_facts=("Sellwood",))
    reply = "You bank at theSellwood branch."
    assert score_answer(scenario, reply).fact_recall == 1.0
    assert scoring_v2.score_answer_v2(scenario, reply).fact_recall == 1.0


def test_v2_never_scores_below_v1_on_the_recorded_smoke_run():
    cases_path = os.path.join(
        HERE, "benchmarks", "results", "d2_wp6_smoke_2026-10-18.json"
    )
    set_path = os.path.join(HERE, "eval", "data", "memory_scenarios_confirm.json")
    with open(cases_path, encoding="utf-8") as handle:
        cases = json.load(handle)["cases"]
    with open(set_path, encoding="utf-8") as handle:
        sets = json.load(handle)["sets"]
    index = {
        str(row["id"]): row for rows in sets.values() for row in rows
    }

    rescored = scoring_v2.rescore_cases(cases, index)
    assert rescored["rows"], "the smoke run must join to the frozen set"
    assert all(row["recall_delta"] >= 0 for row in rescored["rows"]), [
        row["id"] for row in rescored["rows"] if row["recall_delta"] < 0
    ]
    summary = scoring_v2.summarize(rescored)
    assert summary["cases_that_moved_down"] == []
    assert summary["mean_recall_delta"] > 0
    assert summary["paired_bootstrap_95_interval_on_mean_delta"][0] > 0


def test_the_unknown_question_rule_is_unchanged_between_versions():
    """The honesty detector is v1's; the ablation moves value matching only."""
    scenario = _scenario(expected_facts=(), unknown=True)
    honest = "I don't know your blood type."
    invented = "It is O negative."
    for reply, expected in ((honest, 1.0), (invented, 0.0)):
        v1 = score_answer(scenario, reply)
        v2 = scoring_v2.score_answer_v2(scenario, reply)
        assert v2.abstention_honesty == v1.abstention_honesty == expected, reply


def test_the_interval_is_seeded_and_case_level():
    deltas = [1.0, 0.0, 0.0, 0.0]
    first = scoring_v2.bootstrap_mean_interval(deltas)
    assert first == scoring_v2.bootstrap_mean_interval(deltas)
    assert first[0] <= 0.25 <= first[1]
