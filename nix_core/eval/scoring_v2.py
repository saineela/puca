"""Scoring v2 - a versioned proposal, not a replacement for the frozen rule.

**The problem it addresses.** The D2 scoring rule (``eval/memory_sets.py``
``score_answer``) counts an expected value when its literal characters appear in
the answer, after case folding. Two recorded replies in the WP6 smoke run show
what that costs:

* stored/expected ``Cedar & Co`` - reply ``I work at Cedars & Co.`` -> 0.0,
  because the apostrophe-free form the user said ("Cedars") is one letter away
  from the form the set recorded ("Cedar");
* expected ``Cedar & Co`` - reply ``yes, you work at Cedar and Co.`` -> 0.0,
  because the reply spells out the ampersand.

Both replies carry the fact. The rule, not the system, answered zero.

**Why a new version instead of an edit.** The WP2 confirmation run and the
frozen baseline in ``benchmarks/results/d2_memory_confirm_2026-10-18.json`` were
scored with v1. Changing v1 in place would silently change the meaning of a
number that is already published, and any later comparison against it would be
against a rule nobody recorded. So the v1 function is left exactly as it is
(this module imports it and never re-implements it), and v2 lives beside it.

**The fair comparison method.** Both rules are applied to the *same* recorded
replies, in one pass:

    python3 benchmarks/rescore_scoring_v2.py \\
        --cases benchmarks/results/d2_wp6_smoke_2026-10-18.json \\
        --set eval/data/memory_scenarios_confirm.json \\
        --json benchmarks/results/d2_scoring_v2_comparison_<date>.json

The output keeps per-case v1 and v2 scores, the delta, the count of cases that
moved, the mean recall under each rule, and a paired bootstrap interval on the
mean delta over cases (seeded, so it is reproducible). It never overwrites the
frozen result: the baseline number stays the published one, and the v2 number
is reported beside it with the rule that produced it.

**What v2 changes, and what it deliberately does not.** Only value matching
changes: text is normalised (NFKC, ampersand -> "and", punctuation dropped,
whitespace collapsed) and matched token-by-token with a conservative
plural/singular fold. Timeframe matching uses the same normalisation. The
abstention rule is copied verbatim from v1 so the ablation isolates one cause -
"does the value the user said count as the value that was recorded" - and does
not quietly move the honesty gate too.
"""
from __future__ import annotations

import random
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from eval.memory_sets import MemoryScenario, Score, score_answer

#: The version string recorded next to every v2 number.
VERSION = "scoring-v2-2026-10-18"

#: Version of the rule the frozen WP2 results were scored with.
BASELINE_VERSION = "scoring-v1-frozen-2026-10-18"

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})


def normalize(text: str) -> str:
    """The comparison form of a value: spelling kept, punctuation folded.

    ``Cedars & Co.`` -> ``cedars and co``; ``Cedar & Co`` -> ``cedar and co``.
    The two still differ in the plural, which ``tokens`` folds. Ordinals and
    zero padding are folded too, because the recorded reply is a *rendering* of
    the stored value ("October 30th" for a fact set to "October 30").
    """
    folded = unicodedata.normalize("NFKC", str(text or "")).translate(_APOSTROPHES)
    folded = folded.casefold().replace("&", " and ")
    folded = re.sub(r"[^a-z0-9'\-\s]", " ", folded)
    # "30th" -> "30", "03" -> "3": the same number, said differently.
    folded = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", folded)
    folded = re.sub(r"\b0+(\d+)\b", r"\1", folded)
    return " ".join(folded.split())


def _singular(word: str) -> str:
    """A conservative plural fold: only a trailing s/es on a longer word."""
    if len(word) > 3 and word.endswith("es") and not word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokens(text: str) -> list[str]:
    """Normalised tokens with the plural fold applied to each one."""
    return [_singular(token) for token in _TOKEN_RE.findall(normalize(text))]


def contains(haystack: str, needle: str) -> bool:
    """True when the value is present in the answer under v2 normalisation.

    Two tests, and the second is deliberate: a token run in order (so ``Co
    Cedar`` is not ``Cedar Co``), or the normalised substring (so a value that
    the recorded reply runs into neighbouring letters still counts, the way v1
    counted it - measured: ``You bank at theSellwood branch.``). v2 therefore
    never loses a match v1 made; it can only add one, which is what makes the
    comparison one-directional and readable.
    """
    wanted = tokens(needle)
    if not wanted:
        return False
    have = tokens(haystack)
    if have:
        for start in range(len(have) - len(wanted) + 1):
            if have[start : start + len(wanted)] == wanted:
                return True
    return normalize(needle) in normalize(haystack)


def score_answer_v2(scenario: MemoryScenario, answer: str) -> Score:
    """v1's metrics computed with v2 value matching.

    The abstention test is v1's, verbatim, through ``score_answer`` on a
    scenario whose values are rewritten into the normalised form. That keeps the
    honesty rule identical while the values it is compared against change.
    """
    folded = " ".join((answer or "").split())
    matched = [fact for fact in scenario.expected_facts if contains(folded, fact)]
    recall = len(matched) / len(scenario.expected_facts) if scenario.expected_facts else 1.0
    timeframe = contains(folded, scenario.expected_timeframe) if scenario.expected_timeframe else True
    stale = any(contains(folded, fact) for fact in scenario.superseded_facts)
    if scenario.unknown:
        # The honesty rule is v1's: the same detector, through the same
        # function, with only the values replaced by the normalised forms.
        honesty = score_answer(scenario, answer).abstention_honesty
    else:
        honesty = None
    return Score(
        fact_recall=recall,
        temporal_correctness=1.0 if timeframe else 0.0,
        stale_fact=stale,
        abstention_honesty=honesty,
    )


def rescore_cases(
    cases: Sequence[Mapping[str, Any]],
    scenarios: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Dual-score recorded replies with v1 and v2, case by case.

    ``cases`` is a result artifact's ``cases`` list (each with ``id``,
    ``reply``); ``scenarios`` maps an id to the frozen scenario dict. Only a
    case present in both is scored, and every case is reported so the
    denominator is visible.
    """
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for case in cases:
        case_id = str(case.get("id") or "")
        spec = scenarios.get(case_id)
        if spec is None:
            continue
        scenario = MemoryScenario(
            id=case_id,
            question=str(spec.get("question") or ""),
            expected_facts=tuple(str(v) for v in spec.get("expected_facts") or ()),
            expected_timeframe=str(spec.get("expected_timeframe") or ""),
            current_facts=tuple(str(v) for v in spec.get("current_facts") or ()),
            superseded_facts=tuple(str(v) for v in spec.get("superseded_facts") or ()),
            permitted_record_ids=tuple(
                str(v) for v in spec.get("permitted_record_ids") or ()
            ),
            unknown=bool(spec.get("unknown")),
            notes=str(spec.get("notes") or ""),
        )
        reply = str(case.get("reply") or "")
        v1 = score_answer(scenario, reply)
        v2 = score_answer_v2(scenario, reply)
        delta = round(v2.fact_recall - v1.fact_recall, 6)
        deltas.append(delta)
        rows.append(
            {
                "id": case_id,
                "set": case.get("set"),
                "unknown": bool(scenario.unknown),
                "expected_facts": list(scenario.expected_facts),
                "reply": reply,
                "v1": {
                    "fact_recall": v1.fact_recall,
                    "temporal_correctness": v1.temporal_correctness,
                    "stale_fact": v1.stale_fact,
                    "abstention_honesty": v1.abstention_honesty,
                    "scorer": BASELINE_VERSION,
                },
                "v2": {
                    "fact_recall": v2.fact_recall,
                    "temporal_correctness": v2.temporal_correctness,
                    "stale_fact": v2.stale_fact,
                    "abstention_honesty": v2.abstention_honesty,
                    "scorer": VERSION,
                },
                "recall_delta": delta,
            }
        )
    return {
        "rows": rows,
        "deltas": deltas,
    }


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    iterations: int = 10000,
    seed: int = 20261018,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of per-case values.

    The case is the unit, as in the D2 gates: a case contributes one delta, and
    no sample is treated as an extra case.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    count = len(values)
    means: list[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(count):
            total += values[rng.randrange(count)]
        means.append(total / count)
    means.sort()
    lower_index = int((alpha / 2) * iterations)
    upper_index = min(iterations - 1, int((1 - alpha / 2) * iterations))
    return (round(means[lower_index], 6), round(means[upper_index], 6))


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    """Aggregate the per-case rows into the reported comparison numbers."""
    rows = result["rows"]
    deltas = result["deltas"]
    scored = [row for row in rows if not row["unknown"]]
    v1_recall = (
        sum(row["v1"]["fact_recall"] for row in scored) / len(scored) if scored else 0.0
    )
    v2_recall = (
        sum(row["v2"]["fact_recall"] for row in scored) / len(scored) if scored else 0.0
    )
    moved_up = [row["id"] for row in rows if row["recall_delta"] > 0]
    moved_down = [row["id"] for row in rows if row["recall_delta"] < 0]
    v1_stale = sum(1 for row in rows if row["v1"]["stale_fact"])
    v2_stale = sum(1 for row in rows if row["v2"]["stale_fact"])
    triple_1 = sum(1 for row in rows if row["v1"]["fact_recall"] == 1.0)
    triple_2 = sum(1 for row in rows if row["v2"]["fact_recall"] == 1.0)
    pooled = bootstrap_mean_interval(deltas)
    return {
        "scored_cases": len(scored),
        "unknown_cases": len(rows) - len(scored),
        "denominator_cases": len(rows),
        f"mean_fact_recall_{BASELINE_VERSION}": round(v1_recall, 6),
        f"mean_fact_recall_{VERSION}": round(v2_recall, 6),
        "mean_recall_delta": round(v2_recall - v1_recall, 6),
        "paired_bootstrap_95_interval_on_mean_delta": list(pooled),
        "interval_excludes_zero": pooled[0] > 0 or pooled[1] < 0,
        "cases_that_moved_up": moved_up,
        "cases_that_moved_down": moved_down,
        "cases_with_full_recall_v1": triple_1,
        "cases_with_full_recall_v2": triple_2,
        "stale_cases_v1": v1_stale,
        "stale_cases_v2": v2_stale,
        "unit": "case",
        "note": (
            "Both rules scored the same recorded replies in one pass. The frozen "
            "baseline number is unchanged and still reported under its own rule; "
            "v2 is a proposal, not a replacement."
        ),
    }
