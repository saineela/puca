"""D4 - routing ledger statistics and the controlled replay contract.

The router already returns a deterministic decision. This module turns a set of
labelled, replayed requests into the two rates the proposal commits to, with
their denominators stated separately:

* false escalation = unnecessary escalations / all escalations;
* false de-escalation = incorrect fast-path answers / all fast-path answers.

Requests with no decidable label are reported as ``% undetermined`` and are
excluded from both rates. Route correctness and answer quality are separate
fields on purpose: a right route with a poor answer is not a routing failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

Label = Literal["fast_path_sufficient", "escalation_required", "undetermined"]


@dataclass(frozen=True)
class RoutedOutcome:
    """One labelled request after a controlled replay."""

    request_id: str
    chosen_route: str
    label: Label
    #: The answer the chosen route produced, judged against the label.
    route_correct: bool | None
    #: Reply quality, kept apart from routing correctness.
    answer_quality: float | None = None
    counterfactual: str = "not_replayed"
    replayed_route: str = ""
    counterfactual_correct: bool | None = None
    notes: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "chosen_route": self.chosen_route,
            "label": self.label,
            "route_correct": self.route_correct,
            "answer_quality": self.answer_quality,
            "counterfactual": self.counterfactual,
            "replayed_route": self.replayed_route,
            "counterfactual_correct": self.counterfactual_correct,
            "notes": self.notes,
        }


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def ledger_report(rows: Iterable[RoutedOutcome | Mapping[str, Any]]) -> dict[str, Any]:
    """Report both error rates, both denominators and the undetermined share."""
    outcomes: list[RoutedOutcome] = []
    for row in rows:
        if isinstance(row, RoutedOutcome):
            outcomes.append(row)
        else:
            outcomes.append(RoutedOutcome(**dict(row)))

    escalations = [row for row in outcomes if row.label == "escalation_required"]
    fast_path = [row for row in outcomes if row.label == "fast_path_sufficient"]
    undetermined = [row for row in outcomes if row.label == "undetermined"]

    # An escalation is unnecessary when the controlled replay shows the fast
    # path answered the same labelled request correctly. "It turned out fine"
    # is not enough: only a replay row can supply this.
    unnecessary = [
        row for row in escalations
        if row.chosen_route == "knowledge"
        and row.counterfactual == "replayed"
        and row.replayed_route == "chat"
        and row.counterfactual_correct is True
    ]
    incorrect_fast = [
        row for row in fast_path
        if row.chosen_route != "knowledge"
        and row.chosen_route != "unknown"
        and row.route_correct is False
    ]
    replayed = [row for row in outcomes if row.counterfactual == "replayed"]

    return {
        "total": len(outcomes),
        "false_escalation": _rate(len(unnecessary), len(escalations)),
        "false_escalation_numerator": len(unnecessary),
        "false_escalation_denominator": len(escalations),
        "false_de_escalation": _rate(len(incorrect_fast), len(fast_path)),
        "false_de_escalation_numerator": len(incorrect_fast),
        "false_de_escalation_denominator": len(fast_path),
        "undetermined": len(undetermined),
        "undetermined_share": _rate(len(undetermined), len(outcomes)),
        "replayed": len(replayed),
        # Only decided rows: an undetermined turn has no right route to be
        # correct about, so counting it in the denominator understates routing.
        "route_correctness": _rate(
            sum(1 for row in outcomes if row.route_correct is True),
            sum(1 for row in outcomes if row.route_correct is not None),
        ),
        "route_correctness_denominator": sum(
            1 for row in outcomes if row.route_correct is not None
        ),
        "answer_quality_mean": _rate(
            sum(
                row.answer_quality for row in outcomes
                if isinstance(row.answer_quality, (int, float))
            ),
            sum(
                1 for row in outcomes
                if isinstance(row.answer_quality, (int, float))
            ),
        ),
        "note": (
            "False escalation counts only replayed escalations whose controlled "
            "alternative was correct; answer quality is reported separately."
        ),
    }


def replay_row(
    *,
    request_id: str,
    chosen_route: str,
    label: Label,
    replayed_route: str,
    replayed_correct: bool,
    answer_quality: float | None = None,
) -> RoutedOutcome:
    """Build a row from a controlled replay, where the alternative ran frozen."""
    if chosen_route == "unknown" or label == "undetermined":
        route_correct = None
    elif label == "fast_path_sufficient":
        route_correct = chosen_route != "knowledge"
    else:
        route_correct = chosen_route == "knowledge"
    return RoutedOutcome(
        request_id=request_id,
        chosen_route=chosen_route,
        label=label,
        route_correct=route_correct,
        answer_quality=answer_quality,
        counterfactual="replayed",
        replayed_route=replayed_route,
        counterfactual_correct=replayed_correct,
    )
