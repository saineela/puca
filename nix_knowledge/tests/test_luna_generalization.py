import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "nixlm"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import luna_generalization_probes as probes  # noqa: E402


def test_probe_panel_has_fresh_fictional_cases_and_varied_categories():
    assert len(probes.PROBES) >= 8
    assert len({case["category"] for case in probes.PROBES}) >= 7
    assert all(case["id"] and case["turns"] for case in probes.PROBES)
    assert all(
        "SIMULATED TEST" in (case.get("initial_context") or "")
        for case in probes.PROBES
        if case.get("initial_context")
    )
    assert all(case.get("initial_context") is None for case in probes.PROBES if case["id"] == "action_result_arrives_after_request")


def test_probe_turns_have_no_exact_v6_benchmark_overlap():
    benchmark = probes.benchmark_user_turns()
    assert all(
        probes.normalize_text(turn["user"]) not in benchmark
        for _, _, turn in probes.iter_probe_turns()
    )


def test_probe_turns_have_no_exact_overlap_with_training_snapshots():
    audit = probes.audit_probe_overlap()
    assert audit["shared_v6_benchmark"]["exact_overlap_count"] == 0
    assert audit["blocking_exact_overlap_count"] == 0
    assert all(
        snapshot["exact_overlap_count"] == 0
        for snapshot in audit["training_snapshots"].values()
    )
    assert audit["probe_panel"]["all_entities_and_context_are_fictional"] is True


def test_dynamic_context_is_marked_for_injection_after_prior_assistant_turn():
    probe = next(
        case for case in probes.PROBES
        if case["id"] == "action_result_arrives_after_request"
    )
    turns = probe["turns"]
    assert turns[0].get("context_before") is None
    assert "SIMULATED TEST ACTION RESULT" in turns[1]["context_before"]
    assert "review_points" in turns[0] and "review_points" in turns[1]


def test_every_probe_turn_has_specific_manual_review_criteria():
    for _, _, turn in probes.iter_probe_turns():
        assert len(turn.get("review_points", ())) >= 2
