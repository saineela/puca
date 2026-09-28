from casper_v6_eval import CASES, check


def test_good_current_state_response_passes():
    case = next(item for item in CASES if item.case_id == "current-state")
    assert check(case, "Maya is feeling better now.") == []


def test_old_state_and_generic_question_fail():
    case = next(item for item in CASES if item.case_id == "current-state")
    failures = check(case, "Maya is sick. What's on your mind?")
    assert any(item.startswith("forbidden: sick") for item in failures)


def test_v6_corpus_has_boundary_cases():
    ids = {case.case_id for case in CASES}
    assert {"tired", "ambiguous-person", "recurring-event", "multi-intent"} <= ids
