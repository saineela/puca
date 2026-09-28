from casper_v6_multiturn_eval import CASES, check_final


def test_multiturn_suite_covers_emotion_state_and_clarification():
    ids = {case.case_id for case in CASES}
    assert {"tired-follow-up", "state-supersession", "clarification-continuation"} <= ids


def test_current_state_case_rejects_old_state():
    case = next(item for item in CASES if item.case_id == "state-supersession")
    failures = check_final(case, "Person A is better now.")
    assert failures == []
    assert check_final(case, "Person A is sick.")


def test_tired_case_rejects_unnecessary_question():
    case = next(item for item in CASES if item.case_id == "tired-follow-up")
    failures = check_final(case, "I’ll keep it brief. What's on your mind?")
    assert any(item.startswith("forbidden") or item.startswith("question_count") for item in failures)
