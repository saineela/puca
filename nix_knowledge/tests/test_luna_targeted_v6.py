import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "nixlm"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_luna_targeted_v6 as corpus  # noqa: E402


def test_contextual_system_messages_use_real_newlines():
    message = corpus.system_message(corpus.SINGLE_CONTEXT["reminder_cadence"])
    assert "\n\nTrusted Nix Actions result:" in message["content"]
    assert "\\n\\nTrusted Nix Actions result:" not in message["content"]


def test_positive_news_context_does_not_assume_family_before_it_is_mentioned():
    context = corpus.CHAIN_CONTEXT["positive_ambiguity"].casefold()
    assert "apply this memory only if the current turn explicitly mentions a sister" in context
    assert "general good news is not about family unless the user says so" in context

    trajectories = [turns for category, turns in corpus.CHAINS if category == "positive_ambiguity"]
    assert trajectories
    for turns in trajectories:
        first_user, first_answer = turns[0]
        assert "sister" not in first_user.casefold()
        assert "sister" not in first_answer.casefold()


def test_authored_luna_examples_exclude_exact_v6_user_prompts():
    single_rows = [
        corpus.single(
            user,
            answer,
            category,
            context=corpus.SINGLE_CONTEXT.get(category),
        )
        for user, answer, category in corpus.AUTHORED_SINGLE
    ]
    chain_rows = [corpus.chain(turns, category) for category, turns in corpus.CHAINS]

    assert all(not corpus.overlaps_benchmark(row) for row in single_rows + chain_rows)


def test_reminder_cadence_claim_is_grounded_by_actions_success_context():
    context = corpus.SINGLE_CONTEXT["reminder_cadence"].casefold()
    reminders = [item for item in corpus.AUTHORED_SINGLE if item[2] == "reminder_cadence"]

    assert "successfully scheduled" in context
    assert len(reminders) >= 2
    assert all("scheduled" in answer.casefold() for _, answer, _ in reminders)
