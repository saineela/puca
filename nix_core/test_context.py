"""Short-term context budgeting tests."""
from context import select_context


def test_context_keeps_recent_turns_and_both_roles():
    turns = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "system", "content": "must not be forwarded"},
        {"role": "user", "content": "latest question"},
        {"role": "assistant", "content": "latest answer"},
    ]
    selected = select_context(turns, max_turns=4, max_chars=1000)
    assert selected == [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest question"},
        {"role": "assistant", "content": "latest answer"},
    ]


def test_context_respects_character_budget_without_growing_forever():
    turns = [
        {"role": "user", "content": "one " * 100},
        {"role": "assistant", "content": "two " * 100},
        {"role": "user", "content": "latest request"},
    ]
    selected = select_context(turns, max_turns=10, max_chars=80)
    assert len(selected) <= 3
    assert sum(len(turn["content"]) for turn in selected) <= 80
    assert selected[-1]["content"] == "latest request"
