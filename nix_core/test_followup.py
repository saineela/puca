from datetime import datetime, timezone

from followup import FollowUpSession


def test_session_starts_and_exposes_expiry():
    now = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
    session = FollowUpSession(timeout_seconds=12)
    session.start(conversation_id="voice-1", now=now)
    assert session.active is True
    assert session.conversation_id == "voice-1"
    assert session.can_continue(now=now.replace(second=11)) is True
    assert session.can_continue(now=now.replace(second=12)) is False
    assert session.active is False


def test_recent_context_is_bounded():
    session = FollowUpSession(max_turns=3, max_chars=30)
    session.start(conversation_id="voice-2")
    for index in range(5):
        session.record("user", f"turn {index} abcdef")
    context = session.context()
    assert len(context) <= 3
    assert sum(len(item["content"]) for item in context) <= 30
    assert context[-1]["content"] == "turn 4 abcdef"


def test_end_clears_context_and_requires_new_start():
    session = FollowUpSession()
    session.start()
    session.record("user", "hello")
    session.end()
    assert session.active is False
    assert session.context() == []
    assert session.can_continue() is False
