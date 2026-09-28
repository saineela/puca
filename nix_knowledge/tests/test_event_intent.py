from nix_knowledge.event_intent import detect_event_intent
from nix_knowledge.engine import KnowledgeEngine
from nix_knowledge.needle import KnowledgeNeedle
from nix_knowledge.context import TemporalContext
from nix_knowledge.temporal import TemporalResolver
from nix_actions.engine import ActionsEngine


def test_reminder_interval_becomes_event_intent():
    intent = detect_event_intent("Remind me like every 2 days to take my medicine")
    assert intent.requires_event is True
    assert intent.kind == "reminder"
    assert intent.temporal_expression == "every 2 days"
    assert "take my medicine" in intent.title


def test_reminder_with_address_and_from_now():
    intent = detect_event_intent(
        "hey casper, I have to take medicines every 2 days from now remind me"
    )
    assert intent.requires_event is True
    assert intent.temporal_expression == "every 2 days"
    assert "take medicines" in intent.title


def test_plain_repeating_fact_is_not_an_alert():
    intent = detect_event_intent("I take medicine every 2 days")
    assert intent.requires_event is False


def test_event_gate_creates_recurring_action_without_model(tmp_path):
    knowledge = KnowledgeEngine(tmp_path / "knowledge.db")
    actions = ActionsEngine(tmp_path / "actions.db", timezone="America/Chicago")
    needle = object.__new__(KnowledgeNeedle)
    needle.engine = knowledge
    needle.actions_engine = actions
    needle.timezone = "America/Chicago"
    needle.temporal = TemporalResolver("America/Chicago")
    needle.context = TemporalContext("America/Chicago")
    needle.tools = needle._build_tool_schemas()
    needle.tool_functions = {"create_calendar_event": needle._create_calendar_event}

    try:
        payload = needle._process_route("Remind me like every 2 days to take my medicine")
        result = payload["result"]
        assert result["operation"] == "CREATE"
        assert result["data"]["recurring"] is True
        assert result["data"]["recurrence"] == "interval_2_days"
        assert result["actions"]["ok"] is True
        assert result["actions"]["recurrence"] == "interval_2_days"
        assert len(actions.list_actions(status="pending")) == 1
    finally:
        actions.close()
        knowledge.close()
