from router import CHAT, KNOWLEDGE, classify


def test_ordinary_emotional_turn_skips_model_gate():
    route, features = classify("I'm feeling a little overwhelmed today")
    assert route == CHAT
    assert features["rule"] == "ordinary_chat_statement"


def test_indirect_personal_action_uses_fast_knowledge_route():
    route, features = classify("would you keep track that I owe Joel twenty dollars")
    assert route == KNOWLEDGE
    assert features["rule"] == "fast_knowledge_hint"


def test_medicine_reminder_and_delete_are_deterministic():
    for text in (
        "remind me to take my medicines every 2 days",
        "delete my reminder for taking medicienes",
    ):
        route, _features = classify(text)
        assert route == KNOWLEDGE
