from routing_engine import CHAT, KNOWLEDGE, UNKNOWN, CoreRoutingEngine


def test_social_turn_is_zero_model_fast_path():
    decision = CoreRoutingEngine().decide("hello? are you alive")
    assert decision.route == CHAT
    assert decision.requires_model is False
    assert decision.rule == "social_fast_path"
    assert decision.latency_ms < 10


def test_emotional_turn_does_not_hit_knowledge_or_predictor():
    decision = CoreRoutingEngine().decide("I'm exhausted and overwhelmed")
    assert decision.route == CHAT
    assert decision.requires_knowledge is False
    assert decision.requires_model is False


def test_personal_action_is_knowledge_with_high_confidence():
    decision = CoreRoutingEngine().decide("remind me to take medicines every 2 days")
    assert decision.route == KNOWLEDGE
    assert decision.requires_knowledge is True
    assert decision.confidence >= 0.9


def test_world_question_is_fast_chat_even_with_personal_word():
    decision = CoreRoutingEngine().decide("what is my laptop supposed to get this hot")
    assert decision.route == CHAT
    assert decision.requires_model is False


def test_ambiguous_request_abstains_without_model_call():
    decision = CoreRoutingEngine().decide("maybe later")
    assert decision.route in {CHAT, UNKNOWN}
    assert decision.latency_ms < 10


def test_compound_knowledge_clause_is_not_silently_downgraded():
    decision = CoreRoutingEngine().decide("remember that I like tea")
    assert decision.route == KNOWLEDGE
    assert decision.requires_knowledge is True
