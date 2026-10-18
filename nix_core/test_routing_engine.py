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


def test_question_about_the_users_own_data_is_a_knowledge_read():
    """D4 confirmation replay: 15 of 38 decidable turns came back `unknown`
    for this shape, so the gate must read the question, not the classifier."""
    engine = CoreRoutingEngine()
    for text in (
        "Which city do I live in?",
        "Who do I rent from?",
        "What medication should I avoid?",
        "What days am I at the gym now?",
        "What tea did I used to drink?",
        "What's the Wi-Fi key at home?",
        "What colour is my laptop?",
        "What is my garage code?",
        "what is my schedule",
    ):
        decision = engine.decide(text)
        assert decision.route == KNOWLEDGE, (text, decision.route, decision.rule)


def test_world_questions_are_still_chat():
    engine = CoreRoutingEngine()
    for text in (
        "How do I make bread?",
        "Can you tell me a joke?",
        "How many moons does Jupiter have?",
        "How do I get home?",
    ):
        decision = engine.decide(text)
        assert decision.route == CHAT, (text, decision.route, decision.rule)


def test_ambiguous_request_abstains_without_model_call():
    decision = CoreRoutingEngine().decide("maybe later")
    assert decision.route in {CHAT, UNKNOWN}
    assert decision.latency_ms < 10


def test_compound_knowledge_clause_is_not_silently_downgraded():
    decision = CoreRoutingEngine().decide("remember that I like tea")
    assert decision.route == KNOWLEDGE
    assert decision.requires_knowledge is True


def test_past_tense_cancellation_news_is_knowledge():
    # Oct 5 production failure: "my robotics practice has been
    # cancelled" fell to chat, leaving the stored reminder armed.
    for text in (
        "my robotics practice has been cancelled",
        "my robotics practice is cancelled",
        "i cancelled my robotics practice",
    ):
        decision = CoreRoutingEngine().decide(text)
        assert decision.route == KNOWLEDGE, text
        assert decision.requires_knowledge is True, text
        assert decision.rule == "past_tense_cancellation", text
