from brain import (
    Brain,
    OllamaClient,
    creative_chat_request,
    is_creator_identity_request,
    is_user_profile_identity_request,
    _matches_configured_user_introduction,
    is_assistant_identity_request,
    social_companion_reply,
    request_requires_thinking,
    should_delegate_to_knowledge,
)


class Knowledge:
    def memory_block(self, current_text=None):
        return "Stored facts and preferences:\n- user likes tea"

    def digest(self):
        return ""

    def intent(self, text):
        return {"ok": False}

    def learn_keys(self, text):
        return []

    def classify(self, text):
        return "chat"

    def lookup_key_person(self, name):
        return []


class Actions:
    def context(self):
        return []

    def log_turn(self, **kwargs):
        pass


class Chat:
    model = "qwen3.5:4b"

    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return "Hey."


def test_user_profile_identity_question_shapes_are_recognized():
    for text in (
        "Who am I?",
        "What is my name?",
        "What's my name?",
        "Do you know who I am?",
        "Do you know my name?",
    ):
        assert is_user_profile_identity_request(text) is True

    assert is_user_profile_identity_request("Who is Microsoft?") is False


def test_configured_user_introduction_is_acknowledged_without_identity_challenge():
    assert _matches_configured_user_introduction("I am Sai", "Sai Neela") is True
    assert _matches_configured_user_introduction("I'm Sai Neela", "Sai Neela") is True
    assert _matches_configured_user_introduction("My name is Sai", "Sai Neela") is True
    assert _matches_configured_user_introduction("I am Sam", "Sai Neela") is False


def test_identity_extensions_are_not_split_into_knowledge():
    assert is_assistant_identity_request("Who are you? An alien?") is True
    assert creative_chat_request("tell me a story with my name included") is True
    assert creative_chat_request("schedule a dentist appointment tomorrow") is False
    assert social_companion_reply("Casper, you are very sweet?") == "That’s sweet of you to say."
    assert social_companion_reply("I am your father") == "That explains the dramatic entrance. Hi, Dad."
    from brain import low_stakes_preference_reply
    assert low_stakes_preference_reply(
        "do you prefer SpongeBob popsicles or strawberry shortcake popsicles?"
    )


def test_creator_identity_requests_are_recognized():
    for text in (
        "Who created Casper?",
        "Who built you?",
        "Who is your developer?",
        "Who made NIX?",
        "Who developed this PUCA?",
    ):
        assert is_creator_identity_request(text) is True

    assert is_creator_identity_request("Who is Microsoft?") is False


def test_creator_identity_bypasses_model_and_hides_internal_metadata():
    chat = Chat()
    response = Brain(
        knowledge=Knowledge(), actions=Actions(), ollama=chat, log_requests=False
    ).handle(text="Who created Casper?")

    assert response["reply"] == (
        "NIX PUCA was created by Sai Neela; I'm Luna, here in NIX with you."
    )
    assert chat.calls == []
    assert "CHAT" not in response["reply"]
    assert "casper_transformers" not in response["reply"]
    assert response["details"]["model_called"] is False


def test_creative_request_bypasses_knowledge_and_does_not_write_a_fact():
    chat = Chat()
    knowledge = Knowledge()
    response = Brain(
        knowledge=knowledge, actions=Actions(), ollama=chat, log_requests=False
    ).handle(text="tell me a story with my name included")
    assert response["route"] == "chat"
    assert response["rule"] == "creative_request_guard"
    assert knowledge.classify("unused") == "chat"


def test_personal_memory_requests_bypass_chat_routing():
    assert should_delegate_to_knowledge("Do you know who I am?") is True
    assert should_delegate_to_knowledge("What is my name?") is True
    assert should_delegate_to_knowledge("How is my sister?") is True
    assert should_delegate_to_knowledge("Tell me a joke") is False
    assert should_delegate_to_knowledge("What is the capital of Australia?") is False


def test_casual_fragments_never_enter_knowledge():
    from router import classify

    for text in ("myself", "stoppp", "break the system"):
        route, features = classify(text)
        assert route == "chat", (text, features)
        assert features["rule"] == "casual_fragment_guard"


def test_alive_social_turn_bypasses_model(monkeypatch):
    import brain

    monkeypatch.setattr(
        brain,
        "_active_assistant_identity",
        lambda: ("Casper", "PUCA (Personal User Companion Agent)", "test-casper"),
    )
    chat = Chat()
    response = Brain(
        knowledge=Knowledge(), actions=Actions(), ollama=chat, log_requests=False
    ).handle(text="hello? are you alive")
    assert response["rule"] == "social_companion"
    assert response["reply"] == "Yeah, I’m here."
    assert chat.calls == []


def test_playful_social_turn_bypasses_model(monkeypatch):
    import brain

    monkeypatch.setattr(
        brain,
        "_active_assistant_identity",
        lambda: ("Casper", "PUCA (Personal User Companion Agent)", "test-casper"),
    )
    chat = Chat()
    response = Brain(
        knowledge=Knowledge(), actions=Actions(), ollama=chat, log_requests=False
    ).handle(text="I am your father")
    assert response["rule"] == "social_companion"
    assert response["reply"] == "That explains the dramatic entrance. Hi, Dad."
    assert chat.calls == []


def test_only_explicit_complexity_requests_enable_thinking():
    assert request_requires_thinking("hi") is False
    assert request_requires_thinking("I'm tired") is False
    assert request_requires_thinking("remember that I like tea") is False
    assert request_requires_thinking("schedule a dentist appointment tomorrow") is False
    assert request_requires_thinking("analyze the trade-offs and compare both designs") is True
    assert request_requires_thinking("debug this Python code step by step") is True


def test_simple_chat_forces_non_thinking_even_when_global_flag_is_enabled(monkeypatch):
    import brain

    monkeypatch.setattr(brain, "OLLAMA_THINK", True)
    chat = Chat()
    response = Brain(
        knowledge=Knowledge(), actions=Actions(), ollama=chat, log_requests=False
    ).handle(text="hi")
    assert response["reply"] == "Hey."
    assert chat.calls[0]["think"] is False
    assert response["details"]["think"] is False


def test_complex_chat_uses_predictor_thinking_decision(monkeypatch):
    import brain

    class ThinkingKnowledge(Knowledge):
        def thinking_decision(self, text):
            return {"think": False, "mode": "FAST", "source": "qwen2.5-0.5b"}

    monkeypatch.setattr(brain, "OLLAMA_THINK", True)
    chat = Chat()
    response = Brain(
        knowledge=ThinkingKnowledge(), actions=Actions(), ollama=chat, log_requests=False
    ).handle(text="analyze the trade-offs between these designs")
    assert chat.calls[0]["think"] is False
    assert response["details"]["think"] is False


def test_malformed_predictor_boolean_cannot_enable_thinking(monkeypatch):
    import brain

    class MalformedKnowledge(Knowledge):
        def thinking_decision(self, text):
            return {"think": "false", "source": "qwen2.5-0.5b"}

    monkeypatch.setattr(brain, "OLLAMA_THINK", True)
    chat = Chat()
    response = Brain(
        knowledge=MalformedKnowledge(), actions=Actions(), ollama=chat, log_requests=False
    ).handle(text="analyze the trade-offs between these designs")
    assert chat.calls[0]["think"] is False
    assert response["details"]["thinking_source"] == "luna_thinking_disabled"


def test_casper_thinking_is_disabled_even_for_complex_requests(monkeypatch):
    import brain

    monkeypatch.setattr(brain, "OLLAMA_THINK", True)
    chat = Chat()
    response = Brain(
        knowledge=Knowledge(), actions=Actions(), ollama=chat, log_requests=False
    ).handle(text="analyze the trade-offs between these designs")
    assert chat.calls[0]["think"] is False
    assert response["details"]["think"] is False
    assert response["details"]["thinking_source"] == "luna_thinking_disabled"


def test_ollama_request_contains_explicit_think_flag(monkeypatch):
    import brain

    captured = {}

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "hello"}}

    def post(url, json, timeout):
        captured.update(json)
        return Response()

    monkeypatch.setattr(brain.requests, "post", post)
    client = OllamaClient(api_url="http://ollama/api/chat", model="qwen3.5:4b")
    assert client.chat(
        system_prompt="You are Casper.",
        history=[],
        user_text="hi",
        think=False,
    ) == "hello"
    assert captured["think"] is False
