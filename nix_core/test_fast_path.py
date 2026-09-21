from brain import (
    Brain,
    OllamaClient,
    is_creator_identity_request,
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
        "Created and Built by Sai Neela, and living in NIX's PUCA system."
    )
    assert chat.calls == []
    assert "CHAT" not in response["reply"]
    assert "casper_transformers" not in response["reply"]
    assert response["details"]["model_called"] is False


def test_personal_memory_requests_bypass_chat_routing():
    assert should_delegate_to_knowledge("Do you know who I am?") is True
    assert should_delegate_to_knowledge("What is my name?") is True
    assert should_delegate_to_knowledge("How is my sister?") is True
    assert should_delegate_to_knowledge("Tell me a joke") is False
    assert should_delegate_to_knowledge("What is the capital of Australia?") is False


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


def test_complex_chat_can_opt_into_thinking(monkeypatch):
    import brain

    monkeypatch.setattr(brain, "OLLAMA_THINK", True)
    chat = Chat()
    response = Brain(
        knowledge=Knowledge(), actions=Actions(), ollama=chat, log_requests=False
    ).handle(text="analyze the trade-offs between these designs")
    assert chat.calls[0]["think"] is True
    assert response["details"]["think"] is True


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
