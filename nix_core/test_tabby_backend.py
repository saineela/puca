from __future__ import annotations

import json

from runtime_warmup import warm_models
from tabby_client import TabbyClient


def test_tabby_client_sends_openai_compatible_non_streaming_request(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "Hello."}}]}

    def post(url, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return Response()

    import tabby_client

    monkeypatch.setattr(tabby_client.requests, "post", post)
    client = TabbyClient(
        api_url="http://tabby.test/v1/chat/completions",
        model="casper-exl3",
        api_key="secret",
    )
    reply = client.chat(
        system_prompt="You are Casper.",
        history=[{"role": "user", "content": "Hi"}],
        user_text="How are you?",
        think=False,
    )

    assert reply == "Hello."
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["json"]["model"] == "casper-exl3"
    assert captured["json"]["stream"] is False
    assert [m["role"] for m in captured["json"]["messages"]] == [
        "system", "user", "user"
    ]
    assert "think" not in captured["json"]


def test_tabby_health_reports_advertised_model(monkeypatch):
    class Response:
        ok = True

        def json(self):
            return {"data": [{"id": "casper-exl3"}]}

    import tabby_client

    monkeypatch.setattr(tabby_client.requests, "get", lambda *args, **kwargs: Response())
    health = TabbyClient(
        api_url="http://tabby.test/v1/chat/completions",
        model="casper-exl3",
    ).health()
    assert health["ok"] is True
    assert health["model_loaded"] is True


def test_parallel_warmup_initializes_both_boundaries():
    order = []

    def casper():
        order.append("casper")

    def knowledge():
        order.append("knowledge")

    result = warm_models(casper_loader=casper, knowledge_health=knowledge)
    assert result == {"casper": "ready", "knowledge": "ready"}
    assert set(order) == {"casper", "knowledge"}
