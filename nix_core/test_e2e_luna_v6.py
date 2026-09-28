"""Hermetic console → Core → Knowledge → Actions integration test.

This exercises the production HTTP routes and service wiring with isolated
SQLite databases. Model generation is stubbed, but Core's selected model
identity is Luna V6; no model weights or external services are required.
"""
from __future__ import annotations

import ipaddress
import json
from datetime import datetime, timedelta
import socket
from collections import deque
from http.server import ThreadingHTTPServer
from threading import Lock, Thread
from urllib.request import Request, urlopen

import pytest

import assistant_model
import brain
import config
import console
from luna_runtime import LUNA_V6_MODEL_ID
from nix_knowledge.engine import KnowledgeEngine


class _LunaV6TestClient:
    model = LUNA_V6_MODEL_ID

    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        request = kwargs["user_text"]
        if "KNOWLEDGE RESULT" in request:
            return "Done."
        return "I'm Luna, your conversational companion."


def _request(base_url, path, body=None, method="GET"):
    data = json.dumps(body).encode() if body is not None else None
    request = Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urlopen(request, timeout=20) as response:
        return response.status, json.loads(response.read().decode())


@pytest.fixture
def nix_e2e(monkeypatch, tmp_path, request):
    # Save process-wide bridge state before any monkeypatches alter it.
    bridge_original_request = console.requests.Session.request
    bridge_original_services = dict(console._local_services)
    bridge_was_installed = console._bridge_installed

    # Configure the real in-process APIs before importing or resetting them.
    monkeypatch.setenv("NIX_ISOLATE_DASHBOARD_DATA", "1")
    monkeypatch.setenv("NIX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NIX_PROFILE_PATH", str(tmp_path / "profile.json"))
    monkeypatch.setenv("NIX_KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    monkeypatch.setenv("NIX_ACTIONS_DB", str(tmp_path / "actions.db"))
    monkeypatch.setenv("NIX_CORE_DB", str(tmp_path / "nix_core.db"))
    monkeypatch.setenv("NIX_TZ", "America/Chicago")
    monkeypatch.setenv("NIX_REQUEST_LOG", "0")
    monkeypatch.setenv("NIX_KNOWLEDGE_MODEL_GATE", "0")
    monkeypatch.setenv("NIX_CORE_USE_KNOWLEDGE_MODEL_GATE", "0")
    monkeypatch.setenv("NIX_CORE_USE_NEURAL_INTENT", "0")
    monkeypatch.setenv("NIX_CORE_USE_CUSTOM_ROUTING_PREDICTOR", "0")
    monkeypatch.setenv("NIX_CORE_WARMUP_MODELS", "0")
    # resolve_services replaces the Knowledge URL with its internal bridge
    # address; these placeholders keep any pre-bridge fallback harmless.
    monkeypatch.setenv("NIX_KNOWLEDGE_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("NIX_ACTIONS_API_URL", "http://127.0.0.1:9")

    monkeypatch.setattr(console, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(console, "KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    monkeypatch.setattr(console, "ACTIONS_DB", str(tmp_path / "actions.db"))
    monkeypatch.setattr(console, "CORE_DB", str(tmp_path / "nix_core.db"))
    monkeypatch.setattr(console, "_brain", None)
    monkeypatch.setattr(console, "_local_services", {})
    monkeypatch.setattr(console, "_bridge_installed", False)
    monkeypatch.setattr(console, "TRACE", deque(maxlen=600))
    monkeypatch.setattr(console.Handler, "warmup_status", {"warmup": "test_stub"}, raising=False)
    monkeypatch.setattr(console, "CASPER_BACKEND", "transformers")
    monkeypatch.setattr(console, "OPENAI_API_KEY", "")

    # Ensure the code imported before the fixture follows the test-only model
    # and feature flags too; environment changes alone would not update these
    # module constants.
    monkeypatch.setattr(config, "CASPER_BACKEND", "transformers")
    monkeypatch.setattr(config, "USE_NEURAL_INTENT", False)
    monkeypatch.setattr(config, "USE_KNOWLEDGE_MODEL_GATE", False)
    monkeypatch.setattr(config, "USE_CUSTOM_ROUTING_PREDICTOR", False)
    monkeypatch.setattr(config, "WARMUP_MODELS", False)
    monkeypatch.setattr(config, "TIMEZONE", "America/Chicago")
    monkeypatch.setattr(brain, "CASPER_BACKEND", "transformers")
    monkeypatch.setattr(brain, "USE_NEURAL_INTENT", False)
    monkeypatch.setattr(brain, "USE_KNOWLEDGE_MODEL_GATE", False)
    monkeypatch.setattr(brain, "USE_CUSTOM_ROUTING_PREDICTOR", False)
    monkeypatch.setattr(brain, "WARMUP_MODELS", False)
    monkeypatch.setattr(brain, "TIMEZONE", "America/Chicago")

    # Fail closed if any downstream request bypasses the embedded API bridge.
    def deny_requests_egress(_self, method, url, *args, **kwargs):
        raise AssertionError(f"unexpected requests egress: {method} {url}")

    monkeypatch.setattr(
        console.requests.Session,
        "request",
        deny_requests_egress,
    )

    # Also guard direct urllib/socket clients. Only the HTTP port opened by
    # this fixture is permitted; DNS and all other TCP egress fail closed.
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo
    allowed_loopback_ports = set()

    import transformers

    def deny_model_loading(*args, **kwargs):
        pytest.fail("hermetic integration test attempted to load model weights/tokenizer")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", deny_model_loading)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", deny_model_loading)

    def is_loopback_host(value):
        host = str(value).split("%", 1)[0]
        if host.lower() == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host is not None and not is_loopback_host(host):
            raise AssertionError(f"blocked DNS/network lookup: {host}")
        return original_getaddrinfo(host, *args, **kwargs)

    def guard_socket_address(sock, address):
        if sock.family not in (socket.AF_INET, socket.AF_INET6):
            return
        if (
            not isinstance(address, tuple)
            or not address
            or not is_loopback_host(address[0])
            or len(address) < 2
            or address[1] not in allowed_loopback_ports
        ):
            raise AssertionError(f"blocked socket connection: {address}")

    def guarded_connect(sock, address):
        guard_socket_address(sock, address)
        return original_connect(sock, address)

    def guarded_connect_ex(sock, address):
        guard_socket_address(sock, address)
        return original_connect_ex(sock, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)

    import actions_api
    import knowledge_api
    import request_log

    # The embedded HTTP bridge is process-global. Restore its state after
    # closing fixture-owned databases so later tests inherit no test routing.
    # The embedded API modules are process-wide singletons. Give this fixture
    # fresh paths and caches, then restore pre-existing state via monkeypatch.
    monkeypatch.setattr(knowledge_api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(knowledge_api, "KNOWLEDGE_DB", str(tmp_path / "knowledge.db"))
    monkeypatch.setattr(knowledge_api, "ACTIONS_DB", str(tmp_path / "actions.db"))
    monkeypatch.setattr(knowledge_api, "TIMEZONE", "America/Chicago")
    monkeypatch.setattr(knowledge_api, "_needle", None)
    monkeypatch.setattr(knowledge_api, "_actions_engine", None)
    monkeypatch.setattr(knowledge_api, "_intent_classifier", None)
    monkeypatch.setattr(knowledge_api, "_state_lock", Lock())
    monkeypatch.setattr(knowledge_api, "_intent_lock", Lock())

    monkeypatch.setattr(actions_api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(actions_api, "ACTIONS_DB", str(tmp_path / "actions.db"))
    monkeypatch.setattr(actions_api, "CORE_DB", str(tmp_path / "nix_core.db"))
    monkeypatch.setattr(actions_api, "TIMEZONE", "America/Chicago")
    monkeypatch.setattr(actions_api, "_core", None)
    monkeypatch.setattr(actions_api, "_actions", None)

    # Disable the lazy vector layer even on fact/event writes: its first access
    # can otherwise construct a SentenceTransformer-backed embedder.
    monkeypatch.setattr(KnowledgeEngine, "semantic", property(lambda _self: None))
    monkeypatch.setattr(request_log, "_ENABLED", False)
    monkeypatch.setattr(request_log, "_LOG_DIR", str(tmp_path / "logs"))

    monkeypatch.setattr(brain.Brain, "warmup", lambda self: {"warmup": "test_stub"})

    import luna_model

    fake_client = _LunaV6TestClient()
    monkeypatch.setattr(assistant_model, "CASPER_BACKEND", "transformers")
    monkeypatch.setattr(assistant_model, "DEFAULT_MODEL_ID", LUNA_V6_MODEL_ID)
    monkeypatch.setattr(assistant_model, "_selected_model", LUNA_V6_MODEL_ID)
    monkeypatch.setattr(assistant_model, "LUNA_BASE_PATH", tmp_path)
    monkeypatch.setattr(assistant_model, "LUNA_V6_ADAPTER_PATH", tmp_path / "adapter")
    monkeypatch.setattr(assistant_model, "_ensure_model_present", lambda _model: None)
    monkeypatch.setattr(assistant_model, "get_chat_client", lambda: fake_client)
    monkeypatch.setattr(luna_model, "_active_luna_model", None)
    monkeypatch.setattr(luna_model, "_client", None)

    server = ThreadingHTTPServer(("127.0.0.1", 0), console.Handler)
    allowed_loopback_ports.add(server.server_address[1])
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (
            f"http://127.0.0.1:{server.server_address[1]}",
            fake_client,
            knowledge_api,
            actions_api,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

        # Close only fixture-created API connections before the temporary
        # directory disappears. Existing module state is restored by pytest.
        actions_core = getattr(actions_api, "_core", None)
        handles = (
            getattr(getattr(knowledge_api, "_needle", None), "engine", None),
            getattr(knowledge_api, "_actions_engine", None),
            actions_core,
            getattr(actions_core, "actions", None),
            getattr(actions_api, "_actions", None),
        )
        closed = set()
        for handle in handles:
            if handle is None or id(handle) in closed:
                continue
            closed.add(id(handle))
            try:
                close = getattr(handle, "close", None)
                if callable(close):
                    close()
                else:
                    connection = getattr(handle, "connection", None)
                    if connection is not None:
                        connection.close()
            except Exception:
                pass

        console.requests.Session.request = bridge_original_request
        console._local_services.clear()
        console._local_services.update(bridge_original_services)
        console._bridge_installed = bridge_was_installed


def test_luna_v6_core_knowledge_actions_path_is_end_to_end(nix_e2e):
    base_url, fake_client, knowledge_api, _actions_api = nix_e2e
    conversation_id = "hermetic-luna-v6-e2e"

    # Verify official status, then exercise an ordinary chat route. The fake
    # client must receive the prompt, proving Core crossed the model boundary.
    status, model_status = _request(base_url, "/api/model")
    assert status == 200
    assert model_status["active"] == LUNA_V6_MODEL_ID
    assert model_status["assistant_name"] == "Luna"
    assert model_status["active_model_present"] is False  # test adapter is stubbed

    status, profile = _request(base_url, "/api/profile")
    assert status == 200
    assert profile["user_name"] == ""
    status, profile = _request(
        base_url,
        "/api/profile",
        {"user_name": "Sai Neela"},
        method="POST",
    )
    assert status == 200
    assert profile["user_name"] == "Sai Neela"

    chat_text = "Tell me one fun fact about octopuses, keep it short"
    status, hello = _request(base_url, "/api/send", {
        "text": chat_text,
        "location": "test",
        "conversation_id": conversation_id,
    }, method="POST")
    assert status == 200
    assert hello["route"] == "chat"
    assert hello["details"]["official_model"] == LUNA_V6_MODEL_ID
    assert hello["details"]["assistant_name"] == "Luna"
    assert hello["pipeline"][-1]["label"] == f"Luna · {LUNA_V6_MODEL_ID} final response"
    assert "Luna" in hello["reply"]
    assert fake_client.calls[-1]["history"] == []
    chat_call = next(
        call for call in fake_client.calls if call.get("user_text") == chat_text
    )
    assert chat_call["think"] is False
    # The exact concise standalone identity prompt is passed to Luna V6.
    from luna_format import IDENTITY_SYSTEM
    assert chat_call["system_prompt"].startswith(IDENTITY_SYSTEM)
    assert "You are the user's conversational companion" in chat_call["system_prompt"]
    assert "curious, gently playful" not in chat_call["system_prompt"]
    assert "Answer in one or two short sentences" not in chat_call["system_prompt"]
    assert "ANTI-INTERVIEW RULE" not in chat_call["system_prompt"]
    assert 'preferred name in NIX Settings is "Sai Neela"' in chat_call["system_prompt"]

    # The preferred name is available for direct recall and does not require
    # the model to infer who the user is or authenticate the setting.
    calls_before_identity = len(fake_client.calls)
    status, identity = _request(
        base_url,
        "/api/send",
        {"text": "Who am I?", "location": "test", "conversation_id": conversation_id},
        method="POST",
    )
    assert status == 200
    assert identity["reply"] == "You're Sai Neela."
    assert identity["details"]["model_called"] is False
    assert len(fake_client.calls) == calls_before_identity

    status, creator = _request(
        base_url,
        "/api/send",
        {"text": "Who created NIX?", "location": "test", "conversation_id": conversation_id},
        method="POST",
    )
    assert status == 200
    assert "created by Sai Neela" in creator["reply"]
    assert creator["details"]["model_called"] is False

    calls_before_introduction = len(fake_client.calls)
    status, introduced = _request(
        base_url,
        "/api/send",
        {"text": "I am Sai", "location": "test", "conversation_id": conversation_id},
        method="POST",
    )
    assert status == 200
    assert introduced["reply"] == "Got it, Sai Neela."
    assert introduced["details"]["model_called"] is False
    assert len(fake_client.calls) == calls_before_introduction

    # Fact write/recall covers Core → Knowledge persistence and retrieval.
    status, stored = _request(base_url, "/api/send", {
        "text": "remember that I like tea",
        "location": "test",
        "conversation_id": conversation_id,
    }, method="POST")
    assert status == 200
    assert stored["route"] == "knowledge"
    assert stored["details"]["official_model"] == LUNA_V6_MODEL_ID
    assert "tea" in stored["reply"].lower()

    status, recalled = _request(base_url, "/api/send", {
        "text": "what do you remember about tea?",
        "location": "test",
        "conversation_id": conversation_id,
    }, method="POST")
    assert status == 200
    assert recalled["route"] == "knowledge"
    assert "tea" in json.dumps(recalled["details"]).lower()

    # Reminder creation covers symbolic temporal resolution and the
    # Knowledge → Actions scheduling bridge using test-only content.
    status, scheduled = _request(base_url, "/api/send", {
        "text": "remind me to stretch in 2 days",
        "location": "test",
        "conversation_id": conversation_id,
    }, method="POST")
    assert status == 200
    assert scheduled["route"] == "knowledge"
    assert scheduled["details"]["official_model"] == LUNA_V6_MODEL_ID
    result = scheduled["details"]["result"]
    assert result["operation"] == "CREATE"
    assert result["record_type"] == "event"
    assert result["actions"]["ok"] is True

    # Casual voice phrasing with a proper-name acronym and a full time
    # range must still become a confirmed calendar record and linked action.
    status, tsa_meeting = _request(base_url, "/api/send", {
        "text": "alright bro, I have a TSA meeting tmr from 4pm to 6pm",
        "location": "test",
        "conversation_id": conversation_id,
    }, method="POST")
    assert status == 200
    assert tsa_meeting["route"] == "knowledge"
    tsa_result = tsa_meeting["details"]["result"]
    assert tsa_result["operation"] == "CREATE"
    assert tsa_result["record_type"] == "event"
    assert tsa_result["data"]["title"] == "TSA meeting"
    expected_tomorrow = (
        datetime.fromisoformat(tsa_result["temporal_context"]["current_date"])
        + timedelta(days=1)
    ).date().isoformat()
    assert tsa_result["data"]["start"].startswith(f"{expected_tomorrow}T16:00:00")
    assert tsa_result["data"]["end"].startswith(f"{expected_tomorrow}T18:00:00")
    assert tsa_result["actions"]["ok"] is True

    status, feed = _request(base_url, "/api/feed")
    assert status == 200
    assert any("stretch" in json.dumps(event).lower() for event in feed["actions"])
    assert any("tsa meeting" in json.dumps(event).lower() for event in feed["actions"])
    assert any("tea" in json.dumps(record).lower() for record in feed["knowledge"])

    # Guard the main accidental-load paths through each write and lookup above.
    assert knowledge_api._needle is not None
    assert knowledge_api._needle.model is None
    assert knowledge_api._intent_classifier is None
    assert all(call.get("think") is False for call in fake_client.calls)
