"""Tests for the panel's push channel (``nix_core/panel_stream.py``).

The panel display used to learn about a new track, a question or a queue move by
asking the console again every few seconds. It should be told instead. What that
needs pinned is: the WebSocket handshake is a real one (a client library other
than the page can talk to it), a plain HTTP read of the socket path is refused
rather than half-answered, the existing ``/api/panel/state`` read is untouched,
and parking or clearing a question lands as a frame on every connected display.
"""
import http.client
import json
import socket
import threading

import pytest
from websockets.sync.client import connect

import panel_stream
import puca_panel


def _client_frame(payload: bytes, opcode: int = 0x1) -> bytes:
    """A masked client text frame, the way a browser sends one."""
    mask = b"\x01\x02\x03\x04"
    masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
    head = bytearray([0x80 | opcode])
    if len(payload) < 126:
        head.append(0x80 | len(payload))
    else:
        head.append(0x80 | 126)
        head += len(payload).to_bytes(2, "big")
    return bytes(head) + mask + masked


class _Headers(dict):
    """Enough of a request's headers for ``wants_upgrade`` to read."""


def _handler(headers: dict) -> object:
    return type("Handler", (), {"headers": _Headers(headers)})()


# ── the handshake ────────────────────────────────────────────────────────


def test_the_accept_key_is_the_rfc_example():
    # RFC 6455 §1.3: the handshake every browser and client library performs.
    assert panel_stream.accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_an_upgrade_is_told_from_a_plain_read():
    assert panel_stream.wants_upgrade(_handler({
        "Connection": "keep-alive, Upgrade",
        "Upgrade": "websocket",
        "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
        "Sec-WebSocket-Version": "13",
    }))
    assert not panel_stream.wants_upgrade(_handler({
        "Connection": "keep-alive",
        "Upgrade": "websocket",
        "Sec-WebSocket-Key": "dGhlIHNhbXBsZSBub25jZQ==",
    }))
    assert not panel_stream.wants_upgrade(_handler({
        "Connection": "Upgrade",
        "Upgrade": "websocket",
    }))


# ── the frames ───────────────────────────────────────────────────────────


def test_a_masked_client_frame_is_read_back():
    left, right = socket.socketpair()
    try:
        left.sendall(_client_frame(b'{"type":"ping"}'))
        opcode, payload = panel_stream._read_frame(right)
        assert opcode == 0x1
        assert payload == b'{"type":"ping"}'
    finally:
        left.close()
        right.close()


def test_a_server_frame_carries_its_length_and_opcode():
    frame = panel_stream._frame(0x1, b"hello")
    assert frame[0] == 0x81
    assert frame[1] == 5
    assert frame[2:] == b"hello"


def test_an_unmasked_client_frame_is_refused():
    left, right = socket.socketpair()
    try:
        # A client frame that skipped masking is a protocol error, not a message.
        left.sendall(bytes([0x81, 5]) + b"hello")
        assert panel_stream._read_frame(right) is None
    finally:
        left.close()
        right.close()


def test_the_hub_with_nobody_listening_reports_nobody():
    assert panel_stream.PanelStream().broadcast("change") == 0


# ── the listener ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def no_leaked_question():
    puca_panel.clear_question()
    yield
    puca_panel.clear_question()


@pytest.fixture
def panel_server():
    server = puca_panel.create_server("127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _socket_url(server) -> str:
    host, port = server.server_address[:2]
    return f"ws://{host}:{port}{panel_stream.SOCKET_PATH}"


def _http(server, method: str, path: str):
    host, port = server.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=5)
    connection.request(method, path)
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, body


def _options() -> list[dict]:
    return [
        {"video_id": "a", "title": "First", "artist": "Someone", "duration_sec": 200, "artwork_url": ""},
        {"video_id": "b", "title": "Second", "artist": "Someone", "duration_sec": 210, "artwork_url": ""},
    ]


def test_the_page_can_connect_and_is_greeted(panel_server):
    with connect(_socket_url(panel_server), open_timeout=5) as socket_client:
        hello = json.loads(socket_client.recv(timeout=5))
        assert hello["type"] == "hello"
        assert hello["socket"] == panel_stream.SOCKET_PATH


def test_parking_a_question_reaches_every_connected_display(panel_server):
    with connect(_socket_url(panel_server), open_timeout=5) as first, \
            connect(_socket_url(panel_server), open_timeout=5) as second:
        first.recv(timeout=5)   # hello
        second.recv(timeout=5)
        puca_panel.ask_choice("Which one did you mean?", _options(), source="play", query="which")
        for socket_client in (first, second):
            event = json.loads(socket_client.recv(timeout=5))
            assert event["type"] == "event"
            assert event["event"] == "question"
            assert event["question_id"]


def test_clearing_a_question_tells_the_wall(panel_server):
    puca_panel.ask_choice("Which one did you mean?", _options())
    with connect(_socket_url(panel_server), open_timeout=5) as socket_client:
        socket_client.recv(timeout=5)   # hello
        assert puca_panel.clear_question() is True
        event = json.loads(socket_client.recv(timeout=5))
        assert event["event"] == "question-cleared"


def test_the_socket_path_is_refused_when_it_is_not_a_websocket(panel_server):
    status, body = _http(panel_server, "GET", panel_stream.SOCKET_PATH)
    assert status == 426, "a plain read of a websocket must be told to upgrade"
    assert json.loads(body)["ok"] is False


def test_the_ordinary_state_read_is_untouched(panel_server):
    status, body = _http(panel_server, "GET", "/api/panel/state")
    assert status == 200
    payload = json.loads(body)
    assert payload["ok"] is True
    # The state advertises where the push channel is, so the page needs no
    # hard-coded path of its own.
    assert payload["panel"]["stream"] == panel_stream.SOCKET_PATH


def test_the_panel_info_advertises_the_stream():
    assert puca_panel.panel_info()["stream"] == panel_stream.SOCKET_PATH
