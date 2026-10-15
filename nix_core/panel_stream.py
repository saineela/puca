"""A push channel for the walls the console already serves.

The panel display is a page in a room, and the console is a process on the same
box a few milliseconds away. Until now the page found out that something had
moved by asking again every three seconds (``GET /api/panel/state`` on its own
timer): a track that starts, a question that pops up, a queue that advances all
waited for the next poll to be seen. That is the wrong shape for two halves of
the same machine — the display should be *told*, not made to keep asking.

This is the smallest server-side WebSocket that does that, deliberately in this
tree rather than a second listener on a second port:

* it upgrades the panel's **own** connection, so the socket shares the page's
  origin (no CORS, no mixed content) and its TLS listener (the microphone
  origin) serves ``wss://`` for free, with no extra certificate or port to open;
* a message is only ever an *event* — "something changed" — and the page then
  reads the state it already knows how to render. The heavy snapshot
  (``puca_panel.panel_state``) stays a plain HTTP read, so a change on the
  console's side is one small frame and the existing, tested pipeline does the
  rest;
* the hub is a process-wide singleton, so ``music_api`` and ``puca_panel`` can
  both ring it without importing each other.

The frame code is the RFC 6455 subset a push channel needs: text frames in and
out, ping/pong, and a close handshake. Fragmented client frames are reassembled;
anything larger than the channel will ever accept is refused by dropping the
connection rather than buffering it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import threading
import time
from typing import Any

# The fixed GUID every WebSocket server hashes the client's key with (RFC 6455).
_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# What this channel will accept from a client. The page sends nothing but the
# occasional ping, so this is a ceiling on nonsense, not a real message size.
MAX_FRAME = 64 * 1024

# With nothing to say, a client that has gone away is otherwise indistinguishable
# from one sitting quietly on a wall, so the server speaks first this often.
IDLE_SECONDS = 45.0

SOCKET_PATH = "/api/panel/socket"


# ---------------------------------------------------------------------------
# The handshake
# ---------------------------------------------------------------------------


def wants_upgrade(handler: Any) -> bool:
    """Whether a request is a WebSocket upgrade for this handler to answer."""
    connection = (handler.headers.get("Connection") or "").lower()
    upgrade = (handler.headers.get("Upgrade") or "").lower()
    key = handler.headers.get("Sec-WebSocket-Key") or ""
    version = (handler.headers.get("Sec-WebSocket-Version") or "").strip()
    return (
        upgrade == "websocket"
        and "upgrade" in connection
        and bool(key)
        and version in {"", "13"}
    )


def accept_key(key: str) -> str:
    """The ``Sec-WebSocket-Accept`` a client's key is answered with."""
    digest = hashlib.sha1((key + _GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def _handshake(handler: Any) -> None:
    key = handler.headers.get("Sec-WebSocket-Key") or ""
    # A client library refuses a 101 written as HTTP/1.0; the base handler
    # defaults to it, and RFC 6455 expects HTTP/1.1.
    handler.protocol_version = "HTTP/1.1"
    handler.send_response_only(101, "Switching Protocols")
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", accept_key(key))
    handler.end_headers()
    # The HTTP request loop is over; this socket is a frame channel now.
    handler.close_connection = True


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def _frame(opcode: int, payload: bytes) -> bytes:
    """One unmasked server frame (servers never mask; clients always do)."""
    length = len(payload)
    head = bytearray([0x80 | (opcode & 0x0F)])
    if length < 126:
        head.append(length)
    elif length < (1 << 16):
        head.append(126)
        head += length.to_bytes(2, "big")
    else:
        head.append(127)
        head += length.to_bytes(8, "big")
    return bytes(head) + payload


def _read_exact(connection: socket.socket, count: int) -> bytes | None:
    """Exactly ``count`` bytes, or None if the far side closed or timed out."""
    chunks = bytearray()
    while len(chunks) < count:
        try:
            piece = connection.recv(count - len(chunks))
        except socket.timeout:
            raise
        except OSError:
            return None
        if not piece:
            return None
        chunks += piece
    return bytes(chunks)


def _read_frame(connection: socket.socket) -> tuple[int, bytes] | None:
    """One complete client frame as ``(opcode, payload)``, or None on close.

    Client frames are always masked, and the mask is removed here. Control
    frames (close, ping, pong) are returned as they arrive; a fragmented message
    is reassembled across its continuation frames before it is returned.
    """
    message: bytearray | None = None
    while True:
        first = _read_exact(connection, 2)
        if first is None:
            return None
        fin = bool(first[0] & 0x80)
        opcode = first[0] & 0x0F
        masked = bool(first[1] & 0x80)
        length = first[1] & 0x7F
        if length == 126:
            extended = _read_exact(connection, 2)
            if extended is None:
                return None
            length = int.from_bytes(extended, "big")
        elif length == 127:
            extended = _read_exact(connection, 8)
            if extended is None:
                return None
            length = int.from_bytes(extended, "big")
        if length > MAX_FRAME or not masked:
            # Too large to hold, or a client that skipped masking: refuse it.
            return None
        key = _read_exact(connection, 4)
        if key is None:
            return None
        body = _read_exact(connection, length)
        if body is None:
            return None
        buffer = bytearray(b ^ key[i % 4] for i, b in enumerate(body))
        if opcode >= 0x8:            # a control frame may not be fragmented
            return opcode, bytes(buffer)
        if opcode == 0x0:            # continuation of the message being built
            if message is None:
                return None
            message += buffer
            if fin:
                return 0x1, bytes(message)
            continue
        message = buffer
        if fin:
            return opcode, bytes(message)


# ---------------------------------------------------------------------------
# One connection
# ---------------------------------------------------------------------------


class Client:
    """A display that has upgraded; writes are serialised, a dead one is dropped."""

    def __init__(self, handler: Any) -> None:
        self._writer = handler.wfile
        self._connection = handler.connection
        self._lock = threading.Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def send(self, opcode: int, payload: bytes) -> bool:
        with self._lock:
            if self._closed:
                return False
            try:
                self._writer.write(_frame(opcode, payload))
                self._writer.flush()
            except OSError:
                self._closed = True
                return False
            return True

    def send_text(self, text: str) -> bool:
        return self.send(0x1, text.encode("utf-8"))

    def ping(self) -> bool:
        return self.send(0x9, b"")

    def close(self) -> None:
        if not self._closed:
            self.send(0x8, b"")
        self._closed = True

    def read(self) -> tuple[int, bytes] | None:
        return _read_frame(self._connection)


# ---------------------------------------------------------------------------
# The hub every part of the console rings
# ---------------------------------------------------------------------------


class PanelStream:
    """The set of live displays, and the one call that tells them something moved."""

    def __init__(self) -> None:
        self._clients: set[Client] = set()
        self._lock = threading.Lock()

    def count(self) -> int:
        with self._lock:
            return len(self._clients)

    def broadcast(self, event: str, **extra: Any) -> int:
        """Send one event to every live display; drop the ones that did not take it.

        The payload is deliberately small: the display reads the state it
        renders from the console's own HTTP routes, so this is only ever the
        nudge that says "read it now". A count is returned for the tests and the
        console tab.
        """
        message = json.dumps(
            {"type": "event", "event": str(event or "change"), "at": time.time(), **extra},
            ensure_ascii=False,
            default=str,
        )
        with self._lock:
            clients = tuple(self._clients)
        delivered = 0
        for client in clients:
            if client.send_text(message):
                delivered += 1
            else:
                self.discard(client)
        return delivered

    def discard(self, client: Client) -> None:
        with self._lock:
            self._clients.discard(client)
        client.close()

    def serve(self, handler: Any) -> None:
        """Upgrade a handler's connection and stay on it until the display leaves.

        Called from ``do_GET`` when a panel asks for the socket path. It owns the
        connection from here: the request loop is finished, and this method
        returns only once the client has closed or gone quiet past the idle
        window, which is what the surrounding HTTP handler then tears down.
        """
        try:
            _handshake(handler)
        except OSError:
            return
        client = Client(handler)
        with self._lock:
            self._clients.add(client)
        try:
            self._pump(client)
        except OSError:
            pass
        except Exception:  # noqa: BLE001 - a broken display must not take the wall's server down
            pass
        finally:
            with self._lock:
                self._clients.discard(client)
            client.close()

    def _pump(self, client: Client) -> None:
        handler_socket = client._connection
        try:
            handler_socket.settimeout(IDLE_SECONDS)
        except OSError:
            pass
        # The page's first message is not required: a display may sit and listen.
        welcome = json.dumps({"type": "hello", "socket": SOCKET_PATH, "at": time.time()})
        client.send_text(welcome)
        while not client.closed:
            try:
                frame = client.read()
            except socket.timeout:
                if not client.ping():
                    return
                continue
            if frame is None:
                return
            opcode, payload = frame
            if opcode == 0x8:                     # close
                return
            if opcode == 0x9:                     # ping -> pong
                client.send(0xA, payload)
            elif opcode == 0x1:                   # a text message from the page
                self._heard(client, payload)
        return

    def _heard(self, client: Client, payload: bytes) -> None:
        """What the page may say on the channel. Nothing it says changes state."""
        try:
            message = json.loads(payload.decode("utf-8") or "{}")
        except (UnicodeDecodeError, ValueError):
            return
        if not isinstance(message, dict):
            return
        if message.get("type") in {"ping", "hello"}:
            client.send_text(json.dumps({"type": "pong", "at": time.time()}))


# The console's one hub. ``music_api`` and ``puca_panel`` import this module, not
# each other, and both ring the same set of displays.
HUB = PanelStream()


def notify(event: str, **extra: Any) -> int:
    """Tell every live display that something moved. Returns how many were told."""
    return HUB.broadcast(event, **extra)
