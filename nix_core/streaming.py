"""Token streaming boundary for Core.

The model clients (Luna Transformers, Ollama) can emit text as it is
generated instead of only after the whole reply exists.  This module owns
the plumbing:

- A *sink* is attached via a contextvar so every model call inside one
  request can push tokens without threading a callback through Brain.
- ``user_facing()`` windows mark WHICH model calls may stream.  Internal
  calls (skill planner, cleanser, verification retries) never open a
  window, so their output can never leak to the client.
- The sink records first-token latency (``first_token_ms``) measured from
  request start, plus per-event queueing for SSE writers.
- Streamed tokens are provisional.  Brain's authoritative reply still
  crosses the normal return path; the ``final`` event carries it so a
  client can reconcile any drift (e.g. post-hoc casing normalization).

Thread model: the sink contextvar is set inside the thread that runs
``Brain.handle``; token consumption for Transformers streaming also happens
in that thread (TextIteratorStreamer queue), so no cross-thread contextvar
access occurs.
"""
from __future__ import annotations

import contextvars
import queue
import threading
import time
from typing import Any, Callable, Iterator

__all__ = [
    "TokenSink",
    "active_sink",
    "attached_sink",
    "user_facing",
    "emit_token",
    "attach",
    "detach",
]

# (sink, allowed) - both must be set for a model call to stream.
_token_sink: contextvars.ContextVar["TokenSink | None"] = contextvars.ContextVar(
    "nix_token_sink", default=None
)
_stream_allowed: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "nix_stream_allowed", default=False
)


class TokenSink:
    """Queue of stream events for one request."""

    def __init__(self) -> None:
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._lock = threading.Lock()
        self.started_at = time.perf_counter()
        self.first_token_at: float | None = None
        self.streamed_chars = 0
        # Optional word-boundary transform (e.g. grounding-casing restore).
        # Buffered so a shouting token split across decode steps still gets
        # corrected as one word.
        self._transform: Callable[[str], str] | None = None
        self._buffer = ""

    # -- producers (model side) ----------------------------------------
    def set_transform(self, fn: Callable[[str], str] | None) -> None:
        self._flush_buffer()
        self._transform = fn

    def push_token(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            if self.first_token_at is None:
                self.first_token_at = time.perf_counter()
            self.streamed_chars += len(text)
        if self._transform is None:
            self._queue.put({"type": "token", "text": text})
            return
        self._buffer += text
        cut = self._buffer.rfind(" ")
        if cut == -1:
            return  # incomplete word yet
        head, self._buffer = self._buffer[: cut + 1], self._buffer[cut + 1 :]
        self._queue.put({"type": "token", "text": self._transform(head)})

    def _flush_buffer(self) -> None:
        if self._buffer:
            text = self._buffer
            self._buffer = ""
            self._queue.put(
                {"type": "token", "text": self._transform(text) if self._transform else text}
            )

    def push_event(self, event: dict[str, Any]) -> None:
        # Events (reset/final) must not jump ahead of buffered tokens.
        self._flush_buffer()
        self._queue.put(event)

    def reset_stream(self) -> None:
        """Discard first-token state after a provisional-text reset."""
        self._buffer = ""
        with self._lock:
            self.first_token_at = None
            self.streamed_chars = 0

    def close(self) -> None:
        self._flush_buffer()
        self._queue.put(None)

    # -- consumer (SSE writer) -----------------------------------------
    def events(self, poll_timeout: float = 0.25) -> Iterator[dict[str, Any]]:
        """Yield events until the sink is closed."""
        while True:
            try:
                item = self._queue.get(timeout=poll_timeout)
            except queue.Empty:
                yield {"type": "heartbeat"}
                continue
            if item is None:
                return
            yield item

    # -- metrics ---------------------------------------------------------
    @property
    def first_token_ms(self) -> float | None:
        if self.first_token_at is None:
            return None
        return round((self.first_token_at - self.started_at) * 1000, 1)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000, 1)


def active_sink() -> TokenSink | None:
    """Sink for this request, but only inside a user-facing window."""
    if not _stream_allowed.get():
        return None
    return _token_sink.get()


def attached_sink() -> TokenSink | None:
    """Sink for this request regardless of window state.

    Cleanup paths (e.g. discarding an internal envelope that was streamed
    by mistake) must reach the sink even outside a user-facing window.
    """
    return _token_sink.get()


def attach(sink: TokenSink) -> contextvars.Token:
    """Bind a sink to the current thread's context (run before Brain.handle)."""
    return _token_sink.set(sink)


def detach(token: contextvars.Token) -> None:
    _token_sink.reset(token)


class _UserFacing:
    """Context manager: allow streaming while a sink is attached.

    Re-opening the window after tokens were already emitted (retry paths)
    pushes a ``reset`` event so the client discards provisional text.
    """

    def __init__(self, transform: Callable[[str], str] | None = None) -> None:
        self._token: contextvars.Token | None = None
        self._transform = transform

    def __enter__(self) -> "_UserFacing":
        self._token = _stream_allowed.set(True)
        sink = _token_sink.get()
        if sink is not None:
            if sink.streamed_chars > 0:
                sink.push_event({"type": "reset"})
                sink.reset_stream()
            sink.set_transform(self._transform)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _stream_allowed.reset(self._token)
        sink = _token_sink.get()
        if sink is not None:
            # Flush any trailing partial word and detach this window's
            # transform so later internal windows start clean.
            sink.set_transform(None)


def user_facing(transform: Callable[[str], str] | None = None) -> _UserFacing:
    return _UserFacing(transform)


def emit_token(text: str) -> None:
    """Model-client hook: forward a generated piece if streaming is on."""
    if not text:
        return
    sink = active_sink()
    if sink is not None:
        sink.push_token(text)
