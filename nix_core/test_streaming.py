"""Token streaming contracts: sink, windows, transforms, first-token timing."""
from __future__ import annotations

import re
import time

import streaming
from streaming import TokenSink


def _drain(sink: TokenSink) -> list[dict]:
    events = []
    for event in sink.events(poll_timeout=0.05):
        events.append(event)
        if event.get("type") in {"final", "error"}:
            break
    return events


def test_emit_is_silent_without_a_sink():
    # No sink attached: model calls must behave exactly as before.
    streaming.emit_token("hello")  # must not raise


def test_window_gates_emission():
    sink = TokenSink()
    token = streaming.attach(sink)
    try:
        # Outside a window: nothing streams.
        assert streaming.active_sink() is None
        streaming.emit_token("internal text")
        assert sink.streamed_chars == 0

        with streaming.user_facing():
            assert streaming.active_sink() is sink
            streaming.emit_token("visible ")
            streaming.emit_token("text")
        assert sink.streamed_chars == len("visible text")
    finally:
        streaming.detach(token)

    sink.push_event({"type": "final"})
    sink.close()
    kinds = [e["type"] for e in _drain(sink)]
    assert "token" in kinds
    assert kinds[-1] == "final"


def test_first_token_latency_is_measured_from_sink_start():
    sink = TokenSink()
    time.sleep(0.05)
    sink.push_token("first")
    sink.push_token("second")
    assert sink.first_token_ms is not None
    assert sink.first_token_ms >= 50.0
    # Second token must not move the first-token clock.
    first = sink.first_token_ms
    time.sleep(0.02)
    sink.push_token("third")
    assert sink.first_token_ms == first


def test_reopen_window_resets_provisional_text():
    sink = TokenSink()
    token = streaming.attach(sink)
    try:
        with streaming.user_facing():
            streaming.emit_token("wrong start ")
        with streaming.user_facing():
            streaming.emit_token("corrected")
        assert sink.streamed_chars == len("corrected")
    finally:
        streaming.detach(token)
    sink.push_event({"type": "final"})
    sink.close()
    events = _drain(sink)
    resets = [i for i, e in enumerate(events) if e["type"] == "reset"]
    assert len(resets) == 1
    # Reset must come after the wrong tokens and before the corrected ones.
    assert resets[0] > 0


def test_transform_restores_word_case_across_token_boundaries():
    sink = TokenSink()
    token = streaming.attach(sink)
    upper = str.upper  # transform that would SHOUT - grounding must undo it

    def restore(text: str) -> str:
        # Word-aware, like _restore_grounding_casing: whole-word replace so a
        # chunk like "DINESH is " (word + trailing context) still corrects.
        return re.sub(
            r"\bDINESH\b",
            "Dinesh",
            text,
        )

    try:
        with streaming.user_facing(transform=restore):
            # The shout word arrives split across decode steps.
            streaming.emit_token("Your coworker DIN")
            streaming.emit_token("ESH is here")
        streaming.emit_token("")  # no-op
    finally:
        streaming.detach(token)
    sink.push_event({"type": "final"})
    sink.close()
    text = "".join(
        e.get("text", "") for e in _drain(sink) if e["type"] == "token"
    )
    assert "Dinesh" in text
    assert "DINESH" not in text


def test_events_ordered_tokens_before_final_and_close_ends_stream():
    sink = TokenSink()
    sink.push_token("a")
    sink.push_token("b")
    sink.push_event({"type": "final", "result": {"ok": True}})
    sink.close()
    events = [e["type"] for e in _drain(sink)]
    assert events == ["token", "token", "final"]


def test_heartbeat_while_idle_then_close():
    sink = TokenSink()
    sink.close()
    # Queue is empty and closed: first get returns None immediately,
    # so no heartbeat precedes the end.
    events = list(sink.events(poll_timeout=0.05))
    assert events == []
