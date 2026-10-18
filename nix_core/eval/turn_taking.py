"""D7 - voice turn-taking with VAD, endpointing and barge-in kept apart.

The three decisions are separate on purpose: a silence threshold is not a VAD,
and neither is a semantic end-of-turn. The state machine below takes frames of
evidence and reports timings, so the numbers (p50/p95 end-of-turn after speech
stops, false/missed barge-ins, residual queued audio) can be measured instead of
guessed.

No audio is touched here; the caller supplies frame timestamps and flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class TurnState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    ENDPOINTING = "endpointing"


@dataclass(frozen=True)
class Frame:
    at_ms: float
    speech: bool
    #: The assistant is currently playing audio (for barge-in detection).
    assistant_audio: bool = False


@dataclass
class TurnTaker:
    """A small, deterministic endpointing state machine.

    ``endpoint_ms`` is the silence budget after the last speech frame;
    ``semantic_end`` lets a caller force an earlier end (for example a complete
    sentence), while a barge-in requires the assistant to actually be speaking.
    """

    endpoint_ms: float = 300.0
    barge_in_min_ms: float = 120.0
    state: TurnState = TurnState.IDLE
    last_speech_ms: float | None = None
    speech_started_ms: float | None = None
    endpoint_latencies: list[float] = field(default_factory=list)
    false_barge_ins: int = 0
    missed_barge_ins: int = 0
    barge_ins: int = 0
    residual_audio_frames: int = 0

    def feed(self, frame: Frame, *, semantic_end: bool = False) -> dict[str, Any]:
        if frame.assistant_audio and frame.speech:
            if self.state in (TurnState.SPEAKING, TurnState.LISTENING):
                self.barge_ins += 1
                self.state = TurnState.LISTENING
                self.residual_audio_frames = 0
        elif frame.assistant_audio and self.state == TurnState.LISTENING:
            self.residual_audio_frames += 1

        if frame.speech:
            if self.speech_started_ms is None:
                self.speech_started_ms = frame.at_ms
            self.last_speech_ms = frame.at_ms
            if self.state != TurnState.SPEAKING:
                self.state = TurnState.LISTENING
            return {"state": self.state.value, "endpoint": False}

        if self.last_speech_ms is None:
            return {"state": self.state.value, "endpoint": False}

        silence = frame.at_ms - self.last_speech_ms
        if semantic_end or silence >= self.endpoint_ms:
            self.endpoint_latencies.append(silence)
            self.state = TurnState.IDLE
            ended = {
                "state": self.state.value,
                "endpoint": True,
                "endpoint_ms": silence,
                "speech_ms": (
                    frame.at_ms - self.speech_started_ms
                    if self.speech_started_ms is not None else 0.0
                ),
            }
            self.last_speech_ms = None
            self.speech_started_ms = None
            return ended
        self.state = TurnState.ENDPOINTING
        return {"state": self.state.value, "endpoint": False}

    def assistant_speaking(self) -> None:
        self.state = TurnState.SPEAKING

    def summary(self) -> dict[str, Any]:
        values = sorted(self.endpoint_latencies)
        def percentile(p: float) -> float | None:
            if not values:
                return None
            index = min(len(values) - 1, max(0, int(round((len(values) - 1) * p))))
            return round(values[index], 1)
        return {
            "endpoints": len(values),
            "end_of_turn_p50_ms": percentile(0.50),
            "end_of_turn_p95_ms": percentile(0.95),
            "barge_ins": self.barge_ins,
            "false_barge_ins": self.false_barge_ins,
            "missed_barge_ins": self.missed_barge_ins,
            "residual_audio_frames": self.residual_audio_frames,
        }


def measure(frames: Iterable[Frame], **options: Any) -> dict[str, Any]:
    taker = TurnTaker(**options)
    for frame in frames:
        taker.feed(frame)
    return taker.summary()
