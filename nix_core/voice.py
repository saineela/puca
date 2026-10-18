"""Speech to text on the console itself.

The Music page and the PUCA Panel both offer a microphone, and both used to hand
the recognition to the browser's own speech service (`SpeechRecognition`). That
works only in Chrome, only with Google reachable, and — the part that made the
wall display's mic do nothing at all — only on a *secure origin*: over plain
http on a LAN address every browser refuses the microphone before recognition is
even attempted. The panel is opened at ``http://<hub-ip>:49170/``, so there was
nothing to fall back to.

This module is the console's own recogniser instead. The page records a few
seconds of speech through the browser's microphone, posts the samples here, and
the box transcribes them locally with Whisper:

* nothing is uploaded to a speech service — the audio is decoded and recognised
  on this machine, and discarded when the answer is written;
* the microphone still has to come from the browser, so the page has to be on a
  secure origin (see ``puca_panel``'s https listener) — this module removes the
  *service*, not that browser rule;
* the model is loaded on first use (and warmed in the background when the panel
  asks whether the console can hear), so the first spoken request is not the one
  that pays for the weights.

The audio is a plain WAV (16-bit PCM, mono or stereo) because that is what the
browser can build without an encoder and the standard library can read without
an external tool. A clip is a few hundred kilobytes and a couple of seconds:
small enough that the request can carry it whole, which is why there is no job,
no upload id and no second round trip.
"""
from __future__ import annotations

import io
import os
import threading
import time
import wave
from typing import Any

# Ceilings. A spoken command is a phrase, not a recording: 30 seconds is well
# past anything the panel will send, and the browser stops on its own before
# that. The byte cap is the same idea for a malformed or hostile body.
MAX_SECONDS = 30
MAX_AUDIO_BYTES = 8 * 1024 * 1024
# Whisper listens at 16 kHz. The browser records at whatever the microphone's
# own rate is (48 kHz, usually), so the clip is resampled here rather than left
# to the model's pipeline: doing it ourselves keeps the job on numpy and scipy,
# which are already installed, instead of pulling in torchaudio for one step.
TARGET_RATE = 16000
# Below this the clip is the room, not a person: Whisper happily transcribes a
# silent file into "Thank you.", which is worse than saying nothing was heard.
SILENCE_RMS = 0.0016

MODEL_ENV = "NIX_STT_MODEL"
DEVICE_ENV = "NIX_STT_DEVICE"
LANGUAGE_ENV = "NIX_STT_LANGUAGE"
DEFAULT_MODEL = "openai/whisper-small.en"
DEFAULT_LANGUAGE = "en"


class VoiceError(Exception):
    """A bounded, user-facing failure with an HTTP status and a stable code."""

    def __init__(self, message: str, status: int = 400, code: str = "voice_error"):
        super().__init__(message)
        self.status = status
        self.code = code


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def model_name() -> str:
    return str(os.environ.get(MODEL_ENV) or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def language() -> str:
    return str(os.environ.get(LANGUAGE_ENV) or DEFAULT_LANGUAGE).strip().lower() or DEFAULT_LANGUAGE


def device_name() -> str:
    """``cuda`` when this machine has a GPU with room for it, else ``cpu``.

    The console already keeps its chat model in this process, so the graphics
    card may be nearly full. A recogniser that has to swap the chat model out
    would make both slower, so a card without room for ~2 GB is left alone and
    the transcription runs on the processor instead.
    """
    wanted = str(os.environ.get(DEVICE_ENV) or "auto").strip().lower()
    if wanted in {"cpu", "cuda"}:
        return wanted
    try:
        import torch
    except Exception:
        return "cpu"
    try:
        if not torch.cuda.is_available():
            return "cpu"
        free, _total = torch.cuda.mem_get_info()
        return "cuda" if free >= 2 * 1024**3 else "cpu"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


_lock = threading.RLock()
_recognizer: Any = None
_recognizer_key: tuple[str, str] | None = None
_loading = threading.Event()
_status: dict[str, Any] = {"loaded": False, "error": None}


def _build() -> Any:
    """Load the pipeline once, for the model and device configured now."""
    global _recognizer, _recognizer_key
    name, device = model_name(), device_name()
    key = (name, device)
    with _lock:
        if _recognizer is not None and _recognizer_key == key:
            return _recognizer
        try:
            import torch
            from transformers import pipeline
        except Exception as exc:  # transformers/torch missing from this build
            raise VoiceError(
                "The console has no local speech recogniser installed "
                f"({type(exc).__name__}).",
                501,
                "no_stt_engine",
            ) from exc
        kwargs: dict[str, Any] = {"device": 0 if device == "cuda" else -1}
        if device == "cuda":
            kwargs["dtype"] = torch.float16
        try:
            recognizer = pipeline("automatic-speech-recognition", model=name, **kwargs)
        except TypeError:
            # Older/newer transformers spell the precision differently.
            kwargs.pop("dtype", None)
            kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
            recognizer = pipeline("automatic-speech-recognition", model=name, **kwargs)
        except Exception as exc:
            raise VoiceError(
                f"The speech model {name} could not be loaded: {type(exc).__name__}: {str(exc)[:160]}",
                503,
                "stt_model_unavailable",
            ) from exc
        _recognizer = recognizer
        _recognizer_key = key
        _status["loaded"] = True
        _status["error"] = None
        return recognizer


def warm() -> dict[str, Any]:
    """Load the model, on this thread. Used by the console's own warm-up."""
    try:
        _build()
        return {"ok": True, **available()}
    except VoiceError as exc:
        _status["error"] = str(exc)
        return {"ok": False, "error": str(exc), "code": exc.code, **available()}


def warm_in_background() -> None:
    """Start loading without making the caller wait for the weights.

    A display asks whether the console can hear when its page loads; by the time
    somebody taps the microphone, the model is usually already in memory.
    """
    if _recognizer is not None or _loading.is_set():
        return
    _loading.set()

    def run() -> None:
        try:
            _build()
        except VoiceError as exc:
            _status["error"] = str(exc)
        finally:
            _loading.clear()

    threading.Thread(target=run, name="nix-stt-warmup", daemon=True).start()


def available() -> dict[str, Any]:
    """What the pages report about the console's own hearing."""
    return {
        "engine": "whisper",
        "model": model_name(),
        "device": device_name(),
        "language": language(),
        "loaded": _status["loaded"] and _recognizer is not None,
        "loading": _loading.is_set(),
        "max_seconds": MAX_SECONDS,
        "error": _status["error"],
    }


# ---------------------------------------------------------------------------
# Audio in
# ---------------------------------------------------------------------------


def decode_wav(payload: bytes) -> tuple[Any, int, float]:
    """Read 16-bit PCM WAV bytes into (float samples, rate, seconds).

    The browser writes this itself with the Web Audio API, so only the two
    shapes it can produce are accepted: 16-bit PCM, one or two channels.
    """
    if not payload:
        raise VoiceError("Send the recorded audio in the request body.", 400, "no_audio")
    if len(payload) > MAX_AUDIO_BYTES:
        raise VoiceError(
            f"That clip is larger than the {MAX_AUDIO_BYTES // (1024 * 1024)} MB the console accepts.",
            413,
            "audio_too_large",
        )
    try:
        with wave.open(io.BytesIO(payload), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.getnframes()
            raw = handle.readframes(frames)
    except wave.Error as exc:
        raise VoiceError(
            "That audio is not a readable WAV. Record it again.", 400, "audio_unreadable"
        ) from exc
    if width != 2:
        raise VoiceError("That audio is not 16-bit PCM.", 400, "audio_unreadable")
    if channels not in (1, 2):
        raise VoiceError("That audio has too many channels.", 400, "audio_unreadable")
    if rate <= 0 or not raw:
        raise VoiceError("That clip is empty.", 400, "no_audio")
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - numpy ships with the model
        raise VoiceError("The console cannot read audio samples.", 501, "no_stt_engine") from exc
    samples = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    seconds = len(samples) / float(rate)
    if seconds > MAX_SECONDS:
        samples = samples[: int(MAX_SECONDS * rate)]
        seconds = MAX_SECONDS
    return samples, rate, seconds


def _resample(samples: Any, rate: int, target: int = TARGET_RATE) -> tuple[Any, int]:
    """Bring a clip to the rate the recogniser listens at."""
    if rate == target or not rate:
        return samples, rate or target
    try:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(rate), int(target))
        resampled = resample_poly(samples, target // divisor, int(rate) // divisor)
        return resampled.astype("float32"), target
    except Exception:  # scipy missing: interpolate, which is good enough for speech
        import numpy as np

        count = int(round(len(samples) * target / float(rate)))
        if count <= 1:
            return samples[:0], target
        positions = np.linspace(0, len(samples) - 1, count)
        return np.interp(positions, np.arange(len(samples)), samples).astype("float32"), target


def _rms(samples: Any) -> float:
    try:
        import numpy as np

        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(samples))))
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Turn-taking (D7)
# ---------------------------------------------------------------------------

#: Frames the energy VAD is built from. 20 ms is the usual speech frame: short
#: enough to see a stop, long enough that a single click is not a word.
VAD_FRAME_MS = 20


def vad_frames(samples: Any, rate: int, *, frame_ms: int = VAD_FRAME_MS) -> list[tuple[float, bool]]:
    """Energy VAD frames as ``(at_ms, speech)`` pairs.

    The threshold is not a fixed number: the quiet tenth of the clip is taken
    as the room and every frame is judged against a floor above it, so a
    recording made on a noisier box does not read as continuous speech. The
    absolute ``SILENCE_RMS`` is still a floor, so a dead-silent clip stays
    silent.
    """
    import numpy as np

    step = max(1, int(rate * frame_ms / 1000))
    energies: list[float] = []
    for start in range(0, len(samples), step):
        chunk = samples[start:start + step]
        if chunk.size == 0:
            break
        energies.append(float(np.sqrt(np.mean(np.square(chunk)))))
    if not energies:
        return []
    floor = max(SILENCE_RMS, float(np.percentile(energies, 10)) * 2.0)
    return [
        (round(index * frame_ms, 1), energy >= floor)
        for index, energy in enumerate(energies)
    ]


def _last_speech_ms(frames: list[tuple[float, bool]]) -> float | None:
    return next((at for at, speech in reversed(frames) if speech), None)


def turn_taking_profile(payload: bytes, *, endpoint_ms: float = 300.0) -> dict[str, Any]:
    """Measure one recorded clip's turn-taking: VAD, endpointing, trailing silence.

    This is the console's own measurement, taken from the same clip the local
    recogniser is about to read, so the numbers are for this box and this
    microphone rather than a reference machine. Barge-in needs two audio
    streams at once and is not measurable from one upload; it is reported as
    unmeasured here instead of being guessed.
    """
    from eval.turn_taking import Frame, TurnTaker

    samples, rate, seconds = decode_wav(payload)
    frames = vad_frames(samples, rate)
    taker = TurnTaker(endpoint_ms=endpoint_ms)
    for at, speech in frames:
        taker.feed(Frame(at, speech))
    summary = taker.summary()
    last_speech = _last_speech_ms(frames)
    # The clip's own length, not the last frame's start: the frame that carries
    # the final word starts one frame before the speech actually stops.
    audio_ms = (len(samples) / rate * 1000.0) if rate else 0.0
    trailing = (
        None
        if last_speech is None
        else round(max(0.0, audio_ms - last_speech - VAD_FRAME_MS), 1)
    )
    return {
        "frames": len(frames),
        "frame_ms": VAD_FRAME_MS,
        "speech_frames": sum(1 for _at, speech in frames if speech),
        "endpoint_ms": endpoint_ms,
        "end_of_turn_p50_ms": summary["end_of_turn_p50_ms"],
        "end_of_turn_p95_ms": summary["end_of_turn_p95_ms"],
        "trailing_silence_ms": trailing,
        "endpoint_reached": bool(trailing is not None and trailing >= endpoint_ms),
        "barge_ins": None,
        "barge_in_note": "not measurable from a single upload",
        "audio_seconds": round(seconds, 2),
        "model_delay_ms": None,
    }


# ---------------------------------------------------------------------------
# Recognise
# ---------------------------------------------------------------------------


def transcribe(payload: bytes, *, wanted_language: str | None = None) -> dict[str, Any]:
    """Turn recorded speech into text, on this machine."""
    samples, rate, seconds = decode_wav(payload)
    profile: dict[str, Any] | None = None
    try:
        profile = turn_taking_profile(payload)
    except Exception:
        # A measurement that cannot be taken must not stop the transcription
        # the user actually asked for.
        profile = None
    if _rms(samples) < SILENCE_RMS:
        return {
            "text": "", "seconds": round(seconds, 2), "silent": True,
            "turn_taking": profile, **available(),
        }
    samples, rate = _resample(samples, rate)
    recognizer = _build()
    # An English-only checkpoint refuses to be told a task or a language at all
    # (it has neither to choose), so only a multilingual one is given them.
    settings: dict[str, Any] = {}
    if not model_name().endswith(".en"):
        settings["task"] = "transcribe"
        language_hint = (wanted_language or language()).strip().lower()
        if language_hint:
            settings["language"] = language_hint
    started = time.time()
    try:
        result = recognizer({"array": samples, "sampling_rate": rate}, generate_kwargs=settings)
    except Exception as exc:
        raise VoiceError(
            f"The speech model could not read that clip: {type(exc).__name__}: {str(exc)[:160]}",
            500,
            "stt_failed",
        ) from exc
    text = str((result or {}).get("text") or "").strip()
    if profile is not None:
        # The split the review asks for: how much of the wait was the audio
        # endpoint and how much was the model, not one combined latency.
        profile["model_delay_ms"] = round((time.time() - started) * 1000, 1)
        _remember_turn_taking(profile)
    return {
        "text": text,
        "seconds": round(seconds, 2),
        "took_seconds": round(time.time() - started, 2),
        "silent": not text,
        "turn_taking": profile,
        **available(),
    }


_turn_taking_lock = threading.Lock()
_last_turn_taking: dict[str, Any] | None = None


def _remember_turn_taking(profile: dict[str, Any]) -> None:
    global _last_turn_taking
    with _turn_taking_lock:
        _last_turn_taking = dict(profile)


def last_turn_taking() -> dict[str, Any] | None:
    """The most recent clip's turn-taking measurement, or None if none was taken."""
    with _turn_taking_lock:
        return dict(_last_turn_taking) if _last_turn_taking else None
