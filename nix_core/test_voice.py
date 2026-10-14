"""Tests for the console's own speech recogniser.

The weights are not loaded here: recognising real speech is checked live with
clips synthesized on the machine, and what these tests pin is everything around
it — the WAV the browser writes, the ceilings, what a silent clip answers, what
the model is told to expect, and the two HTTP routes on both origins the pages
use. The engine itself is a thin layer over transformers, so it is monkeypatched
where a route is being tested rather than made to download a model in CI.
"""
from __future__ import annotations

import io
import json
import math
import ssl
import struct
import urllib.error
import urllib.parse
import urllib.request
import wave
from http.server import ThreadingHTTPServer
from threading import Thread

import pytest

import console_extend
import puca_panel
import voice
import voice_api

numpy = pytest.importorskip("numpy")


# ---------------------------------------------------------------------------
# Helpers: a WAV, the way the page writes one
# ---------------------------------------------------------------------------


def wav_bytes(seconds: float = 1.0, rate: int = 48000, channels: int = 1, amplitude: float = 0.3) -> bytes:
    """16-bit PCM WAV, as Web Audio hands it to the console."""
    frames = int(seconds * rate)
    samples = []
    for index in range(frames):
        value = int(amplitude * 32767 * math.sin(2 * math.pi * 220 * index / rate))
        for _channel in range(channels):
            samples.append(value)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<%dh" % len(samples), *samples))
    return buffer.getvalue()


def serve(handler_class=console_extend.ExtendedHandler) -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
    Thread(target=server.serve_forever, daemon=True).start()
    return server, int(server.server_address[1])


def get_json(port: int, path: str, *, scheme: str = "http", context=None) -> tuple[int, dict]:
    url = f"{scheme}://127.0.0.1:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=30, context=context) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def post_audio(port: int, path: str, payload: bytes, *, content_type: str = "audio/wav") -> tuple[int, dict]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=payload, headers={"Content-Type": content_type}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


# ---------------------------------------------------------------------------
# The clip
# ---------------------------------------------------------------------------


def test_the_clip_the_page_records_is_read_as_samples():
    payload = wav_bytes(seconds=0.5, rate=48000)
    samples, rate, seconds = voice.decode_wav(payload)
    assert rate == 48000
    assert samples.dtype == numpy.float32
    assert abs(seconds - 0.5) < 0.01
    assert len(samples) == int(0.5 * 48000)
    assert 0.25 < float(numpy.abs(samples).max()) <= 0.31, "the waveform survives the trip"


def test_a_stereo_clip_is_mixed_down_to_one_channel():
    left, right = wav_bytes(seconds=0.1, amplitude=0.2), wav_bytes(seconds=0.1, amplitude=0.6)
    stereo = _interleave(left, right)
    samples, rate, seconds = voice.decode_wav(stereo)
    assert samples.ndim == 1, "the model hears one channel"
    assert abs(seconds - 0.1) < 0.01


def _interleave(left: bytes, right: bytes) -> bytes:
    """Two mono WAVs of the same length, as one stereo clip."""
    with wave.open(io.BytesIO(left)) as handle:
        one = handle.readframes(handle.getnframes())
        rate = handle.getframerate()
    with wave.open(io.BytesIO(right)) as handle:
        two = handle.readframes(handle.getnframes())
    first = struct.unpack("<%dh" % (len(one) // 2), one)
    second = struct.unpack("<%dh" % (len(two) // 2), two)
    mixed = [value for pair in zip(first, second) for value in pair]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<%dh" % len(mixed), *mixed))
    return buffer.getvalue()


def test_a_clip_is_brought_to_the_rate_the_model_listens_at():
    # Whisper listens at 16 kHz and the pipeline refuses to resample without
    # torchaudio, so the console brings the clip down itself (numpy and scipy).
    samples, rate, _seconds = voice.decode_wav(wav_bytes(seconds=1.0, rate=48000))
    resampled, new_rate = voice._resample(samples, rate)
    assert new_rate == voice.TARGET_RATE == 16000
    assert len(resampled) == pytest.approx(16000, abs=40), "one second stays one second"


def test_the_clip_is_capped_where_a_command_ends():
    samples, rate, seconds = voice.decode_wav(wav_bytes(seconds=voice.MAX_SECONDS + 5))
    assert seconds == voice.MAX_SECONDS
    assert len(samples) == voice.MAX_SECONDS * rate


# ---------------------------------------------------------------------------
# What it refuses, and what it answers without the model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload,code,status",
    [
        (b"", "no_audio", 400),
        (b"these are not samples", "audio_unreadable", 400),
        (b"RIFF" + b"\x00" * 40, "audio_unreadable", 400),
    ],
)
def test_audio_it_cannot_read_is_reported_as_such(payload, code, status):
    with pytest.raises(voice.VoiceError) as raised:
        voice.decode_wav(payload)
    assert raised.value.code == code
    assert raised.value.status == status


def test_a_clip_larger_than_the_ceiling_is_refused_before_it_is_read():
    with pytest.raises(voice.VoiceError) as raised:
        voice.decode_wav(b"\x00" * (voice.MAX_AUDIO_BYTES + 1))
    assert raised.value.code == "audio_too_large"
    assert raised.value.status == 413


def test_a_silent_clip_is_answered_without_loading_the_model(monkeypatch):
    monkeypatch.setattr(voice, "_recognizer", None)
    quiet = wav_bytes(seconds=1.0, amplitude=0.0002)
    answer = voice.transcribe(quiet)
    assert answer["text"] == ""
    assert answer["silent"] is True
    assert voice._recognizer is None, "the weights stay unloaded for a room that said nothing"


def test_the_model_and_device_come_from_the_environment(monkeypatch):
    monkeypatch.setenv(voice.MODEL_ENV, "openai/whisper-base.en")
    monkeypatch.setenv(voice.DEVICE_ENV, "cpu")
    monkeypatch.setenv(voice.LANGUAGE_ENV, "EN-GB")
    reported = voice.available()
    assert reported["model"] == "openai/whisper-base.en"
    assert reported["device"] == "cpu"
    assert reported["language"] == "en-gb"


def test_a_silent_clip_answers_the_words_it_can(monkeypatch):
    heard = {}

    class FakeRecognizer:
        def __call__(self, payload, generate_kwargs=None):
            heard["sampling_rate"] = payload["sampling_rate"]
            heard["generate_kwargs"] = generate_kwargs
            return {"text": " Play Bohemian Rhapsody. "}

    monkeypatch.setattr(voice, "_build", lambda: FakeRecognizer())
    monkeypatch.setattr(voice, "model_name", lambda: "openai/whisper-small.en")
    answer = voice.transcribe(wav_bytes(seconds=1.0, rate=48000))
    assert answer["text"] == "Play Bohemian Rhapsody."
    assert heard["sampling_rate"] == voice.TARGET_RATE, "the model is handed 16 kHz"
    assert heard["generate_kwargs"] == {}, "an English-only model is told neither task nor language"


# ---------------------------------------------------------------------------
# The routes
# ---------------------------------------------------------------------------


def test_the_console_says_whether_it_can_hear(monkeypatch):
    started = []
    monkeypatch.setattr(voice, "warm_in_background", lambda: started.append(True))
    server, port = serve()
    try:
        status, payload = get_json(port, "/api/voice")
    finally:
        server.shutdown()
        server.server_close()
    assert status == 200 and payload["ok"] is True
    assert payload["engine"] == "whisper"
    assert payload["available"] is True
    assert payload["max_seconds"] == voice.MAX_SECONDS
    assert started == [True], "asking starts the model loading, so the first tap is not the slow one"


def test_a_recorded_clip_is_recognised_here(monkeypatch):
    monkeypatch.setattr(voice, "transcribe", lambda payload, **kwargs: {
        "text": "pause the music", "seconds": 2.0, "engine": "whisper", "device": "cuda",
    })
    server, port = serve()
    try:
        status, payload = post_audio(port, "/api/voice/transcribe", wav_bytes(seconds=0.2))
    finally:
        server.shutdown()
        server.server_close()
    assert status == 200
    assert payload["ok"] is True and payload["text"] == "pause the music"


def test_a_clip_the_engine_cannot_read_is_a_payload_not_a_500(monkeypatch):
    def explode(payload, **kwargs):
        raise voice.VoiceError("That audio is not a readable WAV.", 400, "audio_unreadable")

    monkeypatch.setattr(voice, "transcribe", explode)
    server, port = serve()
    try:
        status, payload = post_audio(port, "/api/voice/transcribe", b"nonsense")
    finally:
        server.shutdown()
        server.server_close()
    assert status == 400
    assert payload["ok"] is False and payload["code"] == "audio_unreadable"


def test_an_oversized_clip_is_refused_before_it_is_buffered():
    server, port = serve()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/voice/transcribe",
            data=b"\x00" * 1024,
            headers={"Content-Type": "audio/wav", "Content-Length": str(voice.MAX_AUDIO_BYTES + 1)},
        )
        try:
            urllib.request.urlopen(request, timeout=30)
            status, payload = 0, {}
        except urllib.error.HTTPError as exc:
            status, payload = exc.code, json.loads(exc.read() or b"{}")
    finally:
        server.shutdown()
        server.server_close()
    assert status == 413 and payload["code"] == "audio_too_large"


def test_the_panel_origin_hears_on_its_own(monkeypatch):
    """The page records on the panel's origin, so the answer has to come back there."""
    monkeypatch.setattr(voice, "transcribe", lambda payload, **kwargs: {
        "text": "next track", "seconds": 1.5, "engine": "whisper", "device": "cuda",
    })
    monkeypatch.setattr(voice, "warm_in_background", lambda: None)
    server = puca_panel.create_server("127.0.0.1", 0)
    port = int(server.server_address[1])
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, payload = get_json(port, "/api/voice")
        assert status == 200 and payload["engine"] == "whisper"
        status, payload = post_audio(port, "/api/voice/transcribe", wav_bytes(seconds=0.2))
        assert status == 200 and payload["text"] == "next track"
    finally:
        server.shutdown()
        server.server_close()
