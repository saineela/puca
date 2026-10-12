import asyncio
import base64
import enum
import http.server
import importlib.util
import json
import os
import subprocess
import sys
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest


WORKER_PATH = (
    Path(__file__).parent
    / "ring_light_package"
    / "skills"
    / "echo-connect"
    / "ring_light_worker.py"
)
SPEC = importlib.util.spec_from_file_location("nix_ring_light_worker_test", WORKER_PATH)
assert SPEC and SPEC.loader
worker_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker_module
SPEC.loader.exec_module(worker_module)


class FakeLightInfo:
    def __init__(self, key, object_id, name, effects=None):
        self.key = key
        self.object_id = object_id
        self.name = name
        self.effects = effects or []


class FakeFormat:
    def __init__(self, fmt, sample_rate, num_channels, purpose, sample_bytes):
        self.format = fmt
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.purpose = SimpleNamespace(name=purpose)
        self.sample_bytes = sample_bytes


class FakeMediaPlayerInfo:
    def __init__(self, key, object_id, name, supported_formats=None, supports_pause=False):
        self.key = key
        self.object_id = object_id
        self.name = name
        self.supported_formats = supported_formats or []
        self.supports_pause = supports_pause


class FakeSpeakerState:
    def __init__(self, key, volume, muted, state):
        self.key = key
        self.volume = volume
        self.muted = muted
        self.state = state


class FakeAPIClient:
    instances = []

    def __init__(self, address, port, *, noise_psk):
        self.address = address
        self.port = port
        self.noise_psk = noise_psk
        self.callback = None
        self.speaker = {"volume": 0.5, "muted": False, "state": "idle"}
        self.media_commands = []
        self.commands = []
        self.ring = {"state": False, "brightness": 0.5, "red": 0.0, "green": 0.0, "blue": 0.0, "effect": "None"}
        self.instances.append(self)

    async def connect(self, *, login):
        assert login is True

    async def device_info_and_list_entities(self):
        entities = [FakeLightInfo(1, "ring", "LED Ring", ["Rainbow", "Rainbow Twinkle"])]
        entities.extend(FakeLightInfo(100 + number, f"segment_{number}", f"Segment {number}") for number in range(1, 13))
        entities.append(FakeMediaPlayerInfo(
            300, "speaker", "Speaker",
            supported_formats=[
                FakeFormat("wav", 48000, 2, "DEFAULT", 2),
                FakeFormat("wav", 16000, 1, "ANNOUNCEMENT", 2),
            ],
            supports_pause=True,
        ))
        return object(), entities, []

    def media_player_command(self, key, **changes):
        self.media_commands.append((key, changes))
        # Accept both plain command strings and MediaPlayerCommand enum members.
        command = getattr(changes.get("command"), "name", changes.get("command"))
        if "volume" in changes:
            self.speaker["volume"] = float(changes["volume"])
            self.speaker["muted"] = False
        elif command == "MUTE":
            self.speaker["muted"] = True
        elif command == "UNMUTE":
            self.speaker["muted"] = False
        elif command == "STOP":
            self.speaker["state"] = "idle"
        elif command == "PLAY":
            self.speaker["state"] = "playing"
        elif command == "PAUSE":
            self.speaker["state"] = "paused"
        if self.callback:
            self.callback(FakeSpeakerState(key, self.speaker["volume"], self.speaker["muted"], self.speaker["state"]))

    def subscribe_states(self, callback):
        self.callback = callback
        self._emit(1)

    async def device_info(self):
        return object()

    def light_command(self, key, **changes):
        self.commands.append((key, changes))
        if key == 1:
            for name, value in changes.items():
                if name == "rgb":
                    self.ring["red"], self.ring["green"], self.ring["blue"] = value
                else:
                    self.ring[name] = value
            self._emit(1)
        elif self.callback:
            self.callback(SimpleNamespace(key=key))

    def _emit(self, key):
        if self.callback:
            self.callback(SimpleNamespace(key=key, **self.ring))


class StaleStateClient(FakeAPIClient):
    instances = []

    def light_command(self, key, **changes):
        self.commands.append((key, changes))
        # Report the old light state after the command instead of the requested one.
        self._emit(key)

    async def disconnect(self):
        return None


@pytest.fixture(autouse=True)
def fake_esphome_api(monkeypatch):
    FakeAPIClient.instances.clear()
    StaleStateClient.instances.clear()
    api = SimpleNamespace(APIClient=FakeAPIClient, LightInfo=FakeLightInfo, MediaPlayerInfo=FakeMediaPlayerInfo)
    monkeypatch.setitem(sys.modules, "aioesphomeapi", api)


def test_match_execute_and_catalog_use_only_the_simulated_device():
    async def exercise():
        service = worker_module.RingService()
        await service.initialize({
            "device_ip": "192.168.1.20",
            "api_key": base64.b64encode(b"k" * 32).decode("ascii"),
        })
        client = FakeAPIClient.instances[-1]
        assert client.address == "192.168.1.20"
        assert client.port == 6053

        documentary = await service.handle("match", {"text": "Could you explain how the Echo Dot ring works?"})
        assert documentary == {"match": None}
        assert client.commands == []

        state = await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "state"}})
        assert state["result"]["device_connected"] is True
        assert state["result"]["state"] == {"on": False, "brightness": 0.5, "rgb": [0, 0, 0], "effect": "None"}

        on = await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "on"}})
        assert on["result"]["state"]["on"] is True
        assert "confirmed the ring is on" in on["result"]["message"]
        off = await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "off"}})
        assert off["result"]["state"]["on"] is False
        assert "confirmed the ring is off" in off["result"]["message"]

        # Color parsing is intentionally limited to user-specified numeric or
        # hex values; Luna, not the worker, selects RGB for color words.
        assert (await service.handle("match", {"text": "Make the Echo Dot ring blue"}))["match"] is None
        assert (await service.handle("match", {"text": "Make the Echo Dot ring #0000FF"}))["match"] == {"action": "color", "rgb": [0, 0, 255]}
        assert (await service.handle("match", {"text": "Set the Echo Dot ring to RGB 12, 34, 56"}))["match"] == {"action": "color", "rgb": [12, 34, 56]}
        assert (await service.handle("match", {"text": "What colors are available on the Ring Light?"}))["match"] == {"action": "color_catalog"}
        assert (await service.handle("match", {"text": "What colors are available on Echo Connect?"}))["match"] == {"action": "color_catalog"}
        color = await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "color", "rgb": [0, 0, 255]}})
        assert color["result"]["ok"] is True
        assert color["result"]["rgb"] == [0, 0, 255]
        assert color["result"]["state"]["on"] is True
        assert color["result"]["state"]["rgb"] == [0, 0, 255]
        assert color["result"]["message"] == "The Dot reports RGB (0, 0, 255) with effect None."

        phrasings = {
            "Turn the Echo Dot ring on": {"action": "on"},
            "Switch off the Echo Dot ring": {"action": "off"},
            "Turn on Echo Connect": {"action": "on"},
            "Turn the Echo Connect off": {"action": "off"},
            "Switch the ring on light to green": None,
            "hey change the light to red": None,
        }
        for text, expected in phrasings.items():
            assert (await service.handle("match", {"text": text}))["match"] == expected

        assert (await service.handle("match", {"text": "Run the Glitter animation on the Echo Dot ring"}))["match"] is None
        assert (await service.handle("match", {"text": "Run Rainbow Twinkle on the Echo Dot ring"}))["match"] == {"action": "effect", "effect": "Rainbow Twinkle"}
        effect = await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "effect", "effect": "Rainbow Twinkle"}})
        assert effect["result"]["state"]["effect"] == "Rainbow Twinkle"
        assert effect["result"]["message"] == "The Echo Dot confirmed the Rainbow Twinkle animation is running."
        color_catalog = await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "color_catalog"}})
        assert "Luna chooses RGB channels" in color_catalog["result"]["message"]
        assert (await service.handle("match", {"text": "What animations are available on the Echo Dot ring?"}))["match"] == {"action": "catalog"}
        catalog = await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "catalog"}})
        assert "available_effects" in catalog["result"]
        assert not any(key.startswith("available_palette") for key in catalog["result"])
        assert client.commands == [
            (1, {"state": True}),
            (1, {"state": False}),
            (1, {"state": True, "rgb": (0.0, 0.0, 1.0), "effect": "None"}),
            (1, {"state": True, "effect": "Rainbow Twinkle"}),
        ]
        await service.close()

    asyncio.run(exercise())


def test_skill_action_formats_are_strict_before_any_device_command():
    async def exercise():
        service = worker_module.RingService()
        await service.initialize({
            "device_ip": "192.168.1.20",
            "api_key": base64.b64encode(b"k" * 32).decode("ascii"),
        })
        client = FakeAPIClient.instances[-1]
        for arguments in (
            {"action": "color", "rgb": [1, 2, 3], "hex": "#010203"},
            {"action": "color", "color": "red"},
            {"action": "palette", "palette": "ocean"},
            {"action": "color_catalog", "rgb": [255, 255, 255]},
            {"action": "color", "rgb": [256, 0, 0]},
            {"action": "brightness", "brightness": True},
            {"action": "effect", "effect": "x" * 65},
            {"action": "palette", "palette": "not-a-palette", "mode": "animate"},
            {"action": "on", "rgb": [1, 2, 3]},
        ):
            with pytest.raises(ValueError):
                await service.handle("execute", {"tool": "control_ring", "arguments": arguments})
        assert client.commands == []
        for valid in (
            {"action": "catalog"},
            {"action": "color", "rgb": [1, 2, 3]},
        ):
            assert worker_module.RingService._validate_tool_arguments(valid)["action"] == valid["action"]
        await service.close()

    asyncio.run(exercise())


def test_device_confirmation_requires_the_requested_rgb_and_effect(monkeypatch):
    async def exercise():
        class FakeAPI:
            APIClient = StaleStateClient
            LightInfo = FakeLightInfo
        monkeypatch.setitem(sys.modules, "aioesphomeapi", FakeAPI)
        service = worker_module.RingService()
        await service.initialize({
            "device_ip": "192.168.1.20",
            "api_key": base64.b64encode(b"k" * 32).decode("ascii"),
        })
        client = StaleStateClient.instances[-1]
        with pytest.raises(TimeoutError, match="did not confirm"):
            await service.handle("execute", {
                "tool": "control_ring",
                "arguments": {"action": "color", "rgb": [255, 0, 0]},
            })
        assert client.commands
        await service.close()

    asyncio.run(exercise())


def test_invalid_setup_is_rejected_without_constructing_a_client():
    async def exercise():
        service = worker_module.RingService()
        with pytest.raises(ValueError, match="private LAN IPv4"):
            await service.initialize({
                "device_ip": "127.0.0.1",
                "api_key": base64.b64encode(b"k" * 32).decode("ascii"),
            })
        assert FakeAPIClient.instances == []

    asyncio.run(exercise())


def test_speaker_volume_mute_and_speech_actions_round_trip():
    """Volume/mute/stop act on the Dot's media player; say is validated and
    accepted as a declared speech handoff; ring commands stay untouched."""

    async def exercise():
        service = worker_module.RingService()
        await service.initialize({
            "device_ip": "192.168.1.20",
            "api_key": base64.b64encode(b"k" * 32).decode("ascii"),
        })
        client = FakeAPIClient.instances[-1]

        volume = await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "volume", "volume": 0.4}})
        assert volume["result"]["ok"] is True
        assert client.media_commands[-1] == (300, {"volume": 0.4})
        assert volume["result"]["speaker"]["volume"] == 0.4
        assert "40 percent" in volume["result"]["message"]

        mute = await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "mute"}})
        assert mute["result"]["speaker"]["muted"] is True
        unmute = await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "unmute"}})
        assert unmute["result"]["speaker"]["muted"] is False

        say = await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "say", "text": "Dinner is ready"}})
        assert say["result"]["ok"] is True

        stop = await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "stop_speaking"}})
        assert stop["result"]["speaker"]["state"] == "idle"
        assert client.media_commands[-1][1]["command"] == "STOP"

        for bad in (
            {"action": "volume", "volume": 1.5},
            {"action": "volume", "volume": "loud"},
            {"action": "say", "text": ""},
            {"action": "unmute", "extra": 1},
        ):
            with pytest.raises(ValueError):
                await service.handle("execute", {"tool": "control_ring", "arguments": bad})
        assert client.commands == []

        await service.close()

    asyncio.run(exercise())


def test_incomplete_voice_download_is_not_treated_as_installed(tmp_path, monkeypatch):
    """A truncated onnx (or one missing its .onnx.json config) must re-download.

    Piper loads a voice from the weights *and* the config beside them, so a
    half-finished download would otherwise make every request for that voice
    fall back to the robotic espeak-ng path instead of fetching the voice.
    """
    monkeypatch.setattr(worker_module, "_TTS_DIR", str(tmp_path))
    weights = tmp_path / "en_US-amy-medium.onnx"
    config = tmp_path / "en_US-amy-medium.onnx.json"

    assert worker_module._voice_model_path("amy") is None  # nothing downloaded yet

    weights.write_bytes(b"\x08\x08onnx-but-truncated" * 60_000)
    assert weights.stat().st_size > 1_000_000
    assert worker_module._voice_model_path("amy") is None  # no config beside it

    config.write_text("{}", encoding="utf-8")
    assert worker_module._voice_model_path("amy") == str(weights)

    # A stub, 12-byte "model" is still refused even when the config exists.
    weights.write_bytes(b"tiny")
    assert worker_module._voice_model_path("amy") is None


def _wav_bytes(seconds=0.15, rate=22050):
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


class FakePiperDaemon:
    """Records the jobs a single warm helper would have received."""

    def __init__(self):
        self.jobs = []

    def synthesize(self, *, text, model, out, length_scale, noise_scale, noise_w_scale, warm=False):
        self.jobs.append({"text": text, "model": model, "warm": warm})
        with open(out, "wb") as handle:
            handle.write(_wav_bytes())
        return True


def test_speech_reuses_one_warm_helper_and_the_phrase_cache(tmp_path, monkeypatch):
    """Repeated speech must not pay a fresh Piper load, and a repeated phrase
    must not even reach the helper.

    The old path spawned `python -m piper` per request, which cost the
    interpreter start plus the ~60 MB voice load every single time.
    """
    monkeypatch.setattr(worker_module, "_TTS_DIR", str(tmp_path))
    (tmp_path / "en_US-amy-medium.onnx").write_bytes(b"weights" * 400_000)
    (tmp_path / "en_US-amy-medium.onnx.json").write_text("{}", encoding="utf-8")

    daemon = FakePiperDaemon()
    monkeypatch.setattr(worker_module, "_piper_daemon", lambda: daemon)

    def render(text, name):
        return worker_module._synthesize_speech_wav(text, str(tmp_path), name)

    first = render("Dinner is ready", "one")
    second = render("Turning the light down", "two")
    assert [job["text"] for job in daemon.jobs] == ["Dinner is ready", "Turning the light down"]
    assert first != second
    assert Path(first).is_file() and Path(second).is_file()

    # The next request may pass a fresh name, but an identical phrase is served
    # from the cached WAV instead of re-rendering it.
    again = render("Dinner is ready", "three")
    assert again == first
    assert len(daemon.jobs) == 2, "the cached phrase never reached the helper"


def test_piper_helper_is_spawned_once_and_reused(monkeypatch):
    """One long-lived helper per interpreter: the voice stays loaded in it."""
    created = []

    class CountingDaemon:
        def __init__(self, python):
            self.python = python
            created.append(python)

        def stop(self):
            return None

    monkeypatch.setattr(worker_module, "_PiperDaemon", CountingDaemon)
    monkeypatch.setattr(worker_module, "_PIPER_DAEMON", None)
    monkeypatch.setattr(worker_module, "_piper_python", lambda: "/usr/bin/python3")

    first = worker_module._piper_daemon()
    second = worker_module._piper_daemon()
    assert first is second
    assert created == ["/usr/bin/python3"]

    # A different interpreter retires the stale helper instead of mixing them.
    monkeypatch.setattr(worker_module, "_piper_python", lambda: "/usr/bin/python3.13")
    third = worker_module._piper_daemon()
    assert third is not first
    assert created == ["/usr/bin/python3", "/usr/bin/python3.13"]


def test_concurrent_speech_jobs_never_steal_each_others_reply(tmp_path, monkeypatch):
    """The connect-time warm-up runs while a `say` may already be in flight.

    Both jobs share the one helper, and the helper answers on a shared stream,
    so a job that ran concurrently could consume the other's reply and then
    wait out the full timeout before falling back to the slow CLI render.
    """
    import json as json_module
    import textwrap

    stub_helper = textwrap.dedent(
        """
        import json, sys, time
        for line in sys.stdin:
            job = json.loads(line)
            time.sleep(0.3)
            with open(job["out"], "wb") as handle:
                handle.write(b"RIFF" + b"\\x00" * 200)
            sys.stdout.write(json.dumps({"id": job["id"], "ok": True}) + "\\n")
            sys.stdout.flush()
        """
    )
    monkeypatch.setattr(worker_module, "_PIPER_HELPER_SOURCE", stub_helper)
    daemon = worker_module._PiperDaemon(sys.executable, timeout=5.0)
    results: dict[str, object] = {}

    def render(name):
        target = tmp_path / f"{name}.wav"
        results[name] = daemon.synthesize(
            text=f"phrase {name}", model="unused.onnx", out=str(target),
            length_scale=1.0, noise_scale=0.7, noise_w_scale=0.8, warm=name == "warm",
        )

    threads = [threading.Thread(target=render, args=(name,)) for name in ("warm", "say")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)

    try:
        assert not any(thread.is_alive() for thread in threads), "a job was left waiting out the timeout"
        assert results == {"warm": True, "say": True}
        assert (tmp_path / "warm.wav").is_file() and (tmp_path / "say.wav").is_file()
    finally:
        daemon.stop()


def test_synthesize_falls_back_when_the_helper_is_retired_mid_job(monkeypatch):
    """A stop() landing between start() and the write must fall back, not raise.

    stop() is called from other threads - a concurrent `say`, or _piper_daemon()
    retiring the helper when the chosen interpreter changes - and it clears the
    shared process handle. Reading that handle twice left an unhandled
    AttributeError on the warm-up thread (and a crash in a worker thread of a
    live skill), so the job has to give up and let the caller use the fallback.
    """
    daemon = worker_module._PiperDaemon(sys.executable, timeout=1.0)
    monkeypatch.setattr(daemon, "start", lambda: True)
    monkeypatch.setattr(daemon, "stop", lambda: None)

    assert daemon.synthesize(
        text="hello", model="voice.onnx", out="/tmp/never-written.wav",
        length_scale=1.0, noise_scale=0.7, noise_w_scale=0.8,
    ) is False


def test_warm_speech_never_raises_from_the_background_thread(tmp_path, monkeypatch):
    """The connect-time warm-up is best-effort: a broken helper must not turn
    into an unhandled thread exception that fails an unrelated later test."""

    class ExplodingDaemon:
        def synthesize(self, **kwargs):
            raise RuntimeError("the helper died")

    monkeypatch.setattr(worker_module, "_TTS_DIR", str(tmp_path))
    monkeypatch.setattr(worker_module, "_ensure_voice", lambda voice: "voice.onnx")
    monkeypatch.setattr(worker_module, "_piper_daemon", lambda: ExplodingDaemon())

    worker_module._warm_speech()


class FakeUnrealSpeech:
    """Local stand-in for the Unreal Speech API used by the speech tests.

    Serves the documented v8 contract (POST /speech -> JSON with an OutputUri,
    then the MP3) and records every request, so the tests can assert the
    Authorization header, payload, and key rotation without touching the real
    service or spending allowance.
    """

    def __init__(self, audio=b"", statuses=None):
        self.audio = audio or (b"\xff\xfb\x90\x00" + b"\x00" * 4096)
        self.statuses = statuses or {}  # bearer key -> HTTP status to answer with
        self.requests = []              # (path, authorization header, payload dict)
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return None

            def _send(self, status, body, content_type):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.endswith(".mp3"):
                    self._send(200, outer.audio, "audio/mpeg")
                    return
                self._send(404, b"not found", "text/plain")

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                authorization = self.headers.get("Authorization") or ""
                outer.requests.append((self.path, authorization, payload))
                status = outer.statuses.get(authorization.removeprefix("Bearer "), 200)
                if status != 200:
                    # 401/403 are key problems, 429 is one rate-limited request.
                    detail = {
                        401: b"Invalid API key. Please provide a valid API key.",
                        403: b"Monthly quota exceeded for this key.",
                        429: b"Too many requests. Slow down.",
                    }.get(status, b"upstream error")
                    self._send(status, detail, "text/plain")
                    return
                body = json.dumps({
                    "OutputUri": f"{outer.base}/audio.mp3",
                    "TimestampsUri": f"{outer.base}/audio.json",
                }).encode()
                self._send(200, body, "application/json")

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


@pytest.fixture
def cloud_speech(tmp_path, monkeypatch):
    """Point the worker's cloud TTS at a local fake provider and a temp cache."""
    provider = FakeUnrealSpeech()
    monkeypatch.setattr(worker_module, "_TTS_DIR", str(tmp_path))
    monkeypatch.setattr(worker_module, "_UNREAL_SPEECH_BASE", provider.base)
    monkeypatch.setattr(worker_module, "_find_ffmpeg", lambda: "/fake/ffmpeg")
    monkeypatch.setattr(worker_module, "_convert_audio", fake_ffmpeg_convert)
    monkeypatch.setattr(worker_module, "_synthesize_speech_wav", fake_local_speech)
    yield provider
    provider.stop()


def fake_ffmpeg_convert(_src, _ffmpeg, dest, target):
    """Stand-in transcode: a valid 2 s PCM WAV at the requested format."""
    with open(dest, "wb") as handle:
        handle.write(_wav_bytes(seconds=2.0, rate=int(target.get("sample_rate") or 16000)))


def fake_local_speech(_text, out_dir, name, **_kwargs):
    """Stand-in built-in voice, so fallback tests need no Piper/espeak install."""
    path = os.path.join(out_dir, f"{name}-local.wav")
    with open(path, "wb") as handle:
        handle.write(_wav_bytes(seconds=0.5))
    return path


def _usage_document(entries):
    """Seed the cloud usage file with {api key: (characters, seconds)} usage."""
    return {
        "version": 1,
        "keys": {
            worker_module._cloud_key_id(key): {
                "month": worker_module._cloud_month(),
                "characters": characters,
                "seconds": seconds,
                "exhausted": False,
            }
            for key, (characters, seconds) in entries.items()
        },
    }


async def _say(service, text):
    response = await service.handle(
        "execute", {"tool": "control_ring", "arguments": {"action": "say", "text": text}}
    )
    return response["result"]


def test_cloud_speech_uses_the_configured_key_and_tracks_usage(cloud_speech, tmp_path):
    """One `say` calls /speech once with the documented payload, hands the Dot a
    locally served WAV, and records what the key was charged."""

    async def exercise():
        service = worker_module.RingService()
        await service.initialize({
            "device_ip": "192.168.1.20",
            "api_key": base64.b64encode(b"k" * 32).decode("ascii"),
            "unrealspeech_api_key": "key-one-abcdef",
            "unrealspeech_voice": "Scarlett",
        })
        client = FakeAPIClient.instances[-1]

        result = await _say(service, "Dinner is ready")
        assert result["tts"]["engine"] == "unrealspeech"
        assert result["tts"]["provider"] == "auto"
        assert result["tts"]["key_index"] == 1
        assert result["tts"]["voice"] == "scarlett"
        assert result["tts"]["fallback"] == ""
        assert result["tts"]["characters_limit"] == 250_000
        assert result["tts"]["seconds_limit"] == 21_600
        assert result["tts"]["keys"] == [{
            "index": 1, "characters": len("Dinner is ready"), "seconds": 2.0,
            "characters_percent": 0, "seconds_percent": 0, "ready": True,
        }]
        assert "scarlett voice" in result["message"]

        # The Dot gets an announcement pointing at the worker's own file server.
        _, changes = client.media_commands[-1]
        assert changes["announcement"] is True
        assert changes["media_url"].endswith(".wav")

        # Exactly one provider call, with the documented contract and speed map.
        assert len(cloud_speech.requests) == 1
        path, authorization, payload = cloud_speech.requests[0]
        assert path == "/speech"
        assert authorization == "Bearer key-one-abcdef"
        assert payload["Text"] == "Dinner is ready"
        assert payload["VoiceId"] == "Charlotte"
        assert payload["Speed"] == "0.05"  # Piper's 1.05 neutral pace -> API 0.05
        assert payload["TimestampType"] == "sentence"

        # Usage is persisted under a hash of the key, never the key itself.
        usage_file = tmp_path / "unrealspeech-usage.json"
        raw = usage_file.read_text(encoding="utf-8")
        assert "key-one-abcdef" not in raw
        entry = json.loads(raw)["keys"][worker_module._cloud_key_id("key-one-abcdef")]
        assert entry["characters"] == len("Dinner is ready") and entry["seconds"] == 2.0
        assert entry["month"] == worker_module._cloud_month()
        assert entry["exhausted"] is False
        assert (usage_file.stat().st_mode & 0o777) == 0o600

        # A repeated phrase is replayed from the local cache: no second call and
        # no second charge against the allowance.
        again = await _say(service, "Dinner is ready")
        assert again["tts"]["engine"] == "unrealspeech"
        assert again["tts"]["keys"][0]["characters"] == len("Dinner is ready")
        assert len(cloud_speech.requests) == 1
        await service.close()

    asyncio.run(exercise())


def test_cloud_voice_catalog_uses_current_provider_voice_ids():
    """The provider replaced its VoiceId catalog, so the old names are rejected
    with HTTP 400 and every announcement quietly fell back to the local voice.
    Pin the mapping to accepted values so future catalog drift is caught here.
    """
    assert worker_module._CLOUD_VOICE_CATALOG == {
        "amy": "Amelia",
        "scarlett": "Charlotte",
        "dan": "Daniel",
        "liv": "Lauren",
        "will": "Oliver",
    }
    assert worker_module.DEFAULT_CLOUD_VOICE in worker_module._CLOUD_VOICE_CATALOG


def test_an_unknown_cloud_voice_is_rejected_at_setup(cloud_speech):
    """A voice the catalog does not offer fails setup instead of being sent as
    an invalid VoiceId (which the provider would reject at announcement time)."""
    service = worker_module.RingService()
    with pytest.raises(ValueError, match="Unreal Speech voice must be one of"):
        service._configure_speech({"unrealspeech_voice": "not-a-voice"})


def test_keys_hand_over_at_ninety_percent_of_either_limit(cloud_speech, tmp_path):
    """A key at 90% of its characters or of its generated audio steps aside for
    the next key instead of failing; the snapshot says why."""
    (tmp_path / "unrealspeech-usage.json").write_text(
        json.dumps(_usage_document({
            "key-one-abcdef": (225_000, 0.0),    # exactly 90% of the characters
            "key-two-abcdef": (0, 19_440.0),     # exactly 90% of the audio hours
        })),
        encoding="utf-8",
    )

    async def exercise():
        service = worker_module.RingService()
        await service.initialize({
            "device_ip": "192.168.1.20",
            "api_key": base64.b64encode(b"k" * 32).decode("ascii"),
            "unrealspeech_api_key": "key-one-abcdef",
            "unrealspeech_api_key_2": "key-two-abcdef",
            "unrealspeech_api_key_3": "key-three-abcdef",
        })
        result = await _say(service, "Dinner is ready")
        assert result["tts"]["engine"] == "unrealspeech"
        assert result["tts"]["key_index"] == 3
        assert cloud_speech.requests[-1][1] == "Bearer key-three-abcdef"
        keys = result["tts"]["keys"]
        assert [key["ready"] for key in keys] == [False, False, True]
        assert keys[0]["characters_percent"] == 90 and keys[0]["characters"] == 225_000
        assert keys[1]["seconds_percent"] == 90 and keys[1]["seconds"] == 19_440.0
        assert result["tts"]["fallback"] == ""
        # Handover is not a failure: the announcement still spoke in the cloud voice.
        assert "amy voice" in result["message"] and "fallback" not in result["message"]
        await service.close()

    asyncio.run(exercise())


def test_every_spent_key_falls_back_to_the_built_in_voice(cloud_speech, tmp_path):
    """With all three keys at their thresholds the built-in voice speaks, and the
    result reports that instead of pretending the cloud voice was used."""
    (tmp_path / "unrealspeech-usage.json").write_text(
        json.dumps(_usage_document({
            "key-one-abcdef": (240_000, 0.0),
            "key-two-abcdef": (0, 20_000.0),
            "key-three-abcdef": (250_000, 22_000.0),
        })),
        encoding="utf-8",
    )

    async def exercise():
        service = worker_module.RingService()
        await service.initialize({
            "device_ip": "192.168.1.20",
            "api_key": base64.b64encode(b"k" * 32).decode("ascii"),
            "unrealspeech_api_key": "key-one-abcdef",
            "unrealspeech_api_key_2": "key-two-abcdef",
            "unrealspeech_api_key_3": "key-three-abcdef",
        })
        client = FakeAPIClient.instances[-1]
        result = await _say(service, "Dinner is ready")
        assert result["tts"]["engine"] == "local"
        assert result["tts"]["key_index"] == 0
        assert result["tts"]["fallback"]
        assert "local fallback" in result["message"]
        assert all(key["ready"] is False for key in result["tts"]["keys"])
        assert cloud_speech.requests == []
        assert client.media_commands[-1][1]["media_url"].endswith("-local.wav")
        assert result["voice"] == "amy"
        await service.close()

    asyncio.run(exercise())


def test_rejected_key_is_retired_and_the_next_key_speaks(cloud_speech, tmp_path):
    """A 401 retires that key for the month (persisted), so later announcements
    skip the failing key instead of paying for another round trip."""
    cloud_speech.statuses["key-one-abcdef"] = 401

    async def exercise():
        service = worker_module.RingService()
        await service.initialize({
            "device_ip": "192.168.1.20",
            "api_key": base64.b64encode(b"k" * 32).decode("ascii"),
            "unrealspeech_api_key": "key-one-abcdef",
            "unrealspeech_api_key_2": "key-two-abcdef",
        })
        first = await _say(service, "Dinner is ready")
        assert first["tts"]["engine"] == "unrealspeech"
        assert first["tts"]["key_index"] == 2
        assert cloud_speech.requests[0][1] == "Bearer key-one-abcdef"
        stored = json.loads((tmp_path / "unrealspeech-usage.json").read_text(encoding="utf-8"))
        assert stored["keys"][worker_module._cloud_key_id("key-one-abcdef")]["exhausted"] is True

        # A different phrase never retries the retired key: it is attempted
        # exactly once, then only key 2 is used.
        second = await _say(service, "The kettle is boiling")
        assert second["tts"]["key_index"] == 2
        assert [authorization for _, authorization, _ in cloud_speech.requests] == [
            "Bearer key-one-abcdef",
            "Bearer key-two-abcdef",
            "Bearer key-two-abcdef",
        ]
        await service.close()

    asyncio.run(exercise())


def test_rate_limited_key_is_skipped_without_being_retired(cloud_speech, tmp_path):
    """A 429 is one unlucky request, not a spent key: that announcement uses the
    built-in voice, but the key keeps its remaining allowance."""
    cloud_speech.statuses["key-one-abcdef"] = 429

    async def exercise():
        service = worker_module.RingService()
        await service.initialize({
            "device_ip": "192.168.1.20",
            "api_key": base64.b64encode(b"k" * 32).decode("ascii"),
            "unrealspeech_api_key": "key-one-abcdef",
        })
        result = await _say(service, "Dinner is ready")
        assert result["tts"]["engine"] == "local"
        assert "429" in result["tts"]["fallback"]
        assert "local fallback" in result["message"]
        # The key keeps its allowance: nothing was charged or retired.
        usage_file = tmp_path / "unrealspeech-usage.json"
        charges = json.loads(usage_file.read_text(encoding="utf-8"))["keys"] if usage_file.is_file() else {}
        entry = charges.get(worker_module._cloud_key_id("key-one-abcdef")) or {"exhausted": False, "characters": 0}
        assert entry["exhausted"] is False and entry["characters"] == 0
        assert result["tts"]["keys"][0]["ready"] is True
        await service.close()

    asyncio.run(exercise())


def test_usage_resets_when_the_month_rolls_over(cloud_speech, tmp_path):
    """Last month's spent allowance must not block this month's keys."""
    stale = _usage_document({"key-one-abcdef": (250_000, 21_600.0)})
    for entry in stale["keys"].values():
        entry["month"] = "2020-01"
        entry["exhausted"] = True
    (tmp_path / "unrealspeech-usage.json").write_text(json.dumps(stale), encoding="utf-8")

    async def exercise():
        service = worker_module.RingService()
        await service.initialize({
            "device_ip": "192.168.1.20",
            "api_key": base64.b64encode(b"k" * 32).decode("ascii"),
            "unrealspeech_api_key": "key-one-abcdef",
        })
        result = await _say(service, "Dinner is ready")
        assert result["tts"]["engine"] == "unrealspeech"
        assert result["tts"]["key_index"] == 1
        assert result["tts"]["keys"][0]["characters"] == len("Dinner is ready")
        assert result["tts"]["keys"][0]["ready"] is True
        await service.close()

    asyncio.run(exercise())


def test_speech_provider_settings_are_validated_and_normalized(cloud_speech, tmp_path):
    """Keys, voice, and limits are read once at setup: bad values are refused
    before anything is spoken, defaults match the free tier, duplicates and
    blanks are ignored."""
    base = {
        "device_ip": "192.168.1.20",
        "api_key": base64.b64encode(b"k" * 32).decode("ascii"),
    }

    async def exercise():
        service = worker_module.RingService()
        for bad in (
            {"tts_provider": "elevenlabs"},
            {"unrealspeech_voice": "mickey"},
            {"unrealspeech_monthly_characters": "many"},
            {"unrealspeech_monthly_characters": "0"},
            {"unrealspeech_monthly_audio_hours": "0"},
            {"unrealspeech_monthly_audio_hours": "lots"},
            {"unrealspeech_api_key": 12345},
            {"unrealspeech_api_key": "short"},
            {"unrealspeech_api_key": "has a space"},
        ):
            with pytest.raises(ValueError):
                await service.initialize({**base, **bad})

        await service.initialize({
            **base,
            "tts_provider": "AUTO",
            "unrealspeech_api_key": "key-one-abcdef",
            "unrealspeech_api_key_2": "key-one-abcdef",
            "unrealspeech_api_key_3": "   ",
        })
        assert service.tts_provider == "auto"
        assert service.cloud_keys == ["key-one-abcdef"]
        assert service.cloud_voice == "amy"
        assert service.cloud_character_limit == 250_000
        assert service.cloud_seconds_limit == 21_600

        await service.initialize({
            **base,
            "tts_provider": "Local",
            "unrealspeech_voice": "WILL",
            "unrealspeech_monthly_characters": "100000",
            "unrealspeech_monthly_audio_hours": "0.5",
        })
        assert service.tts_provider == "local"
        assert service.cloud_voice == "will"
        assert service.cloud_character_limit == 100_000
        assert service.cloud_seconds_limit == 1800
        await service.close()

    asyncio.run(exercise())


def test_local_provider_never_calls_the_cloud(cloud_speech, tmp_path):
    """Selecting the built-in voice keeps every announcement offline even with
    keys configured, and reports the local engine."""

    async def exercise():
        service = worker_module.RingService()
        await service.initialize({
            "device_ip": "192.168.1.20",
            "api_key": base64.b64encode(b"k" * 32).decode("ascii"),
            "tts_provider": "local",
            "unrealspeech_api_key": "key-one-abcdef",
        })
        result = await _say(service, "Dinner is ready")
        assert result["tts"]["engine"] == "local"
        assert result["tts"]["fallback"] == ""
        assert cloud_speech.requests == []
        await service.close()

    asyncio.run(exercise())


def test_say_results_match_the_shipped_manifest_schema(cloud_speech, tmp_path):
    """Every speaker result, including the new tts quota object, must satisfy
    the exact result_schema the marketplace validates."""
    import skill_runtime

    manifest_path = (
        Path(__file__).parent / "ring_light_package" / "skills" / "echo-connect" / "skill.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest["tools"][0]["result_schema"]

    async def exercise():
        service = worker_module.RingService()
        await service.initialize({
            "device_ip": "192.168.1.20",
            "api_key": base64.b64encode(b"k" * 32).decode("ascii"),
            "unrealspeech_api_key": "key-one-abcdef",
            "unrealspeech_api_key_2": "key-two-abcdef",
        })
        for arguments in (
            {"action": "state"},
            {"action": "catalog"},
            {"action": "color_catalog"},
            {"action": "volume", "volume": 0.5},
            {"action": "say", "text": "Dinner is ready"},
            {"action": "say", "text": "The kettle is boiling", "voice": "jenny", "speed": 1.2},
            {"action": "say", "text": "The kettle is boiling", "voice": "jenny", "speed": 1.2},
        ):
            result = (await service.handle("execute", {"tool": "control_ring", "arguments": arguments}))["result"]
            skill_runtime.validate_json_schema(result, schema, "result")
        await service.close()

    asyncio.run(exercise())


@pytest.mark.skipif(worker_module._find_ffmpeg() is None, reason="ffmpeg is not installed here")
def test_cloud_speech_converts_a_real_mp3_with_ffmpeg(tmp_path, monkeypatch):
    """The conversion boundary is real ffmpeg when it is present: a genuine MP3
    becomes a PCM WAV in the device's announcement format, and its duration is
    measured for the allowance."""
    ffmpeg = worker_module._find_ffmpeg()
    source = tmp_path / "clip.mp3"
    subprocess.run(
        [ffmpeg, "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-acodec", "libmp3lame", str(source)],
        check=True,
    )
    provider = FakeUnrealSpeech(audio=source.read_bytes())
    monkeypatch.setattr(worker_module, "_UNREAL_SPEECH_BASE", provider.base)
    try:
        path, seconds, billed = worker_module._unrealspeech_speech_wav(
            "Test", str(tmp_path), "clip", api_key="key-one-abcdef", voice="amy", speed=1.05,
            playable=[{"format": "wav", "sample_rate": 16000, "num_channels": 1,
                       "purpose": "announcement", "sample_bytes": 2}],
        )
        assert billed is True
        with wave.open(path, "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getframerate() == 16000
            assert abs(seconds - handle.getnframes() / handle.getframerate()) < 0.001
        assert abs(seconds - 2.0) < 0.2
        cached, again, charged = worker_module._unrealspeech_speech_wav(
            "Test", str(tmp_path), "other", api_key="key-one-abcdef", voice="amy", speed=1.05,
            playable=[],
        )
        assert (cached, charged) == (path, False)
        assert again == seconds
    finally:
        provider.stop()


def test_speaker_state_uses_enum_name_not_numeric_value():
    """Real hardware pushes a MediaPlayerState IntEnum whose str() is its
    number on Python 3.11+; the worker must keep the NAME so "playing"/
    "paused"/"idle" comparisons (and the transport predicates built on them)
    work."""
    import enum as _enum

    class HardwareState(_enum.IntEnum):
        IDLE = 1
        PLAYING = 2
        PAUSED = 3

    async def exercise():
        service = worker_module.RingService()
        await service.initialize({
            "device_ip": "192.168.1.20",
            "api_key": base64.b64encode(b"k" * 32).decode("ascii"),
        })
        assert str(HardwareState.PLAYING) == "2"  # the trap being guarded against
        service._on_state(SimpleNamespace(key=300, volume=0.4, muted=False, state=HardwareState.PLAYING))
        assert service.speaker["state"] == "playing"
        service._on_state(SimpleNamespace(key=300, volume=0.4, muted=False, state=HardwareState.PAUSED))
        assert service.speaker["state"] == "paused"
        service._on_state(SimpleNamespace(key=300, volume=0.4, muted=False, state=HardwareState.IDLE))
        assert service.speaker["state"] == "idle"
        await service.close()

    asyncio.run(exercise())


def test_url_playback_pause_resume_and_dj_controls(monkeypatch):
    """play_url hands a decodable URL to the media player; pause/resume act
    on it; stop clears it. Capabilities come from the device's own
    supported_formats, and the transport controls resolve for the Nix music
    API (see music_api._control_for)."""
    # The Dot advertises 48 kHz stereo WAV, so a WAV upstream is passed through
    # untouched (no ffmpeg needed) and playback is confirmed by the state push.
    monkeypatch.setattr(worker_module, "_probe_content_type", lambda url: "audio/wav")

    async def exercise():
        service = worker_module.RingService()
        await service.initialize({
            "device_ip": "192.168.1.20",
            "api_key": base64.b64encode(b"k" * 32).decode("ascii"),
        })
        client = FakeAPIClient.instances[-1]

        assert service.media_supports_pause is True
        assert any(f["format"] == "wav" and f["purpose"] == "default" for f in service.media_formats)

        played = await service.handle("execute", {
            "tool": "control_ring",
            "arguments": {"action": "play_url", "url": "http://192.168.1.9:4100/stream", "title": "Bohemian Rhapsody"},
        })
        assert played["result"]["ok"] is True
        assert "Bohemian Rhapsody" in played["result"]["media"]["title"]
        assert client.media_commands[-1] == (300, {"command": "PLAY", "media_url": "http://192.168.1.9:4100/stream"})
        assert played["result"]["media"]["state_confirmed"] is True
        assert service.playing_url == "http://192.168.1.9:4100/stream"

        paused = await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "pause"}})
        assert client.media_commands[-1] == (300, {"command": "PAUSE"})
        assert paused["result"]["speaker"]["state"] == "paused"

        resumed = await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "resume"}})
        assert client.media_commands[-1] == (300, {"command": "PLAY"})
        assert resumed["result"]["speaker"]["state"] == "playing"

        stopped = await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "stop_speaking"}})
        assert stopped["result"]["speaker"]["state"] == "idle"
        assert service.playing_url is None

        # Strict validation: missing/unsafe URLs and unknown args never reach
        # the device.
        for bad in (
            {"action": "play_url"},
            {"action": "play_url", "url": "file:///etc/passwd"},
            {"action": "play_url", "url": "ftp://host/track"},
            {"action": "play_url", "url": "http://ok/audio", "title": ["x"]},
            {"action": "pause", "url": "http://ok/audio"},
        ):
            with pytest.raises(ValueError):
                await service.handle("execute", {"tool": "control_ring", "arguments": bad})
        await service.close()

    asyncio.run(exercise())


def test_speaker_commands_use_esphome_media_player_command_enum(monkeypatch):
    """Production aioesphomeapi expects MediaPlayerCommand members, not strings."""

    class FakeMediaPlayerCommand(enum.IntEnum):
        MUTE = 7
        UNMUTE = 8
        STOP = 3

    class EnumAPI:
        APIClient = FakeAPIClient
        LightInfo = FakeLightInfo
        MediaPlayerInfo = FakeMediaPlayerInfo
        MediaPlayerCommand = FakeMediaPlayerCommand

    monkeypatch.setitem(sys.modules, "aioesphomeapi", EnumAPI)

    async def exercise():
        service = worker_module.RingService()
        await service.initialize({
            "device_ip": "192.168.1.20",
            "api_key": base64.b64encode(b"k" * 32).decode("ascii"),
        })
        client = FakeAPIClient.instances[-1]
        await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "mute"}})
        await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "unmute"}})
        await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "stop_speaking"}})
        commands = [changes.get("command") for _, changes in client.media_commands]
        assert commands == [FakeMediaPlayerCommand.MUTE, FakeMediaPlayerCommand.UNMUTE, FakeMediaPlayerCommand.STOP]
        assert client.speaker["muted"] is False
        await service.close()

    asyncio.run(exercise())
