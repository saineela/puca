import asyncio
import base64
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


WORKER_PATH = (
    Path(__file__).parent
    / "ring_light_package"
    / "skills"
    / "ring-light"
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


class FakeAPIClient:
    instances = []

    def __init__(self, address, port, *, noise_psk):
        self.address = address
        self.port = port
        self.noise_psk = noise_psk
        self.callback = None
        self.commands = []
        self.ring = {"state": False, "brightness": 0.5, "red": 0.0, "green": 0.0, "blue": 0.0, "effect": "None"}
        self.instances.append(self)

    async def connect(self, *, login):
        assert login is True

    async def device_info_and_list_entities(self):
        entities = [FakeLightInfo(1, "ring", "LED Ring", ["Rainbow", "Rainbow Twinkle"])]
        entities.extend(FakeLightInfo(100 + number, f"segment_{number}", f"Segment {number}") for number in range(1, 13))
        return object(), entities, []

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
    api = SimpleNamespace(APIClient=FakeAPIClient, LightInfo=FakeLightInfo)
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
        color = await service.handle("execute", {"tool": "control_ring", "arguments": {"action": "color", "rgb": [0, 0, 255]}})
        assert color["result"]["ok"] is True
        assert color["result"]["rgb"] == [0, 0, 255]
        assert color["result"]["state"]["on"] is True
        assert color["result"]["state"]["rgb"] == [0, 0, 255]
        assert color["result"]["message"] == "The Dot reports RGB (0, 0, 255) with effect None."

        phrasings = {
            "Turn the Echo Dot ring on": {"action": "on"},
            "Switch off the Echo Dot ring": {"action": "off"},
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
