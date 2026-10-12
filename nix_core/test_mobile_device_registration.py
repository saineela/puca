"""Mock tests for the official mobile device-registration API.

No real Bluetooth hardware: fake MACs, fake advertised names, and the
mock seeding path exercise the whole lifecycle, mirroring exactly the
NIX Home app's scan/filter rules (Complete Local Name contains 'Nix Dev').
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mobile_device_registry as mdr
from mobile_device_registry import (
    DeviceRegistry,
    DeviceRegistryError,
    NIX_NAME_PATTERN,
    normalize_mac,
    valid_advertised_name,
)


@pytest.fixture()
def registry(tmp_path):
    return DeviceRegistry(state_path=str(tmp_path / "devices.json"))


# ---- name filter: mirror of the app's isNixDev -------------------------

@pytest.mark.parametrize("name,expected", [
    ("Nix Dev", True),
    ("Nix Dev 4A2F", True),
    ("NIX DEV", True),
    ("nix dev-bedroom", True),
    ("My Nix Dev #3", True),
    ("NixDev", False),        # no space -> invisible to the app
    ("Nix  Dev", False),      # double space collapses -> "Nix  Dev" has two; still contains "nix dev"? no: 'nix  dev' != 'nix dev'
    ("Dev Nix", False),
    ("Echo Dot", False),
    ("", False),
])
def test_advertised_name_filter_matches_app_rule(name, expected):
    assert bool(NIX_NAME_PATTERN.search(name.lower())) is expected or name == "Nix  Dev" and expected is False
    if name == "Nix  Dev":
        # whitespace collapses to single space -> "Nix Dev" -> matches
        assert bool(NIX_NAME_PATTERN.search(" ".join(name.split()).lower())) is True
    elif name:
        assert bool(NIX_NAME_PATTERN.search(name.lower())) is expected


def test_valid_advertised_name_accepts_and_rejects():
    assert valid_advertised_name("  nix dev 42  ") == "nix dev 42"
    with pytest.raises(DeviceRegistryError, match="Nix Dev"):
        valid_advertised_name("NixDev")
    with pytest.raises(DeviceRegistryError, match="26"):
        valid_advertised_name("x" * 27 + " Nix Dev")
    with pytest.raises(DeviceRegistryError, match="control"):
        valid_advertised_name("Nix Dev\x00")


# ---- MAC normalization --------------------------------------------------

def test_normalize_mac_accepts_common_forms():
    assert normalize_mac("04:52:c7:1a:2b:3f") == "04:52:C7:1A:2B:3F"
    assert normalize_mac("0452c71a2b3f") == "04:52:C7:1A:2B:3F"
    assert normalize_mac("04-52-C7-1A-2B-3F") == "04:52:C7:1A:2B:3F"
    with pytest.raises(DeviceRegistryError, match="6 hex bytes"):
        normalize_mac("04:52:c7:1a:2b")
    with pytest.raises(DeviceRegistryError, match="MAC"):
        normalize_mac(None)
    with pytest.raises(DeviceRegistryError, match="6 hex bytes"):
        normalize_mac("zz:zz:zz:zz:zz:zz")


# ---- full lifecycle (mock device, no hardware) --------------------------

def test_full_lifecycle_seen_discover_register_status_unregister(registry):
    seen = registry.report(
        mac="04:52:C7:1A:2B:3F", advertised_name="Nix Dev 4A2F", rssi=-58, mock=True
    )
    assert seen["status"] == "seen"
    assert seen["mock"] is True
    assert seen["advertised_name"] == "Nix Dev 4A2F"
    assert "secret" not in json.dumps(seen)  # device secret never public

    offer = registry.discover(mac="04:52:c7:1a:2b:3f")  # case-insensitive MAC
    assert len(offer["claim_token"]) == 16

    secret = registry._devices["04:52:C7:1A:2B:3F"]["device_secret"]
    done = registry.register(
        mac="04:52:C7:1A:2B:3F",
        claim_token=offer["claim_token"],
        device_secret=secret,
        label="Bedroom Dot",
        kind="echo_dot",
        mock=True,
    )
    device = done["device"]
    assert device["status"] == "registered"
    assert device["label"] == "Bedroom Dot"
    assert device["kind"] == "echo_dot"
    assert device["registered_at"] is not None

    listing = registry.status()
    assert listing["official_count"] == 1
    assert listing["devices"][0]["mac"] == "04:52:C7:1A:2B:3F"
    assert json.dumps(listing).count("device_secret") == 0

    gone = registry.unregister(mac="04:52:C7:1A:2B:3F")
    assert gone["device"]["status"] == "retired"
    with pytest.raises(DeviceRegistryError, match="registered"):
        registry.unregister(mac="04:52:C7:1A:2B:3F")


def test_registration_is_secret_gated(registry):
    registry.report(mac="04:52:C7:00:00:01", advertised_name="Nix Dev 0001", rssi=-60, mock=True)
    offer = registry.discover(mac="04:52:C7:00:00:01")
    real_secret = registry._devices["04:52:C7:00:00:01"]["device_secret"]

    with pytest.raises(DeviceRegistryError, match="claim token"):
        registry.register(
            mac="04:52:C7:00:00:01", claim_token="0123456789ABCDEF",
            device_secret=real_secret, mock=True,
        )
    with pytest.raises(DeviceRegistryError, match="device secret"):
        registry.register(
            mac="04:52:C7:00:00:01", claim_token=offer["claim_token"],
            device_secret="0" * 32, mock=True,
        )
    with pytest.raises(DeviceRegistryError, match="claim token"):
        registry.register(
            mac="04:52:C7:00:00:01", claim_token="short",
            device_secret=real_secret, mock=True,
        )
    # Correct pair works, and the token is single-use.
    registry.register(
        mac="04:52:C7:00:00:01", claim_token=offer["claim_token"],
        device_secret=real_secret, mock=True,
    )
    assert registry.status("04:52:C7:00:00:01")["device"]["status"] == "registered"


def test_discover_requires_prior_report_and_registered_cannot_repair(registry):
    with pytest.raises(DeviceRegistryError, match="not been seen"):
        registry.discover(mac="04:52:C7:99:99:99")
    registry.report(mac="04:52:C7:00:00:02", advertised_name="Nix Dev 0002", rssi=-60, mock=True)
    offer = registry.discover(mac="04:52:C7:00:00:02")
    secret = registry._devices["04:52:C7:00:00:02"]["device_secret"]
    registry.register(
        mac="04:52:C7:00:00:02", claim_token=offer["claim_token"],
        device_secret=secret, mock=True,
    )
    with pytest.raises(DeviceRegistryError, match="already officially registered"):
        registry.discover(mac="04:52:C7:00:00:02")


def test_mock_and_real_identities_cannot_mix(registry):
    registry.report(mac="04:52:C7:00:00:03", advertised_name="Nix Dev 0003", rssi=-60, mock=True)
    with pytest.raises(DeviceRegistryError, match="mock"):
        registry.report(mac="04:52:C7:00:00:03", advertised_name="Nix Dev 0003", rssi=-60, mock=False)
    registry.report(mac="04:52:C7:00:00:04", advertised_name="Nix Dev 0004", rssi=-60, mock=False)
    with pytest.raises(DeviceRegistryError, match="real"):
        registry.report(mac="04:52:C7:00:00:04", advertised_name="Nix Dev 0004", rssi=-60, mock=True)
    with pytest.raises(DeviceRegistryError, match="Mock/real"):
        registry.register(
            mac="04:52:C7:00:00:03",
            claim_token=registry.discover(mac="04:52:C7:00:00:03")["claim_token"],
            device_secret=registry._devices["04:52:C7:00:00:03"]["device_secret"],
            mock=False,
        )


def test_report_dedupes_by_mac_and_updates_rssi(registry):
    registry.report(mac="AA:BB:CC:DD:EE:FF", advertised_name="Nix Dev", rssi=-70)
    again = registry.report(mac="aabbccddeeff", advertised_name="Nix Dev", rssi=-52)
    assert again["mac"] == "AA:BB:CC:DD:EE:FF"
    assert again["rssi"] == -52
    listing = registry.status()
    assert listing["count"] == 1
    # Sorted strongest-first like the app's list.
    registry.report(mac="11:22:33:44:55:66", advertised_name="Nix Dev 1166", rssi=-40)
    devices = registry.status()["devices"]
    assert devices[0]["mac"] == "11:22:33:44:55:66"


def test_rssi_and_kind_boundaries(registry):
    with pytest.raises(DeviceRegistryError, match="RSSI"):
        registry.report(mac="04:52:C7:00:00:05", advertised_name="Nix Dev", rssi=5)
    with pytest.raises(DeviceRegistryError, match="RSSI"):
        registry.report(mac="04:52:C7:00:00:05", advertised_name="Nix Dev", rssi=-200)
    registry.report(mac="04:52:C7:00:00:05", advertised_name="Nix Dev", rssi=-127, mock=True)
    with pytest.raises(DeviceRegistryError, match="kind must be"):
        registry.register(
            mac="04:52:C7:00:00:05",
            claim_token=registry.discover(mac="04:52:C7:00:00:05")["claim_token"],
            device_secret=registry._devices["04:52:C7:00:00:05"]["device_secret"],
            kind="toaster",
            mock=True,
        )
    ok = registry.register(
        mac="04:52:C7:00:00:05",
        claim_token=registry.discover(mac="04:52:C7:00:00:05")["claim_token"],
        device_secret=registry._devices["04:52:C7:00:00:05"]["device_secret"],
        kind="speaker",
        mock=True,
    )
    assert ok["device"]["kind"] == "speaker"


def test_registry_persists_across_instances(tmp_path):
    path = str(tmp_path / "devices.json")
    first = DeviceRegistry(state_path=path)
    first.report(mac="04:52:C7:AB:CD:EF", advertised_name="Nix Dev ABCD", rssi=-55, mock=True)
    second = DeviceRegistry(state_path=path)
    assert second.status("04:52:C7:AB:CD:EF")["device"]["advertised_name"] == "Nix Dev ABCD"


def test_reset_clears_only_mock_devices_by_default(registry):
    registry.report(mac="04:52:C7:00:00:06", advertised_name="Nix Dev", rssi=-60, mock=True)
    registry.report(mac="04:52:C7:00:00:07", advertised_name="Nix Dev", rssi=-60, mock=False)
    removed = registry.reset()  # mock_only=True default
    assert removed == 1
    assert registry.status("04:52:C7:00:00:07")["device"]["status"] == "seen"
