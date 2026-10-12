"""Official device registration for the NIX Home mobile app.

The NIX Home Android app discovers nearby devices over Bluetooth by an
unfiltered BLE scan and shows every advertiser whose advertised Complete
Local Name contains ``Nix Dev`` (case-insensitive, with the space).
"Pairing" inside the app is UI-only bookkeeping today; this module is the
server-side counterpart that turns a discovered device into an OFFICIALLY
registered device of the NIX system.

Lifecycle (all state persisted in one JSON file):

    seen        the app reported an advertisement; the hub acknowledged it
    claimed     the app claimed the device with a claim token from /discover
    registered  the claim was verified with the device secret -> official
    retired     unregistered; kept for audit, no longer official

Design rules:

- Every input is validated at the boundary with plain-language errors.
- Mock mode (``source: "mock"`` or explicit ``mock: true``) lets tests
  exercise the whole lifecycle with fake addresses/names and NO real
  Bluetooth hardware. Mock devices are stored like real ones but are
  always marked ``mock: true`` and can never be confused with real ones.
- Device identity is the MAC address (upper-cased, colon-separated or
  stripped, consistently normalized).
- The advertised name must contain "Nix Dev" (with the space) — the exact
  rule the app applies, mirrored here so the hub never registers a device
  the app could not have seen.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
from typing import Any

__all__ = [
    "DeviceRegistryError",
    "DeviceRegistry",
    "NIX_NAME_PATTERN",
]

# The one app-side filter, mirrored exactly (PairDeviceScreen.isNixDev).
NIX_NAME_PATTERN = re.compile(r"nix dev", re.IGNORECASE)

_MAC_CLEAN_RE = re.compile(r"[^0-9a-fA-F]")
_MAC_STRICT_RE = re.compile(r"^[0-9A-F]{2}(?::[0-9A-F]{2}){5}$")


class DeviceRegistryError(RuntimeError):
    """Plain-language registration failure shown to the app."""


def normalize_mac(value: Any) -> str:
    """Normalize a Bluetooth MAC to strict AA:BB:CC:DD:EE:FF form."""
    if not isinstance(value, str):
        raise DeviceRegistryError("A Bluetooth MAC address is required.")
    cleaned = _MAC_CLEAN_RE.sub("", value).upper()
    if len(cleaned) != 12:
        raise DeviceRegistryError(
            "The Bluetooth MAC address must have 6 hex bytes, like 04:52:C7:1A:2B:3F."
        )
    normalized = ":".join(cleaned[index : index + 2] for index in range(0, 12, 2))
    return normalized


def valid_advertised_name(value: Any) -> str:
    """The advertised name must exist and contain 'Nix Dev' (with space)."""
    if not isinstance(value, str):
        raise DeviceRegistryError("The advertised device name is required.")
    name = " ".join(value.split())
    if not 1 <= len(name) <= 26:
        raise DeviceRegistryError(
            "The advertised name must be 1 to 26 characters (the BLE legacy "
            "advertising payload leaves ~26 bytes for the name)."
        )
    if any(ord(char) < 32 for char in name):
        raise DeviceRegistryError("The advertised name contains control characters.")
    if not NIX_NAME_PATTERN.search(name):
        raise DeviceRegistryError(
            "Only devices whose advertised name contains 'Nix Dev' "
            "(with the space) can be registered; that is the exact name the "
            "NIX Home app filters for."
        )
    return name


def valid_rssi(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeviceRegistryError("RSSI must be a number in dBm, like -58.")
    rssi = int(value)
    if not -127 <= rssi <= 0:
        raise DeviceRegistryError("RSSI must be between -127 and 0 dBm.")
    return rssi


def valid_device_kind(value: Any) -> str:
    allowed = {"echo_dot", "ring_light", "sensor", "speaker", "other"}
    kind = str(value or "other").strip().lower()
    if kind not in allowed:
        raise DeviceRegistryError(
            f"Device kind must be one of: {', '.join(sorted(allowed))}."
        )
    return kind


class DeviceRegistry:
    """Persisted registry of Bluetooth-discovered NIX devices."""

    def __init__(self, state_path: str | None = None) -> None:
        self.path = state_path or os.environ.get(
            "NIX_MOBILE_DEVICES_PATH",
            os.path.join(
                os.environ.get("NIX_DATA_DIR", "/root/nix_core/data"),
                "mobile_devices.json",
            ),
        )
        self._lock = threading.RLock()
        self._devices: dict[str, dict[str, Any]] = {}
        self._load()

    # ---- persistence -------------------------------------------------
    def _load(self) -> None:
        with self._lock:
            try:
                with open(self.path, "r", encoding="utf-8") as handle:
                    saved = json.load(handle)
                devices = saved.get("devices", {})
                if isinstance(devices, dict):
                    self._devices = devices
            except (OSError, json.JSONDecodeError, AttributeError):
                self._devices = {}

    def _save(self) -> None:
        with self._lock:
            parent = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(parent, mode=0o700, exist_ok=True)
            temporary = f"{self.path}.{secrets.token_hex(6)}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(
                    {"devices": self._devices},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")
            os.replace(temporary, self.path)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass

    # ---- helpers ------------------------------------------------------
    @staticmethod
    def _now() -> float:
        return round(time.time(), 3)

    @staticmethod
    def _public_view(device: dict[str, Any]) -> dict[str, Any]:
        """App-safe projection: never expose the device secret or claim token."""
        hidden = {"device_secret", "claim_token"}
        return {key: value for key, value in device.items() if key not in hidden}

    def _require(self, mac: str, *, statuses: set[str] | None = None) -> dict[str, Any]:
        device = self._devices.get(mac)
        if device is None:
            raise DeviceRegistryError(
                "That device has not been seen yet. Report its Bluetooth "
                "advertisement first (POST /api/mobile/v1/devices/report)."
            )
        if statuses is not None and device["status"] not in statuses:
            raise DeviceRegistryError(
                f"That device is {device['status']}; this action needs one of: "
                f"{', '.join(sorted(statuses))}."
            )
        return device

    # ---- lifecycle ----------------------------------------------------
    def report(
        self,
        *,
        mac: str,
        advertised_name: str,
        rssi: int,
        mock: bool = False,
    ) -> dict[str, Any]:
        """Acknowledge one BLE scan result (dedup by MAC, RSSI updated)."""
        normalized = normalize_mac(mac)
        name = valid_advertised_name(advertised_name)
        rssi = valid_rssi(rssi)
        now = self._now()
        with self._lock:
            device = self._devices.get(normalized)
            if device is None:
                device = {
                    "mac": normalized,
                    "status": "seen",
                    "first_seen": now,
                    "device_secret": secrets.token_hex(16),
                    "claim_token": None,
                    "claimed_at": None,
                    "registered_at": None,
                    "retired_at": None,
                    "label": None,
                    "kind": "other",
                    "mock": bool(mock),
                }
                self._devices[normalized] = device
            if device.get("mock") and not mock:
                raise DeviceRegistryError(
                    "That MAC is already registered as a mock test device and "
                    "cannot be re-reported as a real advertisement."
                )
            if mock and not device.get("mock"):
                raise DeviceRegistryError(
                    "That MAC is already known as a real device and cannot be "
                    "re-reported as a mock."
                )
            device["advertised_name"] = name
            device["rssi"] = rssi
            device["last_seen"] = now
            self._save()
            return self._public_view(device)

    def discover(self, *, mac: str) -> dict[str, Any]:
        """The app asks to pair: mint a short-lived claim token."""
        normalized = normalize_mac(mac)
        with self._lock:
            device = self._require(normalized, statuses={"seen", "registered"})
            if device["status"] == "registered":
                raise DeviceRegistryError(
                    "That device is already officially registered. Unregister "
                    "it first to pair it again."
                )
            token = secrets.token_hex(8).upper()
            device["claim_token"] = token
            device["status"] = "seen"  # still seen until verify
            device["claimed_at"] = self._now()
            self._save()
            return {
                "ok": True,
                "mac": normalized,
                "advertised_name": device["advertised_name"],
                "claim_token": token,
                "hint": (
                    "Show or confirm this token on the device, then call "
                    "/api/mobile/v1/devices/register with the token and the "
                    "device secret to finish registration."
                ),
                "mock": device.get("mock", False),
            }

    def register(
        self,
        *,
        mac: str,
        claim_token: str,
        device_secret: str,
        label: str = "",
        kind: str = "other",
        mock: bool = False,
    ) -> dict[str, Any]:
        """Verify claim token + device secret -> official registration."""
        normalized = normalize_mac(mac)
        kind = valid_device_kind(kind)
        if not isinstance(claim_token, str) or not re.fullmatch(
            r"[0-9A-F]{16}", claim_token.strip().upper()
        ):
            raise DeviceRegistryError(
                "The claim token must be the 16 hex characters given by "
                "/discover."
            )
        if not isinstance(device_secret, str) or len(device_secret.strip()) != 32:
            raise DeviceRegistryError(
                "The device secret must be the 32-character value the hub "
                "issued for this device."
            )
        with self._lock:
            device = self._require(normalized, statuses={"seen"})
            if device.get("mock") != bool(mock):
                raise DeviceRegistryError(
                    "Mock/real mismatch: this device was reported as "
                    f"{'a mock' if device.get('mock') else 'a real'} device."
                )
            if device.get("claim_token") != claim_token.strip().upper():
                raise DeviceRegistryError(
                    "The claim token does not match. Request a fresh one with "
                    "/api/mobile/v1/devices/discover."
                )
            if device.get("device_secret") != device_secret.strip():
                raise DeviceRegistryError(
                    "The device secret does not match this device. Registration "
                    "refused."
                )
            label = " ".join(str(label or "").split())[:48]
            device["status"] = "registered"
            device["label"] = label or device.get("advertised_name")
            device["kind"] = kind
            device["registered_at"] = self._now()
            device["claim_token"] = None  # one-time use
            self._save()
            return {"ok": True, "device": self._public_view(device)}

    def unregister(self, *, mac: str) -> dict[str, Any]:
        normalized = normalize_mac(mac)
        with self._lock:
            device = self._require(normalized, statuses={"registered"})
            device["status"] = "retired"
            device["retired_at"] = self._now()
            self._save()
            return {"ok": True, "device": self._public_view(device)}

    def status(self, mac: str | None = None) -> dict[str, Any]:
        with self._lock:
            if mac is None:
                devices = sorted(
                    (self._public_view(item) for item in self._devices.values()),
                    key=lambda item: (-int(item.get("rssi", -128)), item["mac"]),
                )
                official = [item for item in devices if item["status"] == "registered"]
                return {
                    "ok": True,
                    "count": len(devices),
                    "official_count": len(official),
                    "devices": devices,
                }
            normalized = normalize_mac(mac)
            return {"ok": True, "device": self._public_view(self._require(normalized))}

    # ---- test support ---------------------------------------------------
    def reset(self, *, mock_only: bool = True) -> int:
        """Clear registry entries. Mock-only by default; tests only."""
        with self._lock:
            doomed = [
                mac
                for mac, device in self._devices.items()
                if not mock_only or device.get("mock")
            ]
            for mac in doomed:
                del self._devices[mac]
            if doomed:
                self._save()
            return len(doomed)
