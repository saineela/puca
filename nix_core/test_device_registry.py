"""Tests for the Nix Dev device registry API (server spec: POST /api/devices upsert on mac)."""

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import pytest

import console


@pytest.fixture
def skills_state(monkeypatch, tmp_path):
    monkeypatch.setattr(console, "SKILLS_STATE_PATH", str(tmp_path / "skills.json"))
    monkeypatch.setattr(console, "SKILLS_INSTALL_DIR", str(tmp_path / "installed"))
    monkeypatch.setattr(console, "DATA_DIR", str(tmp_path / "data"))
    return tmp_path


@pytest.fixture
def http_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), console.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _request(base, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = Request(base + path, data=data, method=method,
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode())
    except Exception as exc:  # urllib raises for non-2xx
        status = getattr(exc, "code", None)
        payload = exc.read().decode() if hasattr(exc, "read") else "{}"
        return status, json.loads(payload)


def test_register_creates_then_upserts_on_mac(skills_state, http_server):
    base = http_server
    status, result = _request(base, "POST", "/api/devices/register", {
        "mac": "AA:BB:CC:DD:EE:FF", "device_id": "HHKJ", "name": "Nix Dev HHKJ",
        "model": "Echo Dot 2 (rooted)", "ip": "192.168.0.55",
        "api_key": "cGFzc3dvcmQ=", "skill_id": "echo-connect",
        "owner": "Neela Home",
    })
    assert status == 200
    assert result["ok"] is True and result["created"] is True
    device = result["device"]
    assert device["id"] == "echo-connect:aabbccddeeff"
    assert device["mac"] == "AA:BB:CC:DD:EE:FF"
    assert device["device_label"] == "Nix Dev HHKJ"
    assert device["ip"] == "192.168.0.55"
    assert device["owner"] == "Neela Home"
    assert device["icon"] == "🔊"
    assert "api_key" not in device and "api_key_digest" not in device

    # Upsert on the same mac: no duplicate, fields updated, created=False.
    status, result = _request(base, "POST", "/api/devices/register", {
        "mac": "aa:bb:cc:dd:ee:ff", "ip": "192.168.0.99", "name": "Renamed Dev",
    })
    assert status == 200 and result["created"] is False
    assert result["device"]["id"] == "echo-connect:aabbccddeeff"
    assert result["device"]["ip"] == "192.168.0.99"
    assert result["device"]["device_label"] == "Renamed Dev"
    assert result["device"]["owner"] == "Neela Home"  # untouched

    # Distinct mac => distinct row.
    _request(base, "POST", "/api/devices/register", {"mac": "11:22:33:44:55:66"})
    _status, listing = _request(base, "GET", "/api/devices")
    assert _status == 200 and listing["ok"] is True
    macs = {d["mac"] for d in listing["devices"] if d.get("registered")}
    assert macs == {"AA:BB:CC:DD:EE:FF", "11:22:33:44:55:66"}


def test_registered_devices_appear_in_devices_list_and_never_echo_api_key(skills_state, http_server):
    base = http_server
    _request(base, "POST", "/api/devices/register", {
        "mac": "AA:BB:CC:DD:EE:FF", "api_key": "top-secret-psk", "ip": "192.168.0.55",
    })
    _status, listing = _request(base, "GET", "/api/devices")
    assert listing["ok"] is True
    match = next(d for d in listing["devices"] if d.get("mac") == "AA:BB:CC:DD:EE:FF")
    assert match["registered"] is True
    assert match["device_status"]["device_connected"] is False
    assert match["device_ui"]["controls"][0]["tool"] == "control_ring"
    assert "api_key" not in json.dumps(listing)
    assert "top-secret-psk" not in json.dumps(listing)

    # On-disk state stores only a keyed digest, never the psk.
    saved = json.loads(open(console.SKILLS_STATE_PATH, encoding="utf-8").read())
    digest = saved["device_registry"]["AA:BB:CC:DD:EE:FF"]["api_key_digest"]
    assert digest and "top-secret-psk" not in json.dumps(saved)


def test_get_single_device_and_404(skills_state, http_server):
    base = http_server
    _request(base, "POST", "/api/devices/register", {"mac": "AA:BB:CC:DD:EE:FF", "ip": "192.168.0.55"})
    status, result = _request(base, "GET", "/api/devices/echo-connect:aabbccddeeff")
    assert status == 200 and result["device"]["mac"] == "AA:BB:CC:DD:EE:FF"
    # mac is accepted as the path id too
    status, result = _request(base, "GET", "/api/devices/AA:BB:CC:DD:EE:FF")
    assert status == 200 and result["device"]["id"] == "echo-connect:aabbccddeeff"
    status, result = _request(base, "GET", "/api/devices/nope")
    assert status == 404 and result == {"ok": False, "error": "No such device."}


def test_patch_updates_alias(skills_state, http_server):
    base = http_server
    _request(base, "POST", "/api/devices/register", {"mac": "AA:BB:CC:DD:EE:FF"})
    status, result = _request(base, "PATCH", "/api/devices/echo-connect:aabbccddeeff",
                              {"device_alias": "Kitchen Dot"})
    assert status == 200 and result["ok"] is True
    assert result["device"]["device_alias"] == "Kitchen Dot"
    assert result["device"]["device_label"] == "Kitchen Dot"
    _status, listing = _request(base, "GET", "/api/devices")
    match = next(d for d in listing["devices"] if d.get("mac") == "AA:BB:CC:DD:EE:FF")
    assert match["device_alias"] == "Kitchen Dot"


def test_delete_removes_device(skills_state, http_server):
    base = http_server
    _request(base, "POST", "/api/devices/register", {"mac": "AA:BB:CC:DD:EE:FF"})
    status, result = _request(base, "DELETE", "/api/devices/AA:BB:CC:DD:EE:FF")
    assert status == 200 and result == {"ok": True}
    _status, listing = _request(base, "GET", "/api/devices")
    assert all(d.get("mac") != "AA:BB:CC:DD:EE:FF" for d in listing["devices"])
    status, result = _request(base, "DELETE", "/api/devices/AA:BB:CC:DD:EE:FF")
    assert status == 404 and result == {"ok": False, "error": "No such device."}


def test_register_validates_payload(skills_state, http_server):
    base = http_server
    status, result = _request(base, "POST", "/api/devices/register", {})
    assert status == 400 and result["ok"] is False and "mac" in result["error"]
    status, result = _request(base, "POST", "/api/devices/register", {"mac": "not-a-mac"})
    assert status == 400 and "MAC" in result["error"]
    status, result = _request(base, "PATCH", "/api/devices/ghost", {"device_alias": "x"})
    assert status == 404 and result["error"] == "No such device."


def test_registry_payload_matches_skill_device_shape(skills_state):
    """_devices_payload must merge registry rows shaped like skill devices."""
    console._device_registry_upsert({
        "mac": "AA:BB:CC:DD:EE:FF", "device_id": "HHKJ", "ip": "192.168.0.55",
        "name": "Nix Dev HHKJ",
    })
    payload = console._devices_payload()
    assert payload["ok"] is True
    match = next(d for d in payload["devices"] if d.get("mac") == "AA:BB:CC:DD:EE:FF")
    # parseDevice-required keys
    assert match["id"] == "echo-connect:aabbccddeeff"
    assert match["device_label"] == "Nix Dev HHKJ"
    assert match["icon"]
    assert isinstance(match["device_status"], dict)
    assert isinstance(match["device_ui"], dict)
