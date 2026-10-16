"""Tests for the skill-contributed music page and the music control API.

The console owns the page shell, the routes and the state; installed skills
supply the catalog (provider) and the speaker controls (player). These tests
pin that contract without any hardware or network access.
"""
import json
import time
import urllib.parse
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import console
import console_extend
import music_api
import music_cache
import music_queue
import music_queue_service


@pytest.fixture
def music_env(monkeypatch, tmp_path):
    monkeypatch.setattr(console, "SKILLS_STATE_PATH", str(tmp_path / "skills.json"))
    monkeypatch.setattr(console, "SKILLS_INSTALL_DIR", str(tmp_path / "installed"))
    monkeypatch.setattr(console, "DATA_DIR", str(tmp_path / "data"))
    # The queue's own files follow the console's data directory, so pointing that
    # at this test's directory is what keeps a queue service running on the box
    # (this one publishes its address in the real data directory) from answering
    # these tests' reads.
    monkeypatch.setenv("NIX_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


PROVIDER_MANIFEST = {
    "schema_version": 1,
    "kind": "nix-skill",
    "id": "opendeck",
    "name": "OpenDeck",
    "version": "1.1.0",
    "description": "Music provider used by the music page tests.",
    "category": "Music",
    "publisher": "NIX Community Skills",
    "license": "MIT",
    "files": ["opendeck_worker.py", "ui/music.js", "ui/music.css"],
    "entrypoint": "opendeck_worker.py",
    "runtime": {"protocol": "nix-skill-jsonl-v1", "entrypoint": "opendeck_worker.py"},
    "setup_fields": [{"id": "server_url", "label": "Server", "type": "string", "required": True}],
    "tools": [
        {
            "name": "opendeck_music",
            "description": "Search and play music, read the library, and fetch lyrics.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["search", "play", "library", "lyrics", "queue", "save"]},
                    "query": {"type": "string", "maxLength": 200},
                    "video_id": {"type": "string", "maxLength": 64},
                    "kind": {"type": "string", "enum": ["playlists", "playlist", "likes", "history", "stats", "like", "unlike", "playlist_create", "playlist_add"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 12},
                    "op": {"type": "string", "enum": ["list", "add", "remove", "clear", "next", "play", "shuffle", "repeat"]},
                    "index": {"type": "integer", "minimum": 0, "maximum": 99},
                    "on": {"type": "boolean"},
                    "mode": {"type": "string", "enum": ["off", "all", "one"]},
                    "auto": {"type": "boolean"},
                    "playlist_id": {"type": "string", "maxLength": 100},
                    "name": {"type": "string", "maxLength": 120},
                    "title": {"type": "string", "maxLength": 300},
                    "artist": {"type": "string", "maxLength": 200},
                    "album": {"type": "string", "maxLength": 300},
                },
                "required": ["action"],
                "additionalProperties": False,
                "oneOf": [
                    {"type": "object", "properties": {"action": {"type": "string", "enum": ["search"]}, "query": {"type": "string", "minLength": 1, "maxLength": 200}, "limit": {"type": "integer", "minimum": 1, "maximum": 12}}, "required": ["action", "query"]},
                    {"type": "object", "properties": {"action": {"type": "string", "enum": ["play"]}, "query": {"type": "string", "maxLength": 200}, "video_id": {"type": "string", "maxLength": 64}}, "required": ["action"]},
                    {"type": "object", "properties": {"action": {"type": "string", "enum": ["library"]}, "kind": {"type": "string", "enum": ["playlists", "likes", "history"]}, "limit": {"type": "integer", "minimum": 1, "maximum": 12}}, "required": ["action", "kind"]},
                    {"type": "object", "properties": {"action": {"type": "string", "enum": ["lyrics"]}, "video_id": {"type": "string", "maxLength": 64}}, "required": ["action"]},
                    {"type": "object", "properties": {"action": {"type": "string", "enum": ["queue"]}, "op": {"type": "string", "enum": ["list", "add", "remove", "clear", "next", "play", "shuffle", "repeat", "add_range"]}, "query": {"type": "string", "minLength": 1, "maxLength": 200}, "video_id": {"type": "string", "maxLength": 64}, "playlist_id": {"type": "string", "maxLength": 100}, "index": {"type": "integer", "minimum": 0, "maximum": 99}, "on": {"type": "boolean"}, "mode": {"type": "string", "enum": ["off", "all", "one"]}, "auto": {"type": "boolean"}}, "required": ["action", "op"]},
                    {"type": "object", "properties": {"action": {"type": "string", "enum": ["save"]}, "kind": {"type": "string", "enum": ["like", "unlike", "playlist_create", "playlist_add"]}, "video_id": {"type": "string", "maxLength": 64}, "playlist_id": {"type": "string", "maxLength": 100}, "name": {"type": "string", "maxLength": 120}, "title": {"type": "string", "maxLength": 300}, "artist": {"type": "string", "maxLength": 200}, "album": {"type": "string", "maxLength": 300}}, "required": ["action", "kind"]},
                ],
            },
            "result_schema": {
                "type": "object",
                "properties": {
                    "ok": {"type": "boolean"},
                    "action": {"type": "string", "maxLength": 32},
                    "message": {"type": "string", "maxLength": 400},
                    "video_id": {"type": "string", "maxLength": 64},
                    "title": {"type": "string", "maxLength": 300},
                    "artist": {"type": "string", "maxLength": 200},
                    "duration_sec": {"type": "integer", "minimum": 0, "maximum": 100000},
                    "artwork_url": {"type": "string", "maxLength": 600},
                    "mime_type": {"type": "string", "maxLength": 80},
                    "play_url": {"type": "string", "maxLength": 2000},
                    "tracks": {"type": "array", "maxItems": 12, "items": {"type": "object", "properties": {"video_id": {"type": "string", "maxLength": 64}, "title": {"type": "string", "maxLength": 300}, "artist": {"type": "string", "maxLength": 200}, "duration_sec": {"type": "integer", "minimum": 0, "maximum": 100000}, "artwork_url": {"type": "string", "maxLength": 600}}, "required": ["video_id", "title", "artist", "duration_sec", "artwork_url"], "additionalProperties": False}},
                    "queue": {"type": "array", "maxItems": 100, "items": {"type": "object", "properties": {"video_id": {"type": "string", "maxLength": 64}, "title": {"type": "string", "maxLength": 300}, "artist": {"type": "string", "maxLength": 200}, "duration_sec": {"type": "integer", "minimum": 0, "maximum": 100000}, "artwork_url": {"type": "string", "maxLength": 600}}, "required": ["video_id", "title", "artist", "duration_sec", "artwork_url"], "additionalProperties": False}},
                    "queue_index": {"type": "integer", "minimum": -1, "maximum": 100},
                    "shuffle": {"type": "boolean"},
                    "repeat": {"type": "string", "maxLength": 8},
                    "playlists": {"type": "array", "maxItems": 12, "items": {"type": "object", "properties": {"playlist_id": {"type": "string", "maxLength": 100}, "name": {"type": "string", "maxLength": 300}, "track_count": {"type": "integer", "minimum": 0, "maximum": 100000}}, "required": ["playlist_id", "name", "track_count"], "additionalProperties": False}},
                    "count": {"type": "integer", "minimum": 0, "maximum": 100000},
                    "lyrics": {"type": "array", "maxItems": 400, "items": {"type": "object", "properties": {"time_sec": {"type": "number", "minimum": 0, "maximum": 100000}, "text": {"type": "string", "maxLength": 400}}, "required": ["time_sec", "text"], "additionalProperties": False}},
                    "lyrics_plain": {"type": "string", "maxLength": 8000},
                    "lyrics_timing": {"type": "string", "maxLength": 16},
                    "lyrics_source": {"type": "string", "maxLength": 40},
                },
                "required": ["ok", "action", "message", "video_id", "title", "artist", "duration_sec", "artwork_url", "mime_type", "play_url", "tracks", "playlists", "queue", "queue_index", "shuffle", "repeat", "count", "lyrics", "lyrics_plain", "lyrics_timing", "lyrics_source"],
                "additionalProperties": False,
            },
        }
    ],
    "device_ui": {
        "page": {"id": "music", "title": "Music", "script": "ui/music.js", "styles": ["ui/music.css"]},
        "media": {
            "provider": {
                "search": {"tool": "opendeck_music", "arguments": {"action": "search"}},
                "play": {"tool": "opendeck_music", "arguments": {"action": "play"}},
                "library": {"tool": "opendeck_music", "arguments": {"action": "library", "kind": "playlists"}},
                "lyrics": {"tool": "opendeck_music", "arguments": {"action": "lyrics"}},
                "queue": {"tool": "opendeck_music", "arguments": {"action": "queue", "op": "list"}},
                "save": {"tool": "opendeck_music", "arguments": {"action": "save"}},
            }
        },
    },
}


def player_manifest(upload_control=True, transport_controls=False):
    controls = [
        {"label": "Set volume", "tool": "control_ring", "arguments": {"action": "volume"}, "fields": [{"name": "volume", "label": "Volume", "widget": "number"}]},
        {"label": "Mute", "tool": "control_ring", "arguments": {"action": "mute"}, "fields": []},
        {"label": "Stop", "tool": "control_ring", "arguments": {"action": "stop_speaking"}, "fields": []},
    ]
    if upload_control:
        controls.append(
            {"label": "Play a stream URL", "tool": "control_ring", "arguments": {"action": "play_url"}, "fields": [{"name": "url", "label": "Audio URL", "widget": "text"}, {"name": "title", "label": "Title", "widget": "text"}]}
        )
    # A speaker that can pause and resume the music, the way Echo Connect does.
    if transport_controls:
        controls.extend([
            {"label": "Pause", "tool": "control_ring", "arguments": {"action": "pause"}, "fields": []},
            {"label": "Resume", "tool": "control_ring", "arguments": {"action": "resume"}, "fields": []},
            {"label": "Unmute", "tool": "control_ring", "arguments": {"action": "unmute"}, "fields": []},
        ])
    action_enum = ["state", "volume", "mute", "stop_speaking", "play_url"]
    zero_argument = ["state", "mute", "stop_speaking"]
    if transport_controls:
        action_enum.extend(["pause", "resume", "unmute"])
        zero_argument.extend(["pause", "resume", "unmute"])
    return {
        "schema_version": 1,
        "kind": "nix-skill",
        "id": "echo-connect",
        "name": "Echo Connect",
        "version": "1.4.0",
        "description": "Speaker and ring control used by the music page tests.",
        "category": "Devices",
        "publisher": "NIX Community Skills",
        "license": "MIT",
        "files": ["ring_light_worker.py"],
        "entrypoint": "ring_light_worker.py",
        "runtime": {"protocol": "nix-skill-jsonl-v1", "entrypoint": "ring_light_worker.py"},
        "setup_fields": [{"id": "device_ip", "label": "Address", "type": "ipv4", "required": True}],
        "tools": [
            {
                "name": "control_ring",
                "description": "Control the configured speaker and ring.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": action_enum},
                        "volume": {"type": "number", "minimum": 0, "maximum": 1},
                        "url": {"type": "string", "maxLength": 2048},
                        "title": {"type": "string", "maxLength": 200},
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                    "oneOf": [
                        {"type": "object", "properties": {"action": {"type": "string", "enum": zero_argument}}, "required": ["action"]},
                        {"type": "object", "properties": {"action": {"type": "string", "enum": ["volume"]}, "volume": {"type": "number", "minimum": 0, "maximum": 1}}, "required": ["action", "volume"]},
                        {"type": "object", "properties": {"action": {"type": "string", "enum": ["play_url"]}, "url": {"type": "string", "maxLength": 2048}, "title": {"type": "string", "maxLength": 200}}, "required": ["action", "url"]},
                    ],
                },
                "result_schema": {
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean"},
                        "action": {"type": "string", "maxLength": 32},
                        "message": {"type": "string", "maxLength": 400},
                        "device_name": {"type": "string", "maxLength": 80},
                        "speaker": {"type": "object", "properties": {"volume": {"type": "number", "minimum": 0, "maximum": 1}, "muted": {"type": "boolean"}, "state": {"type": "string", "maxLength": 24}}, "required": ["volume", "muted", "state"], "additionalProperties": False},
                        "state": {"type": "object", "properties": {"on": {"type": "boolean"}, "brightness": {"type": "number", "minimum": 0, "maximum": 1}, "effect": {"type": "string", "maxLength": 64}, "rgb": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "integer", "minimum": 0, "maximum": 255}}}, "required": ["on", "brightness", "effect", "rgb"], "additionalProperties": False},
                    },
                    "required": ["ok", "action", "message", "device_name", "speaker", "state"],
                    "additionalProperties": False,
                },
            }
        ],
        "device_ui": {
            "status_tool": "control_ring",
            "status_arguments": {"action": "state"},
            "visualization": {"type": "status"},
            "fields": [],
            "controls": controls,
        },
    }


class FakeRuntime:
    """Records every tool call and answers with scripted results."""

    def __init__(self, results=None, player=None):
        self.calls = []
        self.started = []
        self.results = results or {}
        self.player = player if player is not None else {
            "ok": True, "action": "state", "message": "reported",
            "device_name": "Light", "speaker": {"volume": 0.5, "muted": False, "state": "playing"},
            "state": {"on": True, "brightness": 1.0, "effect": "None", "rgb": [255, 255, 255]},
        }

    def status(self, skill_id):
        return {"configured": True, "trusted": True, "worker_running": True, "runnable": True, "fingerprint": "digest"}

    def start_skill(self, skill_id):
        self.started.append(skill_id)
        return self.status(skill_id)

    def device_status(self, skill_id):
        self.calls.append({"skill_id": skill_id, "tool": "control_ring", "arguments": {"action": "state"}})
        return self.player

    def validate_proposal(self, proposal):
        return {"skill_id": proposal["skill_id"], "tool": {"name": proposal["tool"]}, "arguments": proposal["arguments"]}

    def execute(self, matched):
        self.calls.append(matched)
        tool = matched["tool"]["name"] if isinstance(matched.get("tool"), dict) else matched.get("tool")
        result = self.results.get(tool, {})
        if callable(result):
            result = result(matched)
        return {"ok": True, "result": result, "skill_id": matched["skill_id"], "tool": tool, "arguments": matched["arguments"]}


def install(monkeypatch, tmp_path, manifests, runtime):
    console._write_skills_state({
        "repositories": [{"key": "nix/test", "owner": "nix", "repository": "test", "default_branch": "main",
                          "skills": [{"id": manifest["id"], "slug": manifest["id"], "name": manifest["name"],
                                      "version": manifest["version"], "device_ui": manifest.get("device_ui"),
                                      "runtime": manifest.get("runtime"), "tools": manifest.get("tools"),
                                      "files": manifest["files"]} for manifest in manifests]}],
        "installed": [manifest["id"] for manifest in manifests],
        "trusted": {manifest["id"]: {"fingerprint": "digest"} for manifest in manifests},
    })
    for manifest in manifests:
        package = Path(console._skill_package_path(manifest["id"]))
        package.mkdir(parents=True, exist_ok=True)
        (package / "skill.json").write_text(json.dumps(manifest), encoding="utf-8")
        for relative in manifest["files"]:
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("// " + relative + "\n", encoding="utf-8")
    monkeypatch.setattr(console, "_marketplace_skill_runtime", lambda: runtime)
    return runtime


def installed_ids():
    """The ids the console actually stored, which the routes address."""
    return {skill["slug"]: skill["id"] for _repository, skill in console._remote_skill_refs(console._read_skills_state())}


def serve():
    server = ThreadingHTTPServer(("127.0.0.1", 0), console_extend.ExtendedHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server


def read_json(server, path, payload=None):
    if payload is None:
        request = Request(f"http://127.0.0.1:{server.server_address[1]}{path}")
    else:
        request = Request(
            f"http://127.0.0.1:{server.server_address[1]}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode()), dict(response.headers)
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode()), dict(exc.headers)


def read_raw(server, path):
    with urlopen(f"http://127.0.0.1:{server.server_address[1]}{path}", timeout=10) as response:
        return response.status, response.read().decode(), dict(response.headers)


def read_bytes(server, path, headers=None):
    """A binary read: covers are images, and a 304 has no body at all."""
    request = Request(f"http://127.0.0.1:{server.server_address[1]}{path}", headers=headers or {})
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, response.read(), dict(response.headers)
    except HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


# ── manifest contract ────────────────────────────────────────────────────


def test_manifest_accepts_page_and_media_provider():
    parsed = console._parse_skill_manifest(json.loads(json.dumps(PROVIDER_MANIFEST)), "skills/opendeck/skill.json", "NIX Community Skills")
    assert parsed["device_ui"]["page"] == {"id": "music", "title": "Music", "script": "ui/music.js", "styles": ["ui/music.css"], "assets": []}
    assert set(parsed["device_ui"]["media"]["provider"]) == {"search", "play", "library", "lyrics", "queue", "save"}
    assert parsed["device_ui"]["media"]["provider"]["library"]["arguments"] == {"action": "library", "kind": "playlists"}
    # The queue action carries a default op so the console can read the queue
    # with no request body at all.
    assert parsed["device_ui"]["media"]["provider"]["queue"]["arguments"] == {"action": "queue", "op": "list"}


def test_a_page_may_ship_the_fonts_and_pictures_it_loads():
    """A page's typography travels in its own package, like the wall panel's.

    The stylesheet a page declares points at faces and marks in the same package,
    so the console serves those too - declared as page assets, never from a CDN.
    """
    manifest = json.loads(json.dumps(PROVIDER_MANIFEST))
    manifest["files"] += ["PUCA-Panel/fonts.css", "PUCA-Panel/fonts/outfit-latin.woff2", "PUCA-Panel/puca-logo.png"]
    manifest["device_ui"]["page"]["styles"].append("PUCA-Panel/fonts.css")
    manifest["device_ui"]["page"]["assets"] = ["PUCA-Panel/fonts/outfit-latin.woff2", "PUCA-Panel/puca-logo.png"]
    parsed = console._parse_skill_manifest(manifest, "skills/opendeck/skill.json", "NIX Community Skills")
    page = parsed["device_ui"]["page"]
    assert page["assets"] == ["PUCA-Panel/fonts/outfit-latin.woff2", "PUCA-Panel/puca-logo.png"]
    assert "PUCA-Panel/fonts/outfit-latin.woff2" in music_api.page_assets("opendeck", page, manifest)


def test_a_page_asset_must_be_declared_in_the_files_list():
    manifest = json.loads(json.dumps(PROVIDER_MANIFEST))
    manifest["device_ui"]["page"]["assets"] = ["PUCA-Panel/fonts/outfit-latin.woff2"]
    with pytest.raises(console.SkillRepositoryError, match="files list"):
        console._parse_skill_manifest(manifest, "skills/opendeck/skill.json", "NIX Community Skills")


def test_a_page_asset_may_not_be_code():
    manifest = json.loads(json.dumps(PROVIDER_MANIFEST))
    manifest["files"].append("ui/extra.js")
    manifest["device_ui"]["page"]["assets"] = ["ui/extra.js"]
    with pytest.raises(console.SkillRepositoryError, match="fonts or pictures"):
        console._parse_skill_manifest(manifest, "skills/opendeck/skill.json", "NIX Community Skills")


def test_a_page_support_file_is_served_with_its_own_content_type(music_env, monkeypatch):
    manifest = json.loads(json.dumps(PROVIDER_MANIFEST))
    manifest["files"] += ["PUCA-Panel/fonts.css", "PUCA-Panel/fonts/outfit-latin.woff2"]
    manifest["device_ui"]["page"]["styles"].append("PUCA-Panel/fonts.css")
    manifest["device_ui"]["page"]["assets"] = ["PUCA-Panel/fonts/outfit-latin.woff2"]
    install(monkeypatch, music_env, [manifest], FakeRuntime())
    package = Path(console._skill_package_path("opendeck"))
    (package / "PUCA-Panel" / "fonts").mkdir(parents=True, exist_ok=True)
    (package / "PUCA-Panel" / "fonts" / "outfit-latin.woff2").write_bytes(b"wOF2font bytes")
    server = serve()
    try:
        status, body, headers = read_bytes(server, "/api/skills/opendeck/page/PUCA-Panel/fonts/outfit-latin.woff2")
        missing_status, _body, _headers = read_bytes(server, "/api/skills/opendeck/page/PUCA-Panel/fonts/nope.woff2")
    finally:
        server.shutdown()
    assert status == 200 and body == b"wOF2font bytes"
    assert headers["Content-Type"] == "font/woff2"
    assert missing_status == 404, "an undeclared file stays unreachable"


def test_manifest_rejects_a_page_asset_that_is_not_declared():
    manifest = json.loads(json.dumps(PROVIDER_MANIFEST))
    manifest["files"] = ["opendeck_worker.py"]
    with pytest.raises(console.SkillRepositoryError, match="files list"):
        console._parse_skill_manifest(manifest, "skills/opendeck/skill.json", "NIX Community Skills")


def test_manifest_rejects_an_unknown_provider_action():
    manifest = json.loads(json.dumps(PROVIDER_MANIFEST))
    manifest["device_ui"]["media"]["provider"]["delete_everything"] = {"tool": "opendeck_music", "arguments": {"action": "search"}}
    with pytest.raises(console.SkillRepositoryError, match="unsupported"):
        console._parse_skill_manifest(manifest, "skills/opendeck/skill.json", "NIX Community Skills")


def test_manifest_rejects_a_page_for_a_console_page_that_does_not_exist():
    manifest = json.loads(json.dumps(PROVIDER_MANIFEST))
    manifest["device_ui"]["page"]["id"] = "admin"
    with pytest.raises(console.SkillRepositoryError, match="console page"):
        console._parse_skill_manifest(manifest, "skills/opendeck/skill.json", "NIX Community Skills")


def test_devices_page_skips_a_page_only_skill(music_env):
    provider = json.loads(json.dumps(PROVIDER_MANIFEST))
    console._write_skills_state({
        "repositories": [{"key": "nix/test", "owner": "nix", "repository": "test", "default_branch": "main",
                          "skills": [{"id": provider["id"], "slug": provider["id"], "name": provider["name"],
                                      "device_ui": provider["device_ui"], "runtime": provider["runtime"], "tools": provider["tools"]}]}],
        "installed": [provider["id"]],
    })
    package = Path(console._skill_package_path(provider["id"]))
    package.mkdir(parents=True, exist_ok=True)
    (package / "skill.json").write_text(json.dumps(provider), encoding="utf-8")
    assert console._devices_payload()["devices"] == []


# ── page + API ───────────────────────────────────────────────────────────


def test_music_index_lists_the_page_provider_and_player(music_env, monkeypatch):
    runtime = FakeRuntime()
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music")
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is True
    assert payload["page"]["id"] == "music"
    assert payload["page"]["script"].endswith("/page/ui/music.js")
    assert payload["page"]["skill_id"] == payload["provider"]["skill_id"]
    assert payload["page"]["styles"][0].endswith("/page/ui/music.css")
    assert payload["provider"]["skill_id"].endswith("opendeck")
    assert payload["player"]["commands"]["volume"] is True
    assert payload["player"]["commands"]["play"] is False
    assert payload["player"]["commands"]["stop"] is True


def test_music_page_asset_serves_only_declared_assets(music_env, monkeypatch):
    install(monkeypatch, music_env, [PROVIDER_MANIFEST], FakeRuntime())
    skill_id = installed_ids()["opendeck"]
    server = serve()
    try:
        status, body, headers = read_raw(server, f"/api/skills/{skill_id}/page/ui/music.js")
        assert status == 200
        assert headers["Content-Type"].startswith("application/javascript")
        assert body.startswith("// ui/music.js")
        status, body, headers = read_raw(server, f"/api/skills/{skill_id}/page/ui/music.css")
        assert status == 200 and headers["Content-Type"].startswith("text/css")
        for refused in ("skill.json", "../skill.json", "ui/music.json"):
            with pytest.raises(HTTPError) as undeclared:
                read_raw(server, f"/api/skills/{skill_id}/page/{refused}")
            assert undeclared.value.code == 404
    finally:
        server.shutdown()


def test_now_playing_merges_the_recorded_track_with_the_player(music_env, monkeypatch):
    runtime = FakeRuntime()
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    Path(console.DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(music_api._state_path()).write_text(json.dumps({
        "track": {"video_id": "BSTsnWoslP4", "title": "Bohemian Rhapsody", "artist": "Queen", "duration_sec": 354},
        "started_at": 1_000.0, "paused_at": None, "paused_seconds": 0.0, "duration_sec": 354,
        "handoff": {"supported": False, "reason": "no URL control"},
    }), encoding="utf-8")
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/now-playing")
        alias_status, alias_payload, _alias_headers = read_json(server, "/api/mobile/v1/music/now-playing")
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is True
    assert payload["track"]["title"] == "Bohemian Rhapsody"
    assert payload["playing"] is True
    assert payload["player"]["state"] == "playing"
    assert payload["player"]["volume"] == 0.5
    assert payload["player"]["device"] == "Light"
    assert payload["position"]["duration_sec"] == 354
    assert payload["position"]["estimated"] is True
    assert payload["handoff"]["supported"] is False
    assert alias_payload["track"] == payload["track"]


def test_play_records_the_track_and_reports_an_unavailable_handoff(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": {
        "ok": True, "action": "play", "message": "serving", "video_id": "abc12345678", "title": "Song",
        "artist": "Artist", "duration_sec": 200, "artwork_url": "http://art", "mime_type": "audio/mp4",
        "play_url": "http://192.168.0.152:4100/api/media/stream?url=x", "tracks": [], "playlists": [], "count": 1,
        "lyrics": [], "lyrics_plain": "", "lyrics_timing": "", "lyrics_source": "",
    }})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest(upload_control=False)], runtime)
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/play", {"query": "Song"})
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is True
    assert payload["track"]["video_id"] == "abc12345678"
    assert payload["handoff"]["supported"] is False
    call = next(item for item in runtime.calls if item["arguments"].get("action") == "play")
    assert call["arguments"] == {"action": "play", "query": "Song"}
    saved = json.loads(Path(music_api._state_path()).read_text(encoding="utf-8"))
    assert saved["track"]["title"] == "Song"
    assert saved["duration_sec"] == 200


def test_play_hands_the_stream_url_to_the_declared_play_url_control(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": {
        "ok": True, "action": "play", "message": "serving", "video_id": "abc12345678", "title": "Song",
        "artist": "Artist", "duration_sec": 200, "artwork_url": "", "mime_type": "audio/mp4",
        "play_url": "http://192.168.0.152:4100/api/media/stream?url=x", "tracks": [], "playlists": [], "count": 1,
        "lyrics": [], "lyrics_plain": "", "lyrics_timing": "", "lyrics_source": "",
    }}, player={"ok": True, "action": "play_url", "message": "playing", "device_name": "Light",
               "speaker": {"volume": 0.5, "muted": False, "state": "playing"},
               "state": {"on": True, "brightness": 1.0, "effect": "None", "rgb": [255, 255, 255]}})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/play", {"query": "Song"})
    finally:
        server.shutdown()
    assert status == 200
    assert payload["handoff"]["supported"] is True
    handoff = next(item for item in runtime.calls if item["arguments"].get("action") == "play_url")
    assert handoff["arguments"]["url"].startswith("http://192.168.0.152:4100/api/media/stream?")
    assert handoff["arguments"]["title"] == "Song"


def test_a_stream_the_speaker_refused_is_not_recorded_as_a_successful_handoff(music_env, monkeypatch):
    # The play_url control can match and the call can still come back ok:false
    # (the Dot refusing the stream). Reporting that as playback is exactly the
    # "Playing …" reply with nothing on the speaker that was measured live, so
    # the refusal travels with the handoff and the reply has to say so.
    runtime = FakeRuntime(results={"opendeck_music": {
        "ok": True, "action": "play", "message": "serving", "video_id": "abc12345678", "title": "Song",
        "artist": "Artist", "duration_sec": 200, "artwork_url": "", "mime_type": "audio/mp4",
        "play_url": "http://192.168.0.152:4100/api/media/stream?url=x", "tracks": [], "playlists": [], "count": 1,
        "lyrics": [], "lyrics_plain": "", "lyrics_timing": "", "lyrics_source": "",
    }, "control_ring": {"ok": False, "error": "the Dot could not decode that stream"}})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/play", {"query": "Song"})
    finally:
        server.shutdown()
    assert status == 200
    assert payload["handoff"]["supported"] is True
    assert "could not decode" in payload["handoff"]["error"]
    saved = json.loads(Path(music_api._state_path()).read_text(encoding="utf-8"))
    assert saved["handoff"]["error"], saved["handoff"]


def test_control_routes_volume_to_the_skill_that_declares_it(music_env, monkeypatch):
    runtime = FakeRuntime()
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/control", {"action": "volume", "value": 0.35})
        missing_status, missing, _h = read_json(server, "/api/music/control", {"action": "pause"})
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is True
    assert payload["command"] == "volume"
    call = next(item for item in runtime.calls if item["arguments"].get("action") == "volume")
    assert call["arguments"] == {"action": "volume", "volume": 0.35}
    assert missing_status == 501 and missing["ok"] is False
    assert missing["code"] == "player_action_unavailable"


def test_apply_transport_runs_the_declared_pause_and_resume_controls(music_env, monkeypatch):
    # The page, the mobile app and the assistant all drive one speaker through
    # this helper, so a spoken "pause" and the panel's pause button end up on
    # the same declared control, with the console's position clock kept in step.
    import time

    runtime = FakeRuntime(player={
        "ok": True, "action": "pause", "message": "paused", "device_name": "Light",
        "speaker": {"volume": 0.5, "muted": False, "state": "paused"},
        "state": {"on": True, "brightness": 1.0, "effect": "None", "rgb": [255, 255, 255]},
    })
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest(transport_controls=True)], runtime)
    Path(console.DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(music_api._state_path()).write_text(json.dumps({
        "track": {"video_id": "abc12345678", "title": "Song", "artist": "Artist", "duration_sec": 200},
        "started_at": time.time() - 30, "paused_at": None, "paused_seconds": 0.0, "duration_sec": 200,
    }), encoding="utf-8")

    payload = music_api.apply_transport("pause")

    assert payload["ok"] is True
    assert payload["command"] == "pause"
    assert payload["control"] == "Pause"
    assert payload["message"] == "Music paused."
    call = next(item for item in runtime.calls if item["arguments"].get("action") == "pause")
    assert call["arguments"] == {"action": "pause"}
    assert payload["player"]["state"] == "paused"
    saved = json.loads(Path(music_api._state_path()).read_text(encoding="utf-8"))
    assert isinstance(saved["paused_at"], (int, float))
    assert saved["paused_seconds"] == 0.0


def test_apply_transport_keeps_stop_meaning_the_speakers_voice(music_env, monkeypatch):
    # A speaker pauses; it does not stop. "stop" is the speaker's own voice
    # control, so the music never maps onto it and the confirmation says what
    # really ran — the assistant translates "stop the music" into pause itself
    # (see the transport tests in test_conversation.py).
    runtime = FakeRuntime(player={
        "ok": True, "action": "stop_speaking", "message": "quiet", "device_name": "Light",
        "speaker": {"volume": 0.5, "muted": False, "state": "playing"},
        "state": {"on": True, "brightness": 1.0, "effect": "None", "rgb": [255, 255, 255]},
    })
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    payload = music_api.apply_transport("stop")
    assert payload["message"] == "The speaker stopped."
    call = next(item for item in runtime.calls if item["arguments"].get("action") == "stop_speaking")
    assert call["arguments"] == {"action": "stop_speaking"}


def test_apply_transport_reports_a_missing_speaker_rather_than_guessing(music_env, monkeypatch):
    install(monkeypatch, music_env, [PROVIDER_MANIFEST], FakeRuntime())
    with pytest.raises(music_api.MusicApiError) as missing:
        music_api.apply_transport("pause")
    assert missing.value.code == "player_action_unavailable"
    assert "pause" in str(missing.value)


def test_control_rejects_an_out_of_range_volume(music_env, monkeypatch):
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], FakeRuntime())
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/control", {"action": "volume", "value": 4})
    finally:
        server.shutdown()
    assert status == 400 and payload["ok"] is False and payload["code"] == "validation_error"


def test_music_index_without_a_provider_explains_how_to_get_one(music_env, monkeypatch):
    install(monkeypatch, music_env, [player_manifest()], FakeRuntime())
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music")
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is False
    assert payload["code"] == "no_music_provider"
    assert payload["page"] is None


def test_search_and_library_forward_to_the_provider(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": {"ok": True, "action": "search", "message": "found", "tracks": [
        {"video_id": "abc12345678", "title": "Song", "artist": "Artist", "duration_sec": 200, "artwork_url": ""}
    ], "playlists": [], "count": 1, "lyrics": [], "lyrics_plain": "", "lyrics_timing": "", "lyrics_source": ""}})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST], runtime)
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/search?q=Miles%20Davis&limit=3")
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is True
    call = next(item for item in runtime.calls if item["arguments"].get("action") == "search")
    assert call["arguments"] == {"action": "search", "query": "Miles Davis", "limit": 3}


# ── the console's catalog cache ──────────────────────────────────────────
#
# Catalog reads are answered from the console's own copy (music_cache) and
# refreshed behind the reader, so a page, the assistant or the mobile app never
# waits on the skill worker for a playlist it has already seen.


def library_result(**overrides):
    result = {
        "ok": True, "action": "library", "message": "", "video_id": "", "title": "", "artist": "",
        "duration_sec": 0, "artwork_url": "", "mime_type": "", "play_url": "", "tracks": [],
        "playlists": [{"playlist_id": "LM", "name": "Liked Music", "track_count": 3}],
        "queue": [], "queue_index": -1, "shuffle": False, "repeat": "off", "count": 1,
        "lyrics": [], "lyrics_plain": "", "lyrics_timing": "", "lyrics_source": "",
    }
    result.update(overrides)
    return result


def provider_calls(runtime, action):
    return [item for item in runtime.calls if item["arguments"].get("action") == action]


def test_library_reads_are_served_from_the_console_cache(music_env, monkeypatch):
    """The second read of one library view comes off disk, not the worker.

    The payload is identical either way; only the cache headers say which copy
    answered, so a reader that ignores them sees exactly what it saw before.
    """
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: library_result()})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST], runtime)
    server = serve()
    try:
        first_status, first, first_headers = read_json(server, "/api/music/library?kind=playlists")
        second_status, second, second_headers = read_json(server, "/api/music/library?kind=playlists")
        other_status, other, other_headers = read_json(server, "/api/music/library?kind=likes")
    finally:
        server.shutdown()
    assert first_status == second_status == other_status == 200
    assert first == second
    assert first_headers["X-Music-Cache"] == "miss"
    assert second_headers["X-Music-Cache"] == "hit"
    assert int(second_headers["X-Music-Cache-Age"]) >= 0
    assert other_headers["X-Music-Cache"] == "miss"
    # One provider read per distinct view, not one per request.
    assert len(provider_calls(runtime, "library")) == 2


def test_refresh_bypasses_the_cached_read(music_env, monkeypatch):
    reads = {"count": 0}

    def answer(_matched):
        reads["count"] += 1
        return library_result(count=reads["count"])

    runtime = FakeRuntime(results={"opendeck_music": answer})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        read_json(server, "/api/music/library?kind=playlists")
        _hit_status, cached, cached_headers = read_json(server, "/api/music/library?kind=playlists")
        _status, fresh, fresh_headers = read_json(server, "/api/music/library?kind=playlists&refresh=1")
    finally:
        server.shutdown()
    assert cached_headers["X-Music-Cache"] == "hit" and cached["count"] == 1
    assert fresh_headers["X-Music-Cache"] == "refresh" and fresh["count"] == 2
    assert reads["count"] == 2


def test_search_results_are_cached_per_query(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": lambda matched: library_result(
        action="search", query=matched["arguments"].get("query"))})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST], runtime)
    server = serve()
    try:
        read_json(server, "/api/music/search?q=Miles")
        _status, cached, cached_headers = read_json(server, "/api/music/search?q=Miles")
        read_json(server, "/api/music/search?q=Coltrane")
    finally:
        server.shutdown()
    assert cached_headers["X-Music-Cache"] == "hit" and cached["query"] == "Miles"
    assert len(provider_calls(runtime, "search")) == 2


def test_a_save_drops_the_cached_library_it_changed(music_env, monkeypatch):
    """Liking a track changes the library the page holds, so it is re-read."""
    def answer(matched):
        if matched["arguments"]["action"] == "save":
            return save_result(count=4)
        return library_result()

    runtime = FakeRuntime(results={"opendeck_music": answer})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    write_state(track={"video_id": "a1", "title": "First", "artist": "Artist"})
    server = serve()
    try:
        read_json(server, "/api/music/library?kind=likes")
        _hit_status, _cached, cached_headers = read_json(server, "/api/music/library?kind=likes")
        read_json(server, "/api/music/save", {"kind": "like"})
        _status, _payload, after_write = read_json(server, "/api/music/library?kind=likes")
    finally:
        server.shutdown()
    assert cached_headers["X-Music-Cache"] == "hit"
    assert after_write["X-Music-Cache"] == "miss"
    assert len(provider_calls(runtime, "library")) == 2


def test_music_index_reports_the_cache_and_the_artwork_route(music_env, monkeypatch):
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], FakeRuntime())
    server = serve()
    try:
        _status, payload, _headers = read_json(server, "/api/music")
    finally:
        server.shutdown()
    assert payload["cache"]["enabled"] is True
    assert payload["cache"]["ttl_sec"] == music_cache.DEFAULT_CATALOG_TTL
    assert "GET /api/music/artwork?src=" in payload["endpoints"]


def test_lyrics_uses_the_recorded_track(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": {"ok": True, "action": "lyrics", "message": "ok",
        "video_id": "abc12345678", "title": "Song", "artist": "Artist",
        "lyrics": [{"time_sec": 1.5, "text": "hello"}], "lyrics_plain": "hello",
        "lyrics_timing": "synced", "lyrics_source": "LRCLIB", "tracks": [], "playlists": [], "count": 1}})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST], runtime)
    Path(console.DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(music_api._state_path()).write_text(json.dumps({
        "track": {"video_id": "abc12345678", "title": "Song", "artist": "Artist", "duration_sec": 200},
        "started_at": 1_000.0, "duration_sec": 200,
    }), encoding="utf-8")
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/lyrics")
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is True
    assert payload["lyrics"] == [{"time_sec": 1.5, "text": "hello"}]
    call = next(item for item in runtime.calls if item["arguments"].get("action") == "lyrics")
    assert call["arguments"] == {"action": "lyrics", "video_id": "abc12345678", "title": "Song", "artist": "Artist", "duration_sec": 200}


def test_lyrics_for_a_named_track_are_answered_for_that_track(music_env, monkeypatch):
    """Naming a video_id must never be answered with the recorded track's words.

    The provider matches on title/artist (LRCLIB has no video ids), so merging
    whatever the console last played over a named id made it look up the previous
    song and return *its* lyrics as the words of the named one - a wrong answer
    reported as a success, which is exactly what the music page showed.
    """
    runtime = FakeRuntime(results={"opendeck_music": {"ok": True, "action": "lyrics", "message": "ok",
        "video_id": "b2", "title": "Other Song", "artist": "Other Artist",
        "lyrics": [], "lyrics_plain": "", "lyrics_timing": "none", "lyrics_source": "LRCLIB",
        "tracks": [], "playlists": [], "count": 0}})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST], runtime)
    write_state(track={"video_id": "a1", "title": "Playing Now", "artist": "Playing Artist",
                       "duration_sec": 200})
    server = serve()
    try:
        status, _payload, _headers = read_json(server, "/api/music/lyrics?video_id=b2")
    finally:
        server.shutdown()
    assert status == 200
    call = next(item for item in runtime.calls if item["arguments"].get("action") == "lyrics")
    assert call["arguments"] == {"action": "lyrics", "video_id": "b2"}, (
        "the recorded track's title, artist and duration must not be attached to a named one"
    )


def test_lyrics_named_by_the_playing_title_keep_the_recorded_duration(music_env, monkeypatch):
    """The playing track named by title is the playing track: its clock comes too."""
    runtime = FakeRuntime(results={"opendeck_music": {"ok": True, "action": "lyrics", "message": "ok",
        "video_id": "a1", "title": "Playing Now", "artist": "Playing Artist",
        "lyrics": [], "lyrics_plain": "", "lyrics_timing": "none", "lyrics_source": "LRCLIB",
        "tracks": [], "playlists": [], "count": 0}})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST], runtime)
    write_state(track={"video_id": "a1", "title": "Playing Now!", "artist": "Playing Artist",
                       "album": "An Album", "duration_sec": 200})
    server = serve()
    try:
        read_json(server, "/api/music/lyrics?title=playing%20now")
    finally:
        server.shutdown()
    call = next(item for item in runtime.calls if item["arguments"].get("action") == "lyrics")
    assert call["arguments"] == {"action": "lyrics", "title": "playing now", "video_id": "a1",
                                 "artist": "Playing Artist", "album": "An Album", "duration_sec": 200}


def test_lyrics_for_another_title_inherit_nothing_from_the_recorded_track(music_env, monkeypatch):
    """A title that is not what is playing is a different track, not a merge."""
    runtime = FakeRuntime(results={"opendeck_music": {"ok": True, "action": "lyrics", "message": "ok",
        "video_id": "", "title": "Yesterday", "artist": "",
        "lyrics": [], "lyrics_plain": "", "lyrics_timing": "none", "lyrics_source": "LRCLIB",
        "tracks": [], "playlists": [], "count": 0}})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST], runtime)
    write_state(track={"video_id": "a1", "title": "Playing Now", "artist": "Playing Artist",
                       "duration_sec": 200})
    server = serve()
    try:
        read_json(server, "/api/music/lyrics?title=Yesterday")
    finally:
        server.shutdown()
    call = next(item for item in runtime.calls if item["arguments"].get("action") == "lyrics")
    assert call["arguments"] == {"action": "lyrics", "title": "Yesterday"}


# ── queue ────────────────────────────────────────────────────────────────


def queue_track(video_id, title="Song"):
    return {"video_id": video_id, "title": title, "artist": "Artist", "duration_sec": 200, "artwork_url": ""}


def queue_result(items, index=-1, message="", play_url=""):
    """A provider queue payload; play_url is set only for a resolved track."""
    result = {
        "ok": True, "action": "queue", "message": message, "video_id": "", "title": "",
        "artist": "", "duration_sec": 0, "artwork_url": "", "mime_type": "", "play_url": play_url,
        "tracks": [], "playlists": [], "queue": items, "queue_index": index, "count": len(items),
        "lyrics": [], "lyrics_plain": "", "lyrics_timing": "", "lyrics_source": "",
    }
    if play_url:
        item = items[index]
        result.update({key: item[key] for key in ("video_id", "title", "artist", "duration_sec", "artwork_url")})
        result["mime_type"] = "audio/mp4"
    return result


def queue_calls(runtime):
    return [item["arguments"] for item in runtime.calls if item["arguments"].get("action") == "queue"]


def write_state(**fields):
    Path(console.DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(music_api._state_path()).write_text(json.dumps(fields), encoding="utf-8")


def test_queue_get_reports_what_the_provider_holds(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: queue_result(
        [queue_track("a1", "First"), queue_track("b2", "Second")], index=0, message="2 track(s) queued.")})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/queue")
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is True
    assert [item["title"] for item in payload["queue"]] == ["First", "Second"]
    assert payload["index"] == 0 and payload["count"] == 2
    assert "2 track(s)" in payload["message"]
    # A bodyless read sends the declaration's own default op and nothing else.
    assert queue_calls(runtime) == [{"action": "queue", "op": "list"}]


def test_queue_add_forwards_the_track_and_leaves_the_speaker_alone(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: queue_result(
        [queue_track("abc12345678", "Song")], index=-1, message="1 track(s) queued.")})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/queue", {"action": "add", "video_id": "abc12345678"})
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is True and payload["count"] == 1
    assert queue_calls(runtime) == [{"action": "queue", "op": "add", "video_id": "abc12345678"}]
    # Queueing is not playing: nothing may be handed to the speaker.
    assert not [item for item in runtime.calls if item["arguments"].get("action") == "play_url"]
    assert music_api._read_music_state() == {}


def test_queue_add_range_forwards_the_playlist_and_the_song_it_started_at(music_env, monkeypatch):
    """The page's "this song and every song under it" button: the console names
    the playlist the row was listed in and the song that was tapped, and the
    provider reads the tail of that playlist."""
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: queue_result(
        [queue_track("b2", "Second"), queue_track("c3", "Third")], index=-1,
        message="Queued 2 song(s) from “Second” to the end of the playlist.")})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/queue", {
            "action": "add_range", "playlist_id": "PL-fake", "video_id": "b2"})
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is True and payload["count"] == 2
    assert queue_calls(runtime) == [
        {"action": "queue", "op": "add_range", "playlist_id": "PL-fake", "video_id": "b2"}
    ]
    # Adding is not playing: nothing may be handed to the speaker.
    assert not [item for item in runtime.calls if item["arguments"].get("action") == "play_url"]


def test_queue_add_range_without_a_playlist_is_refused(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: queue_result([])})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/queue", {
            "action": "add_range", "video_id": "b2"})
    finally:
        server.shutdown()
    assert status == 400 and payload["ok"] is False
    assert queue_calls(runtime) == [], "nothing reaches the provider without a playlist"


def test_queue_add_accepts_a_search_query(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: queue_result([], message="Queued 1 track.")})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/queue", {"action": "add", "query": "Miles Davis"})
    finally:
        server.shutdown()
    assert status == 200
    assert queue_calls(runtime) == [{"action": "queue", "op": "add", "query": "Miles Davis"}]
    assert payload["message"] == "Queued 1 track."


def test_queue_next_hands_the_queued_track_to_the_speaker(music_env, monkeypatch):
    items = [queue_track("a1", "First"), queue_track("b2", "Second")]
    runtime = FakeRuntime(
        results={"opendeck_music": lambda _matched: queue_result(
            items, index=1, message="Playing Second.", play_url="http://192.168.0.152:4100/api/media/stream?url=z")},
        player={"ok": True, "action": "play_url", "message": "playing", "device_name": "Light",
                "speaker": {"volume": 0.5, "muted": False, "state": "playing"},
                "state": {"on": True, "brightness": 1.0, "effect": "None", "rgb": [255, 255, 255]}},
    )
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/queue", {"action": "next"})
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is True
    assert payload["index"] == 1 and payload["count"] == 2
    assert payload["track"]["title"] == "Second"
    assert payload["playing"] is True
    assert queue_calls(runtime) == [{"action": "queue", "op": "next"}]
    handoff = next(item for item in runtime.calls if item["arguments"].get("action") == "play_url")
    assert handoff["arguments"]["url"].endswith("url=z")
    assert handoff["arguments"]["title"] == "Second"
    # Queue-sourced playback is marked so the console may advance it later.
    saved = music_api._read_music_state()
    assert saved["queue_active"] is True and saved["queue_index"] == 1


def test_queue_next_on_an_exhausted_queue_reports_it_without_playing(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: queue_result(
        [queue_track("a1", "First")], index=0, message="The queue has no next track.")})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/queue", {"action": "next"})
    finally:
        server.shutdown()
    assert status == 409 and payload["ok"] is False
    assert payload["code"] == "queue_empty"
    assert payload["error"] == "The queue has no next track."
    assert not [item for item in runtime.calls if item["arguments"].get("action") == "play_url"]
    assert music_api._read_music_state() == {}


def test_queue_rejects_a_bad_action_and_a_missing_target(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: queue_result([], message="")})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        unknown_bad, unknown, _h = read_json(server, "/api/music/queue", {"action": "rewind"})
        target_bad, target, _h = read_json(server, "/api/music/queue", {"action": "add"})
        index_bad, index, _h = read_json(server, "/api/music/queue", {"action": "remove"})
    finally:
        server.shutdown()
    assert unknown_bad == 400 and unknown["code"] == "validation_error"
    assert target_bad == 400 and target["code"] == "validation_error"
    assert index_bad == 400 and index["code"] == "validation_error"
    # A refused request never reaches the provider.
    assert queue_calls(runtime) == []


def test_queue_remove_forwards_the_position(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: queue_result(
        [queue_track("b2", "Second")], index=-1, message="1 track(s) queued.")})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/queue", {"action": "remove", "index": 1})
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is True and payload["count"] == 1
    assert queue_calls(runtime) == [{"action": "queue", "op": "remove", "index": 1}]


def test_queue_move_reorders_a_dragged_row(music_env, monkeypatch):
    # Dragging a row sends one reorder: from -> to, in the provider's own op.
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: queue_result(
        [queue_track("b2", "Second"), queue_track("a1", "First")], index=-1,
        message="Moved “Second” to position 1.")})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/queue", {"action": "move", "index": 1, "to": 0})
        missing_bad, missing, _h = read_json(server, "/api/music/queue", {"action": "move", "index": 1})
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is True
    assert [item["title"] for item in payload["queue"]] == ["Second", "First"]
    assert queue_calls(runtime) == [{"action": "queue", "op": "move", "index": 1, "to": 0}]
    assert missing_bad == 400 and missing["code"] == "validation_error"
    # A reorder is not a play: the speaker is never handed anything.
    assert not [item for item in runtime.calls if item["arguments"].get("action") == "play_url"]


def test_queue_play_next_moves_a_row_up_without_playing_it(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: queue_result(
        [queue_track("a1", "First"), queue_track("b2", "Second")], index=0,
        message="Moved “Second” to position 2 (next up).")})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/queue", {"action": "next_up", "index": 1})
        front_status, _front, _h = read_json(server, "/api/music/queue", {"action": "front", "query": "Second"})
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is True
    assert queue_calls(runtime) == [
        {"action": "queue", "op": "next_up", "index": 1},
        {"action": "queue", "op": "front", "query": "Second"},
    ]
    assert front_status == 200
    # Prioritising a track is not playback.
    assert not [item for item in runtime.calls if item["arguments"].get("action") == "play_url"]


def test_queue_shuffle_and_repeat_come_back_with_the_queue(music_env, monkeypatch):
    def shuffed(_matched):
        return {**queue_result([queue_track("a1", "First")], index=0, message="Shuffling the queue."),
                "shuffle": True, "repeat": "all"}
    runtime = FakeRuntime(results={"opendeck_music": shuffed})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        shuffle_status, shuffled, _h = read_json(server, "/api/music/queue", {"action": "shuffle"})
        toggle_status, toggled, _h = read_json(server, "/api/music/queue", {"action": "shuffle", "on": False})
        repeat_status, repeated, _h = read_json(server, "/api/music/queue", {"action": "repeat", "mode": "all"})
        cycle_status, cycled, _h = read_json(server, "/api/music/queue", {"action": "repeat"})
    finally:
        server.shutdown()
    assert (shuffle_status, toggle_status, repeat_status, cycle_status) == (200, 200, 200, 200)
    # The page paints its two toggles from the reply, so both travel with it.
    assert shuffled["shuffle"] is True and shuffled["repeat"] == "all"
    assert toggled["shuffle"] is True
    assert repeated["count"] == 1 and repeated["repeat"] == "all"
    # No `on`/`mode` is a toggle/cycle, so nothing else is sent.
    assert queue_calls(runtime) == [
        {"action": "queue", "op": "shuffle"},
        {"action": "queue", "op": "shuffle", "on": False},
        {"action": "queue", "op": "repeat", "mode": "all"},
        {"action": "queue", "op": "repeat"},
    ]


def test_queue_shuffle_and_repeat_refuse_a_bad_value(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: queue_result([], message="")})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        on_bad, on, _h = read_json(server, "/api/music/queue", {"action": "shuffle", "on": "yes"})
        mode_bad, mode, _h = read_json(server, "/api/music/queue", {"action": "repeat", "mode": "sometimes"})
    finally:
        server.shutdown()
    assert on_bad == 400 and on["code"] == "validation_error"
    assert mode_bad == 400 and mode["code"] == "validation_error"
    assert queue_calls(runtime) == []


def save_result(**overrides):
    result = {"ok": True, "action": "save", "message": "Saved.", "count": 1}
    result.update(overrides)
    return result


def save_calls(runtime):
    return [item["arguments"] for item in runtime.calls if item["arguments"].get("action") == "save"]


def test_save_likes_the_playing_track(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: save_result(count=4)})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    write_state(track={"video_id": "a1", "title": "First", "artist": "Artist"})
    server = serve()
    try:
        like_status, liked, _h = read_json(server, "/api/music/save", {"kind": "like"})
        unlike_status, unliked, _h = read_json(server, "/api/music/save", {"kind": "unlike"})
    finally:
        server.shutdown()
    assert like_status == 200 and liked["liked"] is True and liked["count"] == 4
    assert unlike_status == 200 and unliked["liked"] is False
    # The page sends only the kind: the console fills in the track so a like
    # never needs a video id the page is not showing.
    assert save_calls(runtime) == [
        {"action": "save", "kind": "like", "video_id": "a1", "title": "First", "artist": "Artist"},
        {"action": "save", "kind": "unlike", "video_id": "a1", "title": "First", "artist": "Artist"},
    ]


def test_save_adds_a_track_to_a_named_playlist(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: save_result(count=9)})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    write_state(track={"video_id": "a1", "title": "First", "artist": "Artist"})
    server = serve()
    try:
        status, payload, _h = read_json(server, "/api/music/save", {"kind": "playlist_add", "playlist_id": "PL-1"})
    finally:
        server.shutdown()
    assert status == 200 and payload["playlist_id"] == "PL-1" and payload["count"] == 9
    assert save_calls(runtime) == [{
        "action": "save", "kind": "playlist_add", "playlist_id": "PL-1",
        "video_id": "a1", "title": "First", "artist": "Artist",
    }]


def test_save_creates_a_playlist_and_reports_its_id(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: save_result(
        playlists=[{"playlist_id": "PL-9", "name": "Road trip", "track_count": 0}])})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        status, payload, _h = read_json(server, "/api/music/save", {"kind": "playlist_create", "name": "Road trip"})
    finally:
        server.shutdown()
    assert status == 200 and payload["playlist_id"] == "PL-9" and payload["name"] == "Road trip"
    # A new playlist needs no track, so it works with nothing playing.
    assert save_calls(runtime) == [{"action": "save", "kind": "playlist_create", "name": "Road trip"}]


def test_save_reports_an_unknown_kind_and_nothing_playing(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: save_result()})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        kind_bad, kind, _h = read_json(server, "/api/music/save", {"kind": "delete_everything"})
        none_bad, none, _h = read_json(server, "/api/music/save", {"kind": "like"})
        name_bad, name, _h = read_json(server, "/api/music/save", {"kind": "playlist_create"})
    finally:
        server.shutdown()
    assert kind_bad == 400 and kind["code"] == "validation_error"
    assert none_bad == 409 and none["code"] == "no_track"
    assert name_bad == 400 and name["code"] == "validation_error"
    assert save_calls(runtime) == []


def finished_queue_state(**overrides):
    """A queue-sourced track that has run to its end (speaker idle)."""
    state = {
        "track": {"video_id": "a1", "title": "First", "artist": "Artist", "duration_sec": 200},
        "started_at": time.time() - 200, "paused_at": None, "paused_seconds": 0.0,
        "duration_sec": 200, "updated_at": time.time(), "source": "opendeck",
        "queue_active": True, "queue_index": 0,
        "handoff": {"supported": True},
    }
    state.update(overrides)
    return state


def idle_player():
    return {"ok": True, "action": "state", "message": "reported", "device_name": "Light",
            "speaker": {"volume": 0.5, "muted": False, "state": "idle"},
            "state": {"on": True, "brightness": 1.0, "effect": "None", "rgb": [255, 255, 255]}}


def test_now_playing_auto_advances_a_finished_queued_track(music_env, monkeypatch):
    items = [queue_track("a1", "First"), queue_track("b2", "Second")]
    runtime = FakeRuntime(
        results={"opendeck_music": lambda _matched: queue_result(
            items, index=1, message="Playing Second.", play_url="http://192.168.0.152:4100/api/media/stream?url=z")},
        player=idle_player(),
    )
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    write_state(**finished_queue_state())
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/now-playing")
    finally:
        server.shutdown()
    assert status == 200
    assert payload["track"]["title"] == "Second"
    # The hand-over is reported as such, so a reader watching the end of a song
    # (the wall display) knows the queue follows this track.
    assert payload["queue_active"] is True
    # ``auto`` marks a track that ended by itself, which is what repeat-one keys
    # off; a skip from the page never sends it.
    assert queue_calls(runtime) == [{"action": "queue", "op": "next", "auto": True}]
    handoff = next(item for item in runtime.calls if item["arguments"].get("action") == "play_url")
    assert handoff["arguments"]["url"].endswith("url=z")
    saved = music_api._read_music_state()
    assert saved["queue_active"] is True and saved["queue_index"] == 1


def test_now_playing_leaves_a_one_off_play_alone(music_env, monkeypatch):
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: queue_result([], message="")}, player=idle_player())
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    write_state(**finished_queue_state(queue_active=False, queue_index=None))
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/now-playing")
    finally:
        server.shutdown()
    assert status == 200
    assert payload["track"]["title"] == "First"
    assert payload["queue_active"] is False, "a song played on its own is followed by nothing"
    assert queue_calls(runtime) == []


def test_now_playing_does_not_advance_a_paused_queue(music_env, monkeypatch):
    runtime = FakeRuntime(
        results={"opendeck_music": lambda _matched: queue_result(
            [queue_track("a1", "First"), queue_track("b2", "Second")], index=1, message="Playing Second.",
            play_url="http://192.168.0.152:4100/api/media/stream?url=z")},
        player={"ok": True, "action": "state", "message": "reported", "device_name": "Light",
                "speaker": {"volume": 0.5, "muted": False, "state": "paused"},
                "state": {"on": True, "brightness": 1.0, "effect": "None", "rgb": [255, 255, 255]}},
    )
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    write_state(**finished_queue_state(paused_at=time.time() - 100))
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/now-playing")
    finally:
        server.shutdown()
    assert status == 200
    assert payload["track"]["title"] == "First"
    assert payload["position"]["paused"] is True
    assert queue_calls(runtime) == []


# ── covers served from the console ───────────────────────────────────────

COVER = b"\x89PNG\r\n\x1a\ncover bytes"


def cover_url(original="https://cdn.example/cover.png"):
    return "/api/music/artwork?src=" + urllib.parse.quote(original, safe="")


def test_an_idle_speaker_holding_a_recorded_pause_is_not_advanced(music_env, monkeypatch):
    """A paused speaker reports itself idle — that is not a finished track.

    This box's speaker answers "idle" for a paused stream, so the finished-track
    check matched and the console started the next queued track (and dropped the
    pause) whenever the wall clock passed the track's length while it was paused.
    """
    items = [queue_track("a1", "First"), queue_track("b2", "Second")]
    runtime = FakeRuntime(
        results={"opendeck_music": lambda _matched: queue_result(
            items, index=1, message="Playing Second.",
            play_url="http://192.168.0.152:4100/api/media/stream?url=z")},
        player=idle_player(),
    )
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    write_state(**finished_queue_state(paused_at=time.time() - 400))
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/now-playing")
    finally:
        server.shutdown()
    assert status == 200
    assert payload["track"]["title"] == "First", "the paused track is still the one playing"
    assert queue_calls(runtime) == []


def test_a_resume_made_outside_the_console_closes_the_pause_record(music_env, monkeypatch):
    """The speaker says it is playing, so the console's pause record is over.

    A track resumed by voice or on the device itself was still read as paused:
    the progress bar stood still and, in a queue, the end-of-track check had a
    pause in hand for music that was playing.
    """
    playing = {"ok": True, "action": "state", "message": "reported", "device_name": "Light",
               "speaker": {"volume": 0.5, "muted": False, "state": "playing"},
               "state": {"on": True, "brightness": 1.0, "effect": "None", "rgb": [255, 255, 255]}}
    runtime = FakeRuntime(results={"opendeck_music": {"ok": True, "action": "now_playing",
        "tracks": [], "playlists": [], "queue": [], "queue_index": -1, "count": 0,
        "lyrics": [], "lyrics_plain": "", "lyrics_timing": "", "lyrics_source": "",
        "video_id": "", "title": "", "artist": "", "duration_sec": 0, "artwork_url": "",
        "mime_type": "", "play_url": ""}}, player=playing)
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    write_state(**finished_queue_state(paused_at=time.time() - 30, paused_seconds=10.0))
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/now-playing")
    finally:
        server.shutdown()
    assert status == 200
    assert payload["position"]["paused"] is False
    saved = music_api._read_music_state()
    assert saved["paused_at"] is None, "the speaker's own resume closes the pause record"
    assert saved["paused_seconds"] >= 39.0, "the time it spent paused is added once, not lost"


def test_artwork_route_serves_a_cover_from_the_cache(music_env, monkeypatch):
    """The first paint fetches a cover; every later paint comes off disk.

    The reply also carries the long-lived cache headers, so the browser keeps
    the image and a re-request is answered with 304 instead of the bytes.
    """
    install(monkeypatch, music_env, [PROVIDER_MANIFEST], FakeRuntime())
    downloads = []

    def fake_download(url, timeout=None):
        downloads.append(url)
        return COVER, "image/png"

    monkeypatch.setattr(music_cache, "download_artwork", fake_download)
    server = serve()
    try:
        first_status, first_body, first_headers = read_bytes(server, cover_url())
        second_status, second_body, second_headers = read_bytes(server, cover_url())
        not_modified_status, not_modified_body, _headers = read_bytes(
            server, cover_url(), {"If-None-Match": first_headers["ETag"]}
        )
    finally:
        server.shutdown()
    assert first_status == second_status == 200
    assert first_body == second_body == COVER
    assert first_headers["Content-Type"] == "image/png"
    assert "max-age=" in first_headers["Cache-Control"] and "immutable" in first_headers["Cache-Control"]
    assert first_headers["X-Music-Cache"] == "miss"
    assert second_headers["X-Music-Cache"] == "hit"
    assert downloads == ["https://cdn.example/cover.png"]
    assert not_modified_status == 304 and not_modified_body == b""


def test_artwork_route_refuses_a_missing_or_unsupported_source(music_env, monkeypatch):
    install(monkeypatch, music_env, [PROVIDER_MANIFEST], FakeRuntime())
    server = serve()
    try:
        missing_status, missing, _headers = read_json(server, "/api/music/artwork")
        file_status, local_file, _headers = read_json(server, cover_url("file:///etc/passwd"))
    finally:
        server.shutdown()
    assert missing_status == 400 and missing["code"] == "validation_error"
    assert file_status == 400 and local_file["ok"] is False


def test_a_cover_that_cannot_be_fetched_is_reported(music_env, monkeypatch):
    install(monkeypatch, music_env, [PROVIDER_MANIFEST], FakeRuntime())

    def refusing(_url, timeout=None):
        raise music_cache.ArtworkError("The artwork could not be fetched: offline")

    monkeypatch.setattr(music_cache, "download_artwork", refusing)
    server = serve()
    try:
        status, payload, _headers = read_json(server, cover_url())
    finally:
        server.shutdown()
    assert status == 502
    assert payload["code"] == "artwork_unavailable"


# ── the independent queue service ─────────────────────────────────────────
#
# The queue itself lives in its own service (nix_core/music_queue_service.py)
# when one is running: the console proxies to it, a whole playlist changes hands
# in a few writes rather than one per song, and a read carries a revision so a
# page that holds the current one is told "unchanged" instead of being sent the
# queue again.


@pytest.fixture()
def queue_service(tmp_path, monkeypatch):
    """The queue service, in-process, on its own port, with the console pointed at it."""
    started = []

    def build(provider_limits=None, **limits):
        queue = music_queue.MusicQueue(str(tmp_path / "queue.db"), music_queue.QueueLimits(**limits))
        service = music_queue_service.QueueService(queue)
        httpd = music_queue_service.ThreadingHTTPServer(
            ("127.0.0.1", 0), music_queue_service._handler_for(service))
        httpd.queue_service = service
        # The address it would publish, for the checks that make the console
        # find the service rather than be told where it is.
        service.address = f"http://127.0.0.1:{httpd.server_address[1]}"
        Thread(target=httpd.serve_forever, daemon=True).start()
        started.append(httpd)
        monkeypatch.setattr(music_api, "QUEUE_SERVICE_URL", f"http://127.0.0.1:{httpd.server_address[1]}")
        monkeypatch.setattr(music_api, "_queue_limits", {"at": 0.0, "skill_id": "", "values": None})
        return service

    yield build
    for httpd in started:
        httpd.shutdown()


def playlist_tracks(count, prefix="Song"):
    return [
        {"video_id": f"v{index:04d}", "title": f"{prefix} {index}", "artist": "Artist",
         "duration_sec": 180, "artwork_url": ""}
        for index in range(count)
    ]


def provider_with_limits(**limits):
    manifest = json.loads(json.dumps(PROVIDER_MANIFEST))
    manifest["device_ui"]["media"]["provider"]["queue"]["limits"] = limits
    return manifest


def test_a_whole_playlist_is_queued_in_one_request(music_env, monkeypatch, queue_service):
    """The point of the service: four hundred songs, one write, no provider call."""
    runtime = FakeRuntime(results={"opendeck_music": {}})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    service = queue_service()
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/queue", {
            "action": "add_many", "tracks": playlist_tracks(400), "playlist_id": "PL-liked"})
        read_status, read, _h = read_json(server, "/api/music/queue?from=0&limit=50")
        tail_status, tail, _h = read_json(server, "/api/music/queue?from=350&limit=50")
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is True
    assert payload["accepted"] == 400 and payload["total"] == 400
    assert "400 tracks queued" in payload["message"]
    # The whole line-up is readable in windows, and the count is the whole queue.
    assert read_status == 200 and read["total"] == 400 and read["count"] == 50
    assert tail_status == 200 and tail["queue"][0]["title"] == "Song 350"
    # Not one provider call: the page handed over the metadata it already listed.
    assert queue_calls(runtime) == []
    assert service.queue.total() == 400


def test_a_reader_that_holds_the_revision_is_told_nothing_changed(music_env, monkeypatch, queue_service):
    runtime = FakeRuntime(results={"opendeck_music": {}})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    queue_service()
    server = serve()
    try:
        read_json(server, "/api/music/queue", {"action": "add_many", "tracks": playlist_tracks(30)})
        _status, first, _h = read_json(server, "/api/music/queue?limit=10")
        status, again, _h = read_json(server, f"/api/music/queue?limit=10&revision={first['revision']}")
    finally:
        server.shutdown()
    assert status == 200 and again["unchanged"] is True and again["queue"] == []
    assert again["total"] == 30
    # A change moves the revision, so the same reader is answered in full again.
    server = serve()
    try:
        read_json(server, "/api/music/queue", {"action": "add", "video_id": "v9999"})
        _status, changed, _h = read_json(server, f"/api/music/queue?limit=10&revision={first['revision']}")
    finally:
        server.shutdown()
    assert changed["unchanged"] is False and changed["total"] == 31


def test_the_service_is_told_the_limits_the_provider_declares(music_env, monkeypatch, queue_service):
    runtime = FakeRuntime(results={"opendeck_music": {}})
    install(monkeypatch, music_env,
            [provider_with_limits(max_tracks=900, max_batch=50, window=25, prefetch=4,
                                  resolves_per_minute=12), player_manifest()], runtime)
    service = queue_service()
    server = serve()
    try:
        read_json(server, "/api/music/queue")
    finally:
        server.shutdown()
    assert service.queue.limits.max_tracks == 900
    assert service.queue.limits.window == 25
    assert service.queue.limits.prefetch == 4
    assert service.queue.bucket.rate == pytest.approx(12 / 60)


def test_a_long_playlist_is_written_in_the_batches_the_provider_allows(music_env, monkeypatch, queue_service):
    runtime = FakeRuntime(results={"opendeck_music": {}})
    install(monkeypatch, music_env,
            [provider_with_limits(max_batch=2, max_tracks=5000), player_manifest()], runtime)
    service = queue_service()
    server = serve()
    try:
        _status, payload, _h = read_json(server, "/api/music/queue", {
            "action": "add_many", "tracks": playlist_tracks(5)})
    finally:
        server.shutdown()
    assert payload["accepted"] == 5 and payload["total"] == 5
    # Five songs, a batch ceiling of two: three writes, not five.
    assert service.queue.stats()["adds"] == 3


def test_a_playlist_beyond_the_ceiling_says_what_did_not_fit(music_env, monkeypatch, queue_service):
    runtime = FakeRuntime(results={"opendeck_music": {}})
    install(monkeypatch, music_env,
            [provider_with_limits(max_tracks=3, max_batch=10), player_manifest()], runtime)
    queue_service()
    server = serve()
    try:
        _status, payload, _h = read_json(server, "/api/music/queue", {
            "action": "add_many", "tracks": playlist_tracks(5)})
    finally:
        server.shutdown()
    assert payload["accepted"] == 3 and payload["refused"] == 2
    assert payload["total"] == 3


def test_the_service_plays_the_track_it_hands_over(music_env, monkeypatch, queue_service):
    runtime = FakeRuntime(
        results={"opendeck_music": lambda matched: (
            queue_result([queue_track("v0001", "Song 1")], index=0, message="Second",
                         play_url="http://192.168.0.152:4100/api/media/stream?url=service")
            if matched["arguments"].get("action") == "play" else {"ok": True, "action": "queue"})},
        player={"ok": True, "action": "play_url", "message": "playing", "device_name": "Light",
                "speaker": {"volume": 0.5, "muted": False, "state": "playing"},
                "state": {"on": True, "brightness": 1.0, "effect": "None", "rgb": [255, 255, 255]}},
    )
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    service = queue_service()
    server = serve()
    try:
        read_json(server, "/api/music/queue", {"action": "add_many", "tracks": playlist_tracks(3)})
        status, payload, _h = read_json(server, "/api/music/queue", {"action": "next"})
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is True
    # The service moved its own cursor and the provider resolved the stream.
    assert service.queue.state().index == 0
    assert payload["track"]["title"] == "Song 1"
    plays = [item["arguments"] for item in runtime.calls if item["arguments"].get("action") == "play"]
    assert plays == [{"action": "play", "video_id": "v0000"}]
    handoff = next(item for item in runtime.calls if item["arguments"].get("action") == "play_url")
    assert handoff["arguments"]["url"].endswith("url=service")
    saved = music_api._read_music_state()
    assert saved["queue_active"] is True and saved["queue_index"] == 0


def test_a_playlist_song_can_be_queued_by_name_and_from_a_row(music_env, monkeypatch, queue_service):
    """The page's own buttons through the service: a search hit, and \"this song
    and every song under it\", which reads the playlist's tail once."""
    runtime = FakeRuntime(results={"opendeck_music": lambda matched: {
        "ok": True, "action": matched["arguments"]["action"], "message": "ok",
        "tracks": [queue_track("hit1", "Hit")],
    }})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    service = queue_service()
    server = serve()
    try:
        _status, by_name, _h = read_json(server, "/api/music/queue", {"action": "add", "query": "Miles"})
        _status, tail, _h = read_json(server, "/api/music/queue", {
            "action": "add_range", "playlist_id": "PL-1", "video_id": "hit1"})
    finally:
        server.shutdown()
    titles = [item["title"] for item in service.queue.snapshot()]
    assert by_name["total"] == 1 and titles[0] == "Hit"
    # The range add re-reads the playlist through the provider and queues its tail.
    reads = [item["arguments"] for item in runtime.calls if item["arguments"].get("action") == "library"]
    assert reads == [{"action": "library", "kind": "playlist", "playlist_id": "PL-1", "limit": 500}]
    assert titles[1] == "Hit"


def test_the_resolve_route_asks_the_provider_for_one_track(music_env, monkeypatch):
    """What the queue service calls back for, once per track it was handed as a
    bare id: the track's metadata, and nothing else — nothing is queued and
    nothing starts playing."""
    runtime = FakeRuntime(results={"opendeck_music": lambda matched: {
        "ok": True, "action": "queue", "message": "Resolved",
        **queue_track("bare123456", "Resolved Song"),
        "queue": [], "queue_index": -1, "count": 0,
    }})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    server = serve()
    try:
        status, payload, _h = read_json(server, "/api/music/queue/resolve", {"video_id": "bare123456"})
        missing_status, missing, _h = read_json(server, "/api/music/queue/resolve", {})
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is True
    assert payload["track"]["title"] == "Resolved Song"
    assert queue_calls(runtime) == [{"action": "queue", "op": "resolve", "video_id": "bare123456"}]
    assert not [item for item in runtime.calls if item["arguments"].get("action") == "play_url"]
    assert missing_status == 400 and missing["code"] == "validation_error"


def test_a_stopped_queue_service_is_reported_rather_than_guessed_at(music_env, monkeypatch):
    """The console must not answer a queue read from the provider while the
    service that owns the queue is unreachable: it would describe a line-up the
    speaker is not following."""
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: queue_result([])})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    monkeypatch.setattr(music_api, "QUEUE_SERVICE_URL", "http://127.0.0.1:1")
    server = serve()
    try:
        status, payload, _h = read_json(server, "/api/music/queue")
    finally:
        server.shutdown()
    assert status == 503 and payload["code"] == "queue_service_unavailable"
    assert queue_calls(runtime) == []


def test_a_published_queue_service_is_found_without_being_configured(monkeypatch, tmp_path, queue_service):
    """The console is told nothing and still uses the queue running beside it.

    The service publishes its address in the data directory, which is what makes
    the console and the skill worker both find the same queue without either
    being handed an environment variable.
    """
    published = str(tmp_path / "music-queue.json")
    service = queue_service()
    monkeypatch.setenv("NIX_MUSIC_QUEUE_FILE", published)
    monkeypatch.setattr(music_api, "QUEUE_SERVICE_URL", "")
    monkeypatch.setattr(music_api, "_queue_probe", {"at": 0.0, "url": "", "alive": False})
    assert music_api._queue_service_url() == ("", False), "nothing published yet"

    music_queue.publish_service(service.address, path=published)
    url, explicit = music_api._queue_service_url()
    assert url.startswith("http") and explicit is False
    assert music_api._queue_service_alive() is True, "it answers, so it is used"
    music_queue.withdraw_service(path=published)


def test_a_published_queue_service_that_stopped_falls_back_to_the_provider(music_env, monkeypatch, tmp_path):
    """A stopped service must not turn every queue route into an error.

    An address someone configured by hand is authoritative — its failures are
    reported. An address the service published for us is not: when it is gone,
    the console goes back to the provider skill's own queue, which is what a box
    that has just uninstalled the service needs.
    """
    runtime = FakeRuntime(results={"opendeck_music": lambda _matched: queue_result(
        [queue_track("a1", "From The Provider")], index=0, message="1 track(s) queued.")})
    install(monkeypatch, music_env, [PROVIDER_MANIFEST, player_manifest()], runtime)
    published = str(tmp_path / "music-queue.json")
    music_queue.publish_service("http://127.0.0.1:1", path=published)
    monkeypatch.setenv("NIX_MUSIC_QUEUE_FILE", published)
    monkeypatch.setattr(music_api, "QUEUE_SERVICE_URL", "")
    monkeypatch.setattr(music_api, "_queue_probe", {"at": 0.0, "url": "", "alive": False})
    server = serve()
    try:
        status, payload, _headers = read_json(server, "/api/music/queue")
    finally:
        server.shutdown()
    assert status == 200 and payload["ok"] is True
    assert [item["title"] for item in payload["queue"]] == ["From The Provider"]
    assert queue_calls(runtime) == [{"action": "queue", "op": "list"}], \
        "the provider's own queue answered, because the service was not there"
