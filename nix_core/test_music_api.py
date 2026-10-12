"""Tests for the skill-contributed music page and the music control API.

The console owns the page shell, the routes and the state; installed skills
supply the catalog (provider) and the speaker controls (player). These tests
pin that contract without any hardware or network access.
"""
import json
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import console
import console_extend
import music_api


@pytest.fixture
def music_env(monkeypatch, tmp_path):
    monkeypatch.setattr(console, "SKILLS_STATE_PATH", str(tmp_path / "skills.json"))
    monkeypatch.setattr(console, "SKILLS_INSTALL_DIR", str(tmp_path / "installed"))
    monkeypatch.setattr(console, "DATA_DIR", str(tmp_path / "data"))
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
                    {"type": "object", "properties": {"action": {"type": "string", "enum": ["queue"]}, "op": {"type": "string", "enum": ["list", "add", "remove", "clear", "next", "play", "shuffle", "repeat"]}, "query": {"type": "string", "minLength": 1, "maxLength": 200}, "video_id": {"type": "string", "maxLength": 64}, "index": {"type": "integer", "minimum": 0, "maximum": 99}, "on": {"type": "boolean"}, "mode": {"type": "string", "enum": ["off", "all", "one"]}, "auto": {"type": "boolean"}}, "required": ["action", "op"]},
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


def player_manifest(upload_control=True):
    controls = [
        {"label": "Set volume", "tool": "control_ring", "arguments": {"action": "volume"}, "fields": [{"name": "volume", "label": "Volume", "widget": "number"}]},
        {"label": "Mute", "tool": "control_ring", "arguments": {"action": "mute"}, "fields": []},
        {"label": "Stop", "tool": "control_ring", "arguments": {"action": "stop_speaking"}, "fields": []},
    ]
    if upload_control:
        controls.append(
            {"label": "Play a stream URL", "tool": "control_ring", "arguments": {"action": "play_url"}, "fields": [{"name": "url", "label": "Audio URL", "widget": "text"}, {"name": "title", "label": "Title", "widget": "text"}]}
        )
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
                        "action": {"type": "string", "enum": ["state", "volume", "mute", "stop_speaking", "play_url"]},
                        "volume": {"type": "number", "minimum": 0, "maximum": 1},
                        "url": {"type": "string", "maxLength": 2048},
                        "title": {"type": "string", "maxLength": 200},
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                    "oneOf": [
                        {"type": "object", "properties": {"action": {"type": "string", "enum": ["state", "mute", "stop_speaking"]}}, "required": ["action"]},
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


# ── manifest contract ────────────────────────────────────────────────────


def test_manifest_accepts_page_and_media_provider():
    parsed = console._parse_skill_manifest(json.loads(json.dumps(PROVIDER_MANIFEST)), "skills/opendeck/skill.json", "NIX Community Skills")
    assert parsed["device_ui"]["page"] == {"id": "music", "title": "Music", "script": "ui/music.js", "styles": ["ui/music.css"]}
    assert set(parsed["device_ui"]["media"]["provider"]) == {"search", "play", "library", "lyrics", "queue", "save"}
    assert parsed["device_ui"]["media"]["provider"]["library"]["arguments"] == {"action": "library", "kind": "playlists"}
    # The queue action carries a default op so the console can read the queue
    # with no request body at all.
    assert parsed["device_ui"]["media"]["provider"]["queue"]["arguments"] == {"action": "queue", "op": "list"}


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
