"""Music page and music control API for the NIX console.

A trusted installed skill can contribute a music page and act as the music
*provider* (search, play, library, lyrics). Transport control is resolved from
the declared device controls of whichever installed skill owns the player —
the same declarations the Devices page renders — so a skill that can play a URL
or set the volume serves this API without any extra NIX-side configuration.

Everything here is read at request time from the installed package manifests, so
installing or removing a provider skill changes the page and the API by itself.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
from typing import Any

import console
import music_cache

PAGE_ID = "music"
PAGE_TITLE = "Music"
PAGE_EXTENSIONS = {".js": "application/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
MUSIC_STATE_NAME = "music.json"
MAX_TEXT = 300

# Console page ids a skill may attach its page to. A package cannot invent a
# new dashboard route; it can only fill one the console already ships.
ALLOWED_PAGE_IDS = frozenset({PAGE_ID})

# Each transport command is matched against the declared device controls of the
# installed skills. The tuple is the preferred action names, in order.
TRANSPORT_ACTIONS: dict[str, tuple[str, ...]] = {
    "play": ("play", "play_url"),
    "pause": ("pause",),
    "resume": ("resume", "play"),
    "stop": ("stop_speaking", "stop"),
    "mute": ("mute",),
    "unmute": ("unmute",),
    "volume": ("volume",),
}


class MusicApiError(Exception):
    """A bounded, user-facing failure with an HTTP status and a stable code."""

    def __init__(self, message: str, status: int = 400, code: str = "music_error"):
        super().__init__(message)
        self.status = status
        self.code = code


# ---------------------------------------------------------------------------
# Installed skill discovery
# ---------------------------------------------------------------------------


def _saved_state() -> dict:
    with console._skills_state_lock:
        return console._read_skills_state()


def _manifest_for(skill_id: str) -> dict:
    path = os.path.join(console._skill_package_path(skill_id), "skill.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MusicApiError("The installed skill package is unreadable.", 500, "package_unreadable") from exc
    if not isinstance(manifest, dict):
        raise MusicApiError("The installed skill package is unreadable.", 500, "package_unreadable")
    return manifest


def _installed_skills() -> list[tuple[str, dict, dict]]:
    """Every installed skill as (skill_id, manifest, device_ui)."""
    saved = _saved_state()
    installed = set(saved.get("installed", []))
    skills: list[tuple[str, dict, dict]] = []
    for _repository, skill in console._remote_skill_refs(saved):
        skill_id = skill.get("id")
        if skill_id not in installed:
            continue
        try:
            manifest = _manifest_for(skill_id)
        except MusicApiError:
            continue
        device_ui = manifest.get("device_ui")
        skills.append((skill_id, manifest, device_ui if isinstance(device_ui, dict) else {}))
    return skills


def _media_block(device_ui: dict) -> dict:
    media = device_ui.get("media")
    return media if isinstance(media, dict) else {}


def _provider_skill() -> tuple[str, dict, dict]:
    """The installed skill supplying catalog/playback for the music page."""
    for skill_id, manifest, device_ui in _installed_skills():
        provider = _media_block(device_ui).get("provider")
        if isinstance(provider, dict) and provider:
            return skill_id, manifest, provider
    raise MusicApiError(
        "No installed skill provides music. Install a music provider skill (for example OpenDeck).",
        409,
        "no_music_provider",
    )


def _declared_controls() -> list[tuple[str, dict]]:
    controls: list[tuple[str, dict]] = []
    for skill_id, _manifest, device_ui in _installed_skills():
        for control in device_ui.get("controls", []) or []:
            if isinstance(control, dict) and isinstance(control.get("arguments"), dict):
                controls.append((skill_id, control))
    return controls


def _control_for(action_names: tuple[str, ...], allowed_fields: set[str] | None = None) -> tuple[str, dict, str] | None:
    """Find the declared device control implementing a transport action.

    ``allowed_fields`` is the set of input names this API knows how to fill in:
    a control matches when every input it declares is in that set (``None``
    means the control must take no inputs), and when its fixed arguments carry
    nothing beyond ``action``.
    """
    for name in action_names:
        matches: list[tuple[str, dict, str]] = []
        for skill_id, control in _declared_controls():
            arguments = control["arguments"]
            if arguments.get("action") != name:
                continue
            base = {key: value for key, value in arguments.items() if key != "action"}
            fields = {
                field.get("name")
                for field in control.get("fields", []) or []
                if isinstance(field, dict)
            }
            if allowed_fields is None:
                if fields or base:
                    continue
            elif base or not fields <= allowed_fields:
                continue
            matches.append((skill_id, control, name))
        if matches:
            # Prefer the first skill that declares it (install order is stable).
            return matches[0]
    return None


def _player_skill_id() -> str | None:
    """The skill that owns player transport, used for its status operation."""
    for command, action_names in TRANSPORT_ACTIONS.items():
        found = _control_for(action_names, {"volume"} if command == "volume" else None)
        if found:
            return found[0]
    for skill_id, _manifest, device_ui in _installed_skills():
        status_arguments = device_ui.get("status_arguments")
        if isinstance(status_arguments, dict) and str(status_arguments.get("action")) in {"state", "speaker"}:
            return skill_id
    return None


# ---------------------------------------------------------------------------
# Worker calls
# ---------------------------------------------------------------------------


# A worker keeps its own session to its device (ESPHome's native API, a LAN
# bridge, …). When that session drops — a device reboot, a Wi-Fi blip, an idle
# timeout — every later tool call keeps failing with a transport error, because
# the worker holds a client it never reconnects. These are the phrases the
# workers' clients raise for exactly that state, and one worker restart clears
# it, so a control call recovers instead of reporting a 502 for a speaker that
# is still on the network.
_DEVICE_SESSION_LOST = (
    "apiconnectionerror",
    "apiconnectionreset",
    "not connected",
    "connection lost",
    "connection reset",
    "connection refused",
    "disconnected",
    "broken pipe",
)


def _call_tool(skill_id: str, tool: str, arguments: dict, *, _retry: bool = True) -> dict:
    runtime = console._marketplace_skill_runtime()
    status = runtime.status(skill_id)
    if not status.get("runnable"):
        raise MusicApiError(
            "Configure and explicitly trust this skill before using its music API.",
            409,
            "skill_not_runnable",
        )
    if not status.get("worker_running"):
        runtime.start_skill(skill_id)
    try:
        outcome = runtime.execute(
            runtime.validate_proposal(
                {"type": "skill_tool_call", "skill_id": skill_id, "tool": tool, "arguments": arguments}
            )
        )
    except Exception as exc:  # SkillRuntimeError and worker failures alike
        detail = f"{type(exc).__name__}: {str(exc)[:200]}"
        lowered = detail.lower()
        if _retry and any(phrase in lowered for phrase in _DEVICE_SESSION_LOST):
            runtime.stop(skill_id)
            runtime.start_skill(skill_id)
            return _call_tool(skill_id, tool, arguments, _retry=False)
        raise MusicApiError(detail, 502, "skill_call_failed") from exc
    result = outcome.get("result")
    return result if isinstance(result, dict) else {}


def _provider_call(key: str, extra: dict | None = None) -> tuple[str, dict]:
    skill_id, _manifest, provider = _provider_skill()
    entry = provider.get(key)
    if not isinstance(entry, dict) or not isinstance(entry.get("tool"), str):
        raise MusicApiError(f"This music skill does not support '{key}'.", 501, "provider_action_unsupported")
    base = entry.get("arguments")
    arguments = dict(base) if isinstance(base, dict) else {}
    arguments.update(extra or {})
    return skill_id, _call_tool(skill_id, entry["tool"], arguments)


def _player_status() -> dict | None:
    skill_id = _player_skill_id()
    if skill_id is None:
        return None
    runtime = console._marketplace_skill_runtime()
    status = runtime.status(skill_id)
    if not status.get("runnable"):
        return None
    try:
        if not status.get("worker_running"):
            runtime.start_skill(skill_id)
        return runtime.device_status(skill_id)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Music state (the console remembers the track the player was handed)
# ---------------------------------------------------------------------------


def _state_path() -> str:
    return os.path.join(console.DATA_DIR, MUSIC_STATE_NAME)


def _read_music_state() -> dict:
    try:
        with open(_state_path(), "r", encoding="utf-8") as handle:
            saved = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return saved if isinstance(saved, dict) else {}


def _write_music_state(state: dict) -> None:
    path = _state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False)
    os.replace(temporary, path)


def _position(state: dict, player: dict | None) -> dict:
    """Elapsed playback time, estimated from the console's own hand-off time.

    The Echo Dot's ESPHome media player reports state and volume but no
    position, so both the dashboard page and the mobile app derive progress
    from ``started_at`` minus any paused time. Clients may keep their own clock
    between polls; this is the shared fallback.
    """
    started_at = state.get("started_at")
    duration = state.get("duration_sec") or 0
    if not isinstance(started_at, (int, float)):
        return {"elapsed_sec": None, "duration_sec": duration or None, "estimated": True, "basis": "unknown"}
    paused = _player_state_name(player) in {"paused", "idle", "off"}
    reference = state.get("paused_at") if paused and isinstance(state.get("paused_at"), (int, float)) else time.time()
    elapsed = reference - float(started_at) - float(state.get("paused_seconds") or 0.0)
    if duration:
        elapsed = min(elapsed, float(duration))
    return {
        "elapsed_sec": max(0.0, round(elapsed, 2)),
        "duration_sec": duration or None,
        "estimated": True,
        "basis": "console_started_at",
        "paused": paused,
    }


def _player_state_name(player: dict | None) -> str:
    if not isinstance(player, dict):
        return "unknown"
    speaker = player.get("speaker")
    for candidate in (
        speaker.get("state") if isinstance(speaker, dict) else None,
        (player.get("state") or {}).get("state") if isinstance(player.get("state"), dict) else None,
        player.get("playback_state"),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate.lower()
    return "unknown"


def _player_summary(player: dict | None) -> dict:
    """Flatten a player skill's reported state into the page/app contract."""
    summary: dict = {"state": _player_state_name(player), "volume": None, "muted": None, "confirmed": player is not None}
    if isinstance(player, dict):
        speaker = player.get("speaker")
        if isinstance(speaker, dict):
            summary["volume"] = speaker.get("volume")
            summary["muted"] = speaker.get("muted")
        for key in ("device_name", "device_label"):
            if isinstance(player.get(key), str) and player[key]:
                summary["device"] = player[key]
    return summary


def _now_playing_payload(include_lyrics: bool = False) -> dict:
    state = _read_music_state()
    player = _player_status()
    track = state.get("track") if isinstance(state.get("track"), dict) else None
    payload: dict = {
        "ok": True,
        "page": {"id": PAGE_ID, "title": PAGE_TITLE},
        "track": track,
        "playing": _player_state_name(player) == "playing",
        "player": _player_summary(player),
        "position": _position(state, player),
        "handoff": state.get("handoff") if isinstance(state.get("handoff"), dict) else None,
        "updated_at": state.get("updated_at"),
        "source": state.get("source"),
        # Whether this track came out of the queue and may therefore be followed
        # by the next one when it ends. A reader that is watching the end of a
        # song (the wall display) has to know the difference between "the queue
        # moves on by itself" and "this song was all there was": one is waiting,
        # the other is finished.
        "queue_active": bool(state.get("queue_active")),
    }
    if include_lyrics and track:
        try:
            _skill, lyrics = _provider_call("lyrics", _lyrics_arguments(track))
            payload["lyrics"] = lyrics.get("lyrics")
            payload["lyrics_source"] = lyrics.get("lyrics_source")
            payload["lyrics_timing"] = lyrics.get("lyrics_timing")
        except MusicApiError as exc:
            payload["lyrics"] = None
            payload["lyrics_error"] = str(exc)
    return payload


def _lyrics_arguments(track: dict) -> dict:
    arguments: dict = {}
    for key in ("video_id", "title", "artist", "album"):
        value = track.get(key)
        if isinstance(value, str) and value:
            arguments[key] = value[:MAX_TEXT]
    duration = track.get("duration_sec")
    if isinstance(duration, (int, float)) and duration > 0:
        arguments["duration_sec"] = int(duration)
    return arguments


# ---------------------------------------------------------------------------
# Page assets
# ---------------------------------------------------------------------------


def _page_blocks() -> list[tuple[str, dict, dict]]:
    blocks: list[tuple[str, dict, dict]] = []
    for skill_id, manifest, device_ui in _installed_skills():
        page = device_ui.get("page")
        if isinstance(page, dict) and page:
            blocks.append((skill_id, manifest, page))
    return blocks


def page_assets(skill_id: str, page: dict, manifest: dict) -> list[str]:
    """Declared, servable paths of one skill's contributed page."""
    declared = {path for path in manifest.get("files", []) if isinstance(path, str)}
    paths = [page.get("script")] + list(page.get("styles", []) or [])
    servable: list[str] = []
    for path in paths:
        if isinstance(path, str) and path in declared and os.path.splitext(path)[1].lower() in PAGE_EXTENSIONS:
            servable.append(path)
    return servable


def _page_capability() -> dict | None:
    for skill_id, manifest, page in _page_blocks():
        page_id = str(page.get("id") or "")
        if page_id not in ALLOWED_PAGE_IDS:
            continue
        script = page.get("script")
        if not isinstance(script, str) or script not in page_assets(skill_id, page, manifest):
            continue
        styles = [
            f"/api/skills/{urllib.parse.quote(skill_id, safe='')}/page/{style}"
            for style in page.get("styles", []) or []
            if style in page_assets(skill_id, page, manifest)
        ]
        return {
            "id": page_id,
            "title": str(page.get("title") or PAGE_TITLE)[:80],
            "skill_id": skill_id,
            "skill_name": manifest.get("name", skill_id),
            "script": f"/api/skills/{urllib.parse.quote(skill_id, safe='')}/page/{script}",
            "styles": styles,
            "api": "/api/music",
        }
    return None


# ---------------------------------------------------------------------------
# HTTP handlers (registered by console_extend.py)
# ---------------------------------------------------------------------------


def _query(handler) -> dict:
    parsed = urllib.parse.urlparse(handler.path)
    return {key: values[0] for key, values in urllib.parse.parse_qs(parsed.query).items() if values}


def _text(value: Any, field: str, limit: int = MAX_TEXT) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise MusicApiError(f"'{field}' must be text.", 400, "validation_error")
    value = re.sub(r"[\x00-\x20\x7f]+", " ", value).strip()
    if len(value) > limit:
        raise MusicApiError(f"'{field}' is too long.", 400, "validation_error")
    return value


def _flag(value: Any, field: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise MusicApiError(f"'{field}' must be true or false.", 400, "validation_error")


def _int(value: Any, field: str, minimum: int, maximum: int) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise MusicApiError(f"'{field}' must be a whole number.", 400, "validation_error") from exc
    if not minimum <= number <= maximum:
        raise MusicApiError(f"'{field}' must be between {minimum} and {maximum}.", 400, "validation_error")
    return number


def _body(handler) -> dict:
    try:
        length = int(handler.headers.get("Content-Length") or 0)
    except (TypeError, ValueError):
        return {}
    if length <= 0 or length > 64 * 1024:
        return {}
    try:
        payload = json.loads(handler.rfile.read(length) or b"{}")
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _respond(handler, payload: dict, status: int = 200, cache: tuple[str, float | None] | None = None) -> bool:
    body = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store, max-age=0")
    handler.send_header("X-Content-Type-Options", "nosniff")
    # Whether the reply came from the console's own catalog cache and how old that
    # copy is. The body stays identical either way, so a reader that ignores these
    # headers sees exactly the same JSON it always did.
    if cache is not None:
        cache_status, age = cache
        handler.send_header("X-Music-Cache", str(cache_status))
        if isinstance(age, (int, float)):
            handler.send_header("X-Music-Cache-Age", str(int(age)))
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
    return True


def _fail(handler, exc: MusicApiError) -> bool:
    return _respond(handler, {"ok": False, "code": exc.code, "error": str(exc)}, exc.status)


def _guarded(function):
    """Turn any MusicApiError raised by a handler into its JSON error reply."""

    def wrapped(handler, match=None) -> bool:
        try:
            return function(handler, match)
        except MusicApiError as exc:
            return _fail(handler, exc)
        except Exception as exc:  # never let a music route answer with a non-JSON body
            return _respond(handler, {
                "ok": False,
                "code": "music_internal_error",
                "error": f"The music API hit an unexpected error: {type(exc).__name__}: {str(exc)[:160]}",
            }, 500)

    wrapped.__name__ = function.__name__
    return wrapped


@_guarded
def music_index(handler, _match=None) -> bool:
    try:
        provider_skill, provider_manifest, provider = _provider_skill()
    except MusicApiError as exc:
        return _respond(handler, {
            "ok": False,
            "code": exc.code,
            "error": str(exc),
            "page": _page_capability(),
            "provider": None,
        }, 200)
    player_skill = _player_skill_id()
    # `play` here means a control that starts playback on its own. A player that
    # plays a stream URL declares `play_url`, which takes the resolved URL from
    # POST /api/music/play instead of /api/music/control, so it is reported as
    # unavailable on purpose: the console never calls it that way.
    transport = {
        command: _control_for(actions, {"volume"} if command == "volume" else None) is not None
        for command, actions in TRANSPORT_ACTIONS.items()
    }
    return _respond(handler, {
        "ok": True,
        "api_version": "1",
        "page": _page_capability(),
        "provider": {
            "skill_id": provider_skill,
            "skill_name": provider_manifest.get("name", provider_skill),
            "actions": sorted(key for key in provider if isinstance(provider.get(key), dict)),
        },
        "player": {"skill_id": player_skill, "commands": transport},
        "endpoints": [
            "GET /api/music",
            "GET /api/music/now-playing",
            "POST /api/music/control",
            "GET /api/music/search?q=&limit=",
            "POST /api/music/play",
            "GET /api/music/queue",
            "POST /api/music/queue",
            "GET /api/music/library?kind=&limit=&playlist_id=",
            "POST /api/music/save",
            "GET /api/music/lyrics",
            "GET /api/music/artwork?src=",
        ],
        "cache": music_cache.stats(),
        "documentation": "nix_core/MUSIC_API.md",
    })


def _maybe_advance_queue() -> None:
    """Start the queue's next track once the current one has finished.

    The speaker reports no end-of-track event, so the console notices on the
    now-playing poll: the queue-sourced track is at (or past) its length and the
    speaker has gone idle. A one-off play never auto-advances, and a paused
    track is not finished, so this cannot hijack a deliberate pause.
    """
    state = _read_music_state()
    if not state.get("queue_active"):
        return
    player = _player_status()
    if _player_state_name(player) not in {"idle", "off", "stopped"}:
        return
    duration = state.get("duration_sec") or 0
    elapsed = _position(state, player).get("elapsed_sec") or 0
    if not duration or elapsed < duration - 1:
        return
    try:
        # ``auto`` tells the provider this is a track that ended by itself, not a
        # skip: under repeat-one it restarts the same track, while an explicit
        # skip from the page moves on.
        skill_id, result = _provider_call("queue", {"op": "next", "auto": True})
    except MusicApiError:
        return
    if not result.get("play_url"):
        return
    _start_playback(result, skill_id, queue={"index": result.get("queue_index")})


@_guarded
def music_now_playing(handler, _match=None) -> bool:
    try:
        include_lyrics = _query(handler).get("lyrics") in {"1", "true"}
        if not include_lyrics:
            _maybe_advance_queue()
        return _respond(handler, _now_playing_payload(include_lyrics))
    except MusicApiError as exc:
        return _fail(handler, exc)


# The page, the mobile app and the assistant all drive the same speaker, so one
# implementation runs each transport command for all of them. The wording is
# what the assistant says back: it states what the speaker confirmed, never more
# (a paused speaker is never reported as stopped).
TRANSPORT_MESSAGES: dict[str, str] = {
    "play": "The speaker is playing.",
    "pause": "Music paused.",
    "resume": "Music resumed.",
    # "stop" is the speaker's voice control, never the music: the assistant
    # translates "stop the music" into pause itself.
    "stop": "The speaker stopped.",
    "mute": "The speaker is muted.",
    "unmute": "The speaker is unmuted.",
    "volume": "Speaker volume set to {percent}%.",
}


def apply_transport(command: str, value: Any = None) -> dict:
    """Run one transport command against the installed speaker controls.

    Resolves the declared device control for the command, commands the worker
    once, and keeps the console's own playback bookkeeping (the position clock)
    in step, so every caller sees the same state. Raises MusicApiError with a
    user-facing message when nothing declares the control or the speaker
    refuses; the caller decides how to phrase the failure.
    """
    command = str(command or "").strip().lower()
    if command not in TRANSPORT_ACTIONS:
        raise MusicApiError(
            f"action must be one of: {', '.join(sorted(TRANSPORT_ACTIONS))}.", 400, "validation_error")
    needs_value = command == "volume"
    found = _control_for(TRANSPORT_ACTIONS[command], {"volume"} if needs_value else None)
    if found is None:
        raise MusicApiError(
            f"No installed skill declares a '{command}' player control. Install or update the skill that owns the speaker.",
            501,
            "player_action_unavailable",
        )
    skill_id, control, action = found
    arguments = dict(control["arguments"])
    if needs_value:
        try:
            level = float(value)
        except (TypeError, ValueError) as exc:
            raise MusicApiError("'value' must be a number between 0 and 1.", 400, "validation_error") from exc
        if not 0.0 <= level <= 1.0:
            raise MusicApiError("'value' must be between 0 and 1.", 400, "validation_error")
        arguments["volume"] = level
    result = _call_tool(skill_id, control["tool"], arguments)
    state = _read_music_state()
    if state:
        now = time.time()
        if command in {"pause", "stop"}:
            state["paused_at"] = now
        elif command in {"play", "resume"}:
            if isinstance(state.get("paused_at"), (int, float)):
                state["paused_seconds"] = float(state.get("paused_seconds") or 0.0) + max(0.0, now - float(state["paused_at"]))
            state["paused_at"] = None
        state["updated_at"] = now
        _write_music_state(state)
    player = _player_status()
    summary = _player_summary(player)
    payload = {
        "ok": True,
        "command": command,
        "skill_id": skill_id,
        "control": control.get("label"),
        "result": result,
        "player": summary,
    }
    level = summary.get("volume") if isinstance(summary, dict) else None
    payload["message"] = TRANSPORT_MESSAGES[command].format(
        percent=int(round(float(level) * 100)) if isinstance(level, (int, float)) else 0
    )
    return payload


@_guarded
def music_control(handler, _match=None) -> bool:
    body = _body(handler)
    command = _text(body.get("action") or body.get("command"), "action", 32).lower()
    value = body.get("value", body.get("volume"))
    return _respond(handler, apply_transport(command, value))


# ---------------------------------------------------------------------------
# Catalog cache
# ---------------------------------------------------------------------------


# The page, the mobile app and the assistant all read the same catalog, and a
# playlist or a search answer does not change second by second. These reads are
# served from the console's own copy and refreshed in the background, so opening
# the Music page never waits on the device for data it already has. Writes that
# change the account invalidate the copy they affect.
def _cache_key(arguments: dict) -> str:
    """A stable key for one provider read: its arguments, in a fixed order."""
    return "|".join(f"{name}={arguments[name]}" for name in sorted(arguments))


def _refresh_requested(handler) -> bool:
    """Whether a reader asked to skip the cached copy (``?refresh=1``)."""
    return str(_query(handler).get("refresh") or "").strip().lower() in {"1", "true", "yes"}


def _cached_provider_read(namespace: str, provider_key: str, arguments: dict, refresh: bool) -> tuple[dict, str, float | None]:
    """Answer one catalog read from the cache, refreshing it when it goes stale."""

    def fetch() -> dict:
        skill_id, result = _provider_call(provider_key, arguments)
        return {"ok": True, "skill_id": skill_id, **result}

    return music_cache.read(namespace, _cache_key(arguments), fetch, refresh=refresh)


@_guarded
def music_search(handler, _match=None) -> bool:
    query = _query(handler)
    text = _text(query.get("q") or query.get("query"), "q", 200)
    if not text:
        return _fail(handler, MusicApiError("Provide q=<search text>.", 400, "validation_error"))
    arguments: dict = {"query": text}
    limit = _int(query.get("limit"), "limit", 1, 50)
    if limit:
        arguments["limit"] = limit
    try:
        payload, cache_status, age = _cached_provider_read("search", "search", arguments, _refresh_requested(handler))
    except MusicApiError as exc:
        return _fail(handler, exc)
    return _respond(handler, payload, cache=(cache_status, age))


def _start_playback(result: dict, skill_id: str, *, queue: dict | None = None) -> dict:
    """Record the handed-over track and command the speaker to play it.

    Shared by POST /api/music/play and by playing an item from the queue so both
    paths hand the URL over and anchor the position clock identically.
    """
    play_url = result.get("play_url") if isinstance(result.get("play_url"), str) else ""
    now = time.time()
    state = {
        "track": {
            "video_id": result.get("video_id"),
            "title": result.get("title"),
            "artist": result.get("artist"),
            "album": result.get("album"),
            "duration_sec": result.get("duration_sec"),
            "artwork_url": result.get("artwork_url"),
            "mime_type": result.get("mime_type"),
            "play_url": play_url,
        },
        "started_at": now,
        "paused_at": None,
        "paused_seconds": 0.0,
        "duration_sec": result.get("duration_sec") or 0,
        "updated_at": now,
        "source": skill_id,
        # A track started from the queue may advance by itself when it ends; a
        # one-off play never does.
        "queue_active": bool(queue),
        "queue_index": (queue or {}).get("index"),
        "handoff": {"supported": False, "reason": "No installed skill declares a URL playback control."},
    }
    handoff = _control_for(("play_url",), {"url", "title"})
    if handoff is not None and play_url:
        handoff_skill, handoff_control, _action = handoff
        fields = {
            field.get("name")
            for field in handoff_control.get("fields", []) or []
            if isinstance(field, dict)
        }
        if "url" in fields:
            handoff_arguments = dict(handoff_control["arguments"])
            handoff_arguments["url"] = play_url
            if "title" in fields:
                handoff_arguments["title"] = str(result.get("title") or "")[:200]
            # Playback starts when the speaker skill receives this command, not
            # when the provider started resolving the track. Resolving plus
            # converting can take seconds, so anchoring the position clock at
            # the earlier moment made the page's progress bar and the lyrics
            # run ahead of the audio (about a line, on a converted track).
            state["started_at"] = time.time()
            try:
                handoff_result = _call_tool(handoff_skill, handoff_control["tool"], handoff_arguments)
                # The control matched and the call came back, but a speaker that
                # refused the stream reports ok:false. Treating that as success is
                # exactly how a reply claimed "Playing …" while nothing reached the
                # device (measured live), so a refusal is recorded as the error it
                # is and the reply says so.
                refused = (
                    isinstance(handoff_result, dict)
                    and handoff_result.get("ok") is False
                )
                if refused:
                    state["handoff"] = {
                        "supported": True,
                        "skill_id": handoff_skill,
                        "result": handoff_result,
                        "error": str(
                            handoff_result.get("error")
                            or "the speaker refused the stream"
                        )[:200],
                    }
                else:
                    state["handoff"] = {"supported": True, "skill_id": handoff_skill, "result": handoff_result}
            except MusicApiError as exc:
                state["handoff"] = {"supported": True, "skill_id": handoff_skill, "error": str(exc)}
    _write_music_state(state)
    return state


def play_options(result: dict) -> list[dict] | None:
    """The matches a provider refused to choose between, or None when it played.

    A provider that cannot tell which of several catalog matches a request meant
    answers with the candidates instead of silently playing the first one. Core
    turns that into a question for the user, so the guess is never made here.
    """
    if not isinstance(result, dict) or not result.get("ambiguous"):
        return None
    options = result.get("options")
    if not isinstance(options, list):
        return []
    return [item for item in options if isinstance(item, dict)]


def _question_for_play(result: dict, offered: list[dict], query: str) -> dict:
    """Park the provider's matches as the question the panel asks."""
    # Deferred: the panel module imports this one for the music handlers.
    import puca_panel

    count = len(offered)
    prompt = str(result.get("message") or "").strip()
    if not prompt:
        prompt = (
            f"{count} tracks match “{query}”: which one did you mean?"
            if query
            else f"{count} tracks match: which one did you mean?"
        )
    return puca_panel.ask_choice(prompt, offered, source="play", query=query)


@_guarded
def music_play(handler, _match=None) -> bool:
    body = _body(handler)
    query = _text(body.get("query"), "query", 200)
    video_id = _text(body.get("video_id"), "video_id", 64)
    if not query and not video_id:
        return _fail(handler, MusicApiError("Provide query=<search text> or video_id=<id>.", 400, "validation_error"))
    arguments = {"query": query} if query else {"video_id": video_id}
    try:
        skill_id, result = _provider_call("play", arguments)
    except MusicApiError as exc:
        return _fail(handler, exc)
    offered = play_options(result)
    payload = _now_playing_payload()
    if offered is not None:
        # Nothing played: the answer is the question, and the page/panel shows
        # the choices. `/api/panel/state` carries the same question, so the wall
        # display asks even when the request came from somewhere else.
        question = _question_for_play(result, offered, query)
        # Whether the question has a screen to be asked on travels with it: a
        # page that tells the user to look at a panel that is not installed is
        # worse than one that tells them to say which track they meant.
        import puca_panel

        panel = puca_panel.panel_info()
        return _respond(handler, {
            "ok": True,
            "ambiguous": True,
            "skill_id": skill_id,
            "message": result.get("message") or "",
            "question": question,
            "options": question["options"],
            "panel_available": bool(panel.get("available")),
            "panel_url": panel.get("url"),
            **{key: payload[key] for key in ("track", "playing", "player", "position", "handoff")},
        })
    _start_playback(result, skill_id)
    payload = _now_playing_payload()
    return _respond(handler, {"ok": True, "skill_id": skill_id, "result": result, **{
        key: payload[key] for key in ("track", "playing", "player", "position", "handoff")
    }})


# The provider's own queue vocabulary (see the skill's declared op enum). The
# route mirrors it rather than a subset, so a page that can drag a row to a new
# position is not answered with a 400 for an operation the provider declares.
QUEUE_OPS = {
    "list", "add", "remove", "clear", "next", "play", "move", "front", "next_up",
    "shuffle", "repeat",
}
REPEAT_MODES = ("off", "all", "one")

# How many songs one library read may hand over. This is the reader's ceiling,
# not the provider's page size: a wall display listing a playlist asks for as
# many as it will show, and the provider answers with the list it holds.
LIBRARY_LIMIT_MAX = 200


def _queue_payload(skill_id: str, result: dict) -> dict:
    """Normalize the provider's queue result into the page/app contract."""
    queue = result.get("queue") if isinstance(result.get("queue"), list) else []
    index = result.get("queue_index")
    index = index if isinstance(index, int) and -1 <= index < len(queue) else -1
    repeat = result.get("repeat")
    return {
        "ok": True,
        "skill_id": skill_id,
        "action": "queue",
        "queue": queue,
        "index": index,
        "count": len(queue),
        # Shuffle/repeat travel with the queue so a page can render its toggles
        # from the same reply that changed or listed the queue.
        "shuffle": bool(result.get("shuffle")),
        "repeat": repeat if repeat in REPEAT_MODES else "off",
        "message": result.get("message") or "",
    }


@_guarded
def music_queue(handler, _match=None) -> bool:
    """Read or change the play queue, or start the queue's next track.

    The queue itself lives in the music provider skill (so the assistant can
    queue tracks through its own tool); the console reads and drives it through
    the declared provider action and performs the speaker hand-off here, exactly
    as POST /api/music/play does.
    """
    if handler.command == "GET":
        try:
            skill_id, result = _provider_call("queue", {"op": "list"})
        except MusicApiError as exc:
            return _fail(handler, exc)
        return _respond(handler, _queue_payload(skill_id, result))

    body = _body(handler)
    op = (_text(body.get("action") or body.get("op"), "action", 16) or "list").lower()
    if op not in QUEUE_OPS:
        return _fail(handler, MusicApiError(
            f"action must be one of: {', '.join(sorted(QUEUE_OPS))}.", 400, "validation_error"))
    arguments: dict = {"op": op}
    if op == "add":
        query = _text(body.get("query"), "query", 200)
        video_id = _text(body.get("video_id"), "video_id", 64)
        if not query and not video_id:
            return _fail(handler, MusicApiError("Provide query=<search text> or video_id=<id>.", 400, "validation_error"))
        if query:
            arguments["query"] = query
        else:
            arguments["video_id"] = video_id
    elif op in {"remove", "play"}:
        index = _int(body.get("index"), "index", 0, 99)
        if index is None:
            return _fail(handler, MusicApiError("Provide index=<queue position>.", 400, "validation_error"))
        arguments["index"] = index
    elif op == "move":
        # Dragging a row to a new position: "move the third track to first" is
        # the same reorder, in one call.
        index = _int(body.get("index"), "index", 0, 99)
        to = _int(body.get("to"), "to", 0, 99)
        if index is None or to is None:
            return _fail(handler, MusicApiError("Provide index=<from> and to=<to>.", 400, "validation_error"))
        arguments["index"] = index
        arguments["to"] = to
    elif op in {"front", "next_up"}:
        # "Play next" / "Play first" on a queued row (or on a track that is not
        # queued yet, which the provider inserts at that spot).
        index = _int(body.get("index"), "index", 0, 99)
        query = _text(body.get("query"), "query", 200)
        video_id = _text(body.get("video_id"), "video_id", 64)
        if index is None and not query and not video_id:
            return _fail(handler, MusicApiError(
                "Provide index=<queue position>, query=<track> or video_id=<id>.", 400, "validation_error"))
        if index is not None:
            arguments["index"] = index
        if query:
            arguments["query"] = query
        if video_id:
            arguments["video_id"] = video_id
    elif op == "shuffle":
        # No `on` means "toggle", which is what one press of the Shuffle button
        # sends; an explicit true/false is honoured as given.
        on = _flag(body.get("on"), "on")
        if on is not None:
            arguments["on"] = on
    elif op == "repeat":
        # No `mode` cycles off -> all -> one, so a page (or an app) can send one
        # press without reading the current state first.
        mode = _text(body.get("mode"), "mode", 8).lower()
        if mode and mode not in REPEAT_MODES:
            return _fail(handler, MusicApiError(f"mode must be one of: {', '.join(REPEAT_MODES)}.", 400, "validation_error"))
        if mode:
            arguments["mode"] = mode
    try:
        skill_id, result = _provider_call("queue", arguments)
    except MusicApiError as exc:
        return _fail(handler, exc)
    payload = _queue_payload(skill_id, result)
    if op in {"play", "next"}:
        # The provider resolved the queued track; hand its URL to the speaker and
        # record it as now-playing. An exhausted queue carries no play_url.
        if not result.get("play_url"):
            payload["ok"] = False
            payload["code"] = "queue_empty"
            payload["error"] = result.get("message") or "The queue has no track to play."
            return _respond(handler, payload, 409)
        _start_playback(result, skill_id, queue={"index": payload["index"]})
        snapshot = _now_playing_payload()
        payload.update({key: snapshot[key] for key in ("track", "playing", "player", "position", "handoff")})
    return _respond(handler, payload)


@_guarded
def music_library(handler, _match=None) -> bool:
    query = _query(handler)
    kind = _text(query.get("kind") or "playlists", "kind", 32).lower()
    arguments: dict = {"kind": kind}
    # A library read is a list to browse rather than a page of answers: the wall
    # display lists a playlist's songs, and a playlist is not twelve songs long.
    # The provider caps this where it has to (history at 50); what it hands over
    # is what the reader gets.
    limit = _int(query.get("limit"), "limit", 1, LIBRARY_LIMIT_MAX)
    playlist_id = _text(query.get("playlist_id"), "playlist_id", 100)
    if limit:
        arguments["limit"] = limit
    if playlist_id:
        arguments["playlist_id"] = playlist_id
    try:
        payload, cache_status, age = _cached_provider_read("library", "library", arguments, _refresh_requested(handler))
    except MusicApiError as exc:
        return _fail(handler, exc)
    return _respond(handler, payload, cache=(cache_status, age))


# Library writes the Music page performs on the user's own OpenDeck account:
# liking (or unliking) the track, and putting a track into a playlist.
SAVE_KINDS = ("like", "unlike", "playlist_add", "playlist_create")


@_guarded
def music_save(handler, _match=None) -> bool:
    '''Save a track in the user's OpenDeck library.

    The page's Like and Add-to-playlist controls use this. The provider skill is
    the only thing that writes to OpenDeck; this handler fills in the track that
    is playing when the caller names none, so a like never needs the page to
    know a video id it does not display.
    '''
    body = _body(handler)
    kind = _text(body.get("kind") or "like", "kind", 32).lower()
    if kind not in SAVE_KINDS:
        return _fail(handler, MusicApiError(
            "kind must be one of: " + ", ".join(SAVE_KINDS) + ".", 400, "validation_error"))
    arguments: dict = {"kind": kind}
    if kind == "playlist_create":
        name = _text(body.get("name"), "name", 120)
        if not name:
            return _fail(handler, MusicApiError("Provide name=<playlist name>.", 400, "validation_error"))
        arguments["name"] = name
    else:
        state = _read_music_state()
        playing = state.get("track") if isinstance(state.get("track"), dict) else {}
        video_id = _text(body.get("video_id"), "video_id", 64) or _text(playing.get("video_id"), "video_id", 64)
        if not video_id:
            return _fail(handler, MusicApiError(
                "Nothing is playing and no video_id was given, so there is no track to save.",
                409, "no_track"))
        arguments["video_id"] = video_id
        for key, limit in (("title", 300), ("artist", 200), ("album", 300)):
            value = _text(body.get(key), key, limit) or _text(playing.get(key), key, limit)
            if value:
                arguments[key] = value
        if kind == "playlist_add":
            playlist_id = _text(body.get("playlist_id"), "playlist_id", 100)
            if not playlist_id:
                return _fail(handler, MusicApiError("Provide playlist_id=<playlist>.", 400, "validation_error"))
            arguments["playlist_id"] = playlist_id
    try:
        skill_id, result = _provider_call("save", arguments)
    except MusicApiError as exc:
        return _fail(handler, exc)
    # Liking a track or putting it in a playlist changes the library the page is
    # showing, so the cached catalog is dropped rather than replayed as if the
    # write had not happened. The next read fetches and re-caches it.
    music_cache.invalidate()
    payload = {
        "ok": True,
        "skill_id": skill_id,
        "action": "save",
        "kind": kind,
        "message": result.get("message") or "",
        "count": result.get("count"),
        "track": {key: arguments[key] for key in ("video_id", "title", "artist", "album") if arguments.get(key)},
    }
    if kind in {"like", "unlike"}:
        # The page lights its Like button from this, so the reply says what the
        # write made true instead of the page inferring it from a count.
        payload["liked"] = kind == "like"
    if kind == "playlist_add":
        payload["playlist_id"] = arguments["playlist_id"]
    if kind == "playlist_create":
        created = (result.get("playlists") or [{}])[0]
        payload["playlist_id"] = created.get("playlist_id")
        payload["name"] = created.get("name") or arguments["name"]
        payload["playlists"] = result.get("playlists") or []
    return _respond(handler, payload)


@_guarded
def music_lyrics(handler, _match=None) -> bool:
    query = _query(handler)
    state = _read_music_state()
    track = state.get("track") if isinstance(state.get("track"), dict) else {}
    arguments = _lyrics_arguments(track)
    arguments.update({key: value for key, value in (
        ("video_id", _text(query.get("video_id"), "video_id", 64)),
        ("title", _text(query.get("title"), "title", 200)),
        ("artist", _text(query.get("artist"), "artist", 200)),
        ("album", _text(query.get("album"), "album", 200)),
    ) if value})
    duration = _int(query.get("duration_sec"), "duration_sec", 1, 60 * 60)
    if duration:
        arguments["duration_sec"] = duration
    if not arguments.get("video_id") and not arguments.get("title"):
        return _fail(handler, MusicApiError("Nothing is playing and no video_id/title was given.", 409, "no_track"))
    try:
        skill_id, result = _provider_call("lyrics", arguments)
    except MusicApiError as exc:
        return _fail(handler, exc)
    player = _player_status()
    return _respond(handler, {
        "ok": True,
        "skill_id": skill_id,
        "position": _position(state, player),
        "player": _player_summary(player),
        **result,
    })


@_guarded
def music_artwork(handler, _match=None) -> bool:
    """Serve a cover from the console's own artwork cache.

    The Music page paints covers through this route, so a cover the console has
    already seen is served from local disk — and from the browser's own cache
    after the first paint — instead of a CDN round trip. The key is the image
    URL itself, so the same key always means the same image and the reply is safe
    to keep; a caller that already holds it is answered with 304.
    """
    source = _text(_query(handler).get("src"), "src", 1000)
    if not source:
        return _fail(handler, MusicApiError("Provide src=<image url>.", 400, "validation_error"))
    try:
        body, content_type, etag, cache_status = music_cache.artwork(source)
    except music_cache.ArtworkError as exc:
        return _fail(handler, MusicApiError(str(exc), exc.status, exc.code))
    not_modified = handler.headers.get("If-None-Match") == etag
    handler.send_response(304 if not_modified else 200)
    handler.send_header("Cache-Control", f"public, max-age={music_cache.ARTWORK_BROWSER_SECONDS}, immutable")
    if not_modified:
        handler.send_header("ETag", etag)
        handler.end_headers()
        return True
    handler.send_header("Content-Type", content_type)
    handler.send_header("ETag", etag)
    handler.send_header("X-Music-Cache", cache_status)
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
    return True


@_guarded
def music_page_asset(handler, match) -> bool:
    skill_id = urllib.parse.unquote(match.group(1))
    asset = urllib.parse.unquote(match.group(2))
    saved = _saved_state()
    if skill_id not in set(saved.get("installed", [])):
        return _respond(handler, {"ok": False, "error": "No such installed skill."}, 404)
    try:
        manifest = _manifest_for(skill_id)
    except MusicApiError as exc:
        return _fail(handler, exc)
    device_ui = manifest.get("device_ui")
    page = device_ui.get("page") if isinstance(device_ui, dict) else None
    if not isinstance(page, dict) or asset not in page_assets(skill_id, page, manifest):
        return _respond(handler, {"ok": False, "error": "That page asset is not declared by this skill."}, 404)
    package = os.path.realpath(os.path.abspath(console._skill_package_path(skill_id)))
    target = os.path.realpath(os.path.join(package, *asset.split("/")))
    if os.path.commonpath((package, target)) != package or not os.path.isfile(target):
        return _respond(handler, {"ok": False, "error": "Page asset not found."}, 404)
    try:
        with open(target, "rb") as handle:
            body = handle.read()
    except OSError:
        return _respond(handler, {"ok": False, "error": "Page asset not found."}, 404)
    content_type = PAGE_EXTENSIONS[os.path.splitext(asset)[1].lower()]
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Cache-Control", "no-cache, max-age=0")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
    return True
