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
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import console
import music_cache
import music_queue as queue_engine
import panel_stream

PAGE_ID = "music"
PAGE_TITLE = "Music"
# What a declared page may load from its own package. The script and stylesheets
# are the page; the fonts and pictures are what those files point at, so a page
# paints with the provider's own typography instead of a font CDN (the wall
# panel ships its fonts in the same package).
PAGE_EXTENSIONS = {
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".woff2": "font/woff2",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
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


def _adopt_unrecorded_resume(state: dict, player: dict | None) -> dict:
    """Adopt a resume that did not come through the console.

    ``paused_at`` is what makes the position clock stand still, and only the
    console's own Pause sets it. A track resumed by voice, by the speaker's
    button or from another client therefore kept reading as paused: its progress
    bar sat where it was and, once the queue was involved, the finished-track
    check treated it as stopped. The speaker saying "playing" is the authority,
    so the pause record is closed out here - the same bookkeeping apply_transport
    does for a resume it was asked to make.
    """
    if _player_state_name(player) != "playing":
        return state
    paused_at = state.get("paused_at")
    if not isinstance(paused_at, (int, float)):
        return state
    now = time.time()
    state = dict(state)
    state["paused_seconds"] = float(state.get("paused_seconds") or 0.0) + max(0.0, now - float(paused_at))
    state["paused_at"] = None
    state["updated_at"] = now
    _write_music_state(state)
    return state


def _now_playing_payload(include_lyrics: bool = False) -> dict:
    state = _read_music_state()
    player = _player_status()
    state = _adopt_unrecorded_resume(state, player)
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


def _normalized_title(value: Any) -> str:
    """A track title folded for comparison: case, punctuation and spacing only."""
    return " ".join(re.sub(r"[^\w\s]+", " ", str(value or "").casefold()).split())


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
    paths = [page.get("script")] + list(page.get("styles", []) or []) + list(page.get("assets", []) or [])
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
            "POST /api/music/queue/resolve",
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
    if _queue_service_on() and _queue_service_alive():
        _advance_through_queue_service()
        return
    state = _read_music_state()
    if not state.get("queue_active"):
        return
    # A paused speaker also reports itself idle, and the console's clock only
    # stands still for a pause it made itself (paused_at). A track paused on the
    # speaker - by its own button, by voice, or by another client - therefore kept
    # counting to its end and the console advanced the queue under it, dropping
    # the track from the queue and starting the next one with nobody asking. The
    # recorded pause is what says "this stopped because someone stopped it".
    if isinstance(state.get("paused_at"), (int, float)):
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
    panel_stream.notify("transport", command=command)
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
    # What is playing just changed, and the wall display is the surface that
    # shows it: tell it now instead of waiting for its next poll.
    panel_stream.notify("playing")
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
    # "Add this song and everything under it": the songs from one row to the end
    # of the playlist it was listed in, which is the button on a playlist page.
    "add_range",
    # A whole list in one write, and a read-only resolution of one track. Both
    # belong to the independent queue service (see THE QUEUE SERVICE below): a
    # playlist of four hundred songs is one call rather than four hundred.
    "add_many", "resolve",
}
REPEAT_MODES = ("off", "all", "one")

# How far into a queue a position may point. The queue itself is capped by the
# provider the limits are read from; this is only the range the API accepts, so
# a request cannot name a position the queue could never hold.
QUEUE_INDEX_MAX = 4999

# ---------------------------------------------------------------------------
# THE QUEUE SERVICE
# ---------------------------------------------------------------------------
# The play queue lives in a service of its own (nix_core/music_queue_service.py)
# so that a playlist longer than the skill's own hundred-track ceiling is stored
# once and read in windows, and so that warming the tracks ahead of the cursor
# is a background job rather than a page's polling traffic. It is reached over
# HTTP; setting NIX_MUSIC_QUEUE_URL is what turns it on. With the variable
# unset, every route below behaves exactly as it did before, through the
# provider skill — which is what a box that has not started the service wants.
QUEUE_SERVICE_URL = os.environ.get("NIX_MUSIC_QUEUE_URL", "").strip()
QUEUE_SERVICE_TIMEOUT = float(os.environ.get("NIX_MUSIC_QUEUE_TIMEOUT", "20"))
# How long a liveness probe of a *discovered* service is trusted. Nothing is
# probed when the address was configured by hand: an address someone wrote down
# is answered from, and its failures are reported rather than papered over.
QUEUE_PROBE_TTL = 5.0
_queue_probe = {"at": 0.0, "url": "", "alive": False}
# How long the provider's declared limits are trusted before being read from the
# installed manifest and pushed again. Installing a different music skill is
# what changes them, and that is not a per-request event.
QUEUE_LIMITS_TTL = 30.0
_queue_limits = {"at": 0.0, "skill_id": "", "values": None}


def _queue_service_url() -> tuple[str, bool]:
    """Where the queue service is, and whether that was configured by hand.

    The environment wins; otherwise the address the service published in the
    data directory is used, which is how a console that was told nothing still
    finds the queue running beside it.
    """
    if QUEUE_SERVICE_URL:
        return QUEUE_SERVICE_URL.rstrip("/"), True
    found = queue_engine.discover_service()
    if found:
        return found["url"], False
    return "", False


def _queue_service_token() -> str:
    token = os.environ.get("NIX_MUSIC_QUEUE_TOKEN", "").strip()
    if token:
        return token
    found = queue_engine.discover_service()
    return found["token"] if found else ""


def _queue_service_on() -> bool:
    return bool(_queue_service_url()[0])


def _queue_service_alive() -> bool:
    """Is a *discovered* service answering? Cached, because it is a probe."""
    url, explicit = _queue_service_url()
    if not url or explicit:
        return bool(url)
    now = time.time()
    if _queue_probe["url"] == url and now - _queue_probe["at"] < QUEUE_PROBE_TTL:
        return bool(_queue_probe["alive"])
    alive = True
    try:
        with urllib.request.urlopen(url + "/health", timeout=2):
            pass
    except (urllib.error.URLError, OSError, ValueError):
        alive = False
    _queue_probe.update({"at": now, "url": url, "alive": alive})
    return alive


def _queue_service_request(method: str, path: str, body: dict | None = None,
                           timeout: float | None = None) -> dict:
    """One call to the queue service, as a bounded, reported failure."""
    url = _queue_service_url()[0] + path
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    token = _queue_service_token()
    if token:
        headers["X-Nix-Token"] = token
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout or QUEUE_SERVICE_TIMEOUT) as response:
            payload = json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read() or b"{}").get("error", "")
        except (ValueError, AttributeError):
            detail = ""
        raise MusicApiError(
            detail or f"The music queue service refused the request ({exc.code}).",
            409 if exc.code == 409 else 502, "queue_service_error") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise MusicApiError(
            "The music queue service is not reachable.", 503, "queue_service_unavailable") from exc
    if not isinstance(payload, dict):
        raise MusicApiError("The music queue service sent an unreadable reply.", 502,
                            "queue_service_error")
    return payload


def _push_queue_limits(*, force: bool = False) -> dict:
    """Tell the queue service what this OpenDeck instance allows.

    The limits are the provider skill's own declaration, read from the installed
    manifest, so a queue that is queued against a skill with a different ceiling
    follows that skill. The read is cached; the push is not, because the service
    may have been restarted under us and would otherwise keep its old defaults.
    """
    if not _queue_service_on():
        return {}
    now = time.time()
    if not force and _queue_limits["values"] is not None and now - _queue_limits["at"] < QUEUE_LIMITS_TTL:
        return _queue_limits["values"]
    values = queue_engine.QueueLimits().as_dict()
    skill_id = ""
    try:
        skill_id, manifest, _provider = _provider_skill()
        values = queue_engine.limits_from_manifest(manifest).as_dict()
    except MusicApiError:
        pass
    try:
        reply = _queue_service_request("POST", "/limits", {"limits": values}, timeout=5)
        values = reply.get("limits") if isinstance(reply.get("limits"), dict) else values
    except MusicApiError:
        return _queue_limits["values"] or values
    _queue_limits.update({"at": now, "skill_id": skill_id, "values": values})
    return values


def _queue_service_state(offset: int, limit: int, revision: int | None) -> dict:
    _push_queue_limits()
    query = f"/state?from={max(0, offset)}&limit={max(1, min(2000, limit))}"
    if revision is not None:
        query += f"&revision={max(0, revision)}"
    return _queue_service_request("GET", query)


def _queue_service_op(body: dict) -> dict:
    _push_queue_limits()
    return _queue_service_request("POST", "/ops", body)


def _queue_service_add(tracks: list[dict], playlist_id: str = "", mode: str = "append") -> dict:
    _push_queue_limits()
    return _queue_service_request("POST", "/add", {
        "tracks": tracks, "playlist_id": playlist_id, "mode": mode,
    })


def _queue_service_message(reply: dict) -> str:
    accepted = reply.get("accepted") or 0
    total = reply.get("total") or 0
    refused = reply.get("refused") or 0
    if accepted == 1:
        message = f"1 track queued · {total} in the queue."
    else:
        message = f"{accepted} tracks queued · {total} in the queue."
    if refused:
        message += (f" {refused} did not fit: the queue holds {reply.get('max_tracks', 'no more')}."
                    if reply.get("max_tracks") else f" {refused} did not fit.")
    return message


def _queue_service_payload(reply: dict) -> dict:
    """The service's window, in the shape every existing reader already knows.

    ``queue`` is the window, ``count`` is how many are in it (as before) and
    ``total`` is the whole line-up, which is the number a page puts beside the
    queue. ``revision`` is what a polling reader sends back to be told
    "unchanged" without the queue being sent again.
    """
    items = reply.get("queue") if isinstance(reply.get("queue"), list) else []
    index = reply.get("index")
    index = index if isinstance(index, int) and -1 <= index < (reply.get("total") or 0) else -1
    return {
        "ok": True,
        "action": "queue",
        "queue": items,
        "index": index,
        "count": len(items),
        "total": int(reply.get("total") or 0),
        "offset": int(reply.get("offset") or 0),
        "revision": int(reply.get("revision") or 0),
        "unchanged": bool(reply.get("unchanged")),
        "shuffle": bool(reply.get("shuffle")),
        "repeat": reply.get("repeat") if reply.get("repeat") in REPEAT_MODES else "off",
        "message": reply.get("message") or "",
    }

# How many songs one library read may hand over. This is the reader's ceiling,
# not the provider's page size: a wall display listing a playlist asks for as
# many as it will show, and the provider answers with the list it holds.
# How many songs one library read may hand over. A real library is bigger than a
# page of results — a Liked Music playlist can hold hundreds — and both the page
# and the wall display list what they are given, so this is a memory bound on one
# reply rather than a page size. The provider skill declares the same ceiling.
LIBRARY_LIMIT_MAX = 500


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

    The queue belongs to the independent queue service when one is running (see
    THE QUEUE SERVICE above), and to the music provider skill otherwise. Either
    way the caller speaks the same vocabulary: a read is a read, and the speaker
    hand-off for a play or a skip is performed here, exactly as POST
    /api/music/play does it.
    """
    # A service whose address was published for us is only used while it is
    # answering: a queue service that was stopped or uninstalled leaves the
    # console with the provider's own queue rather than an API that says no.
    if _queue_service_on() and _queue_service_alive():
        return _music_queue_service(handler)
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
    elif op == "add_range":
        # The playlist the song was listed in, and the song the user chose. With
        # no video_id the whole playlist is queued, which is what the button on
        # a playlist header means.
        playlist_id = _text(body.get("playlist_id"), "playlist_id", 100)
        if not playlist_id:
            return _fail(handler, MusicApiError(
                "Provide playlist_id=<the playlist the song came from>.", 400, "validation_error"))
        arguments["playlist_id"] = playlist_id
        video_id = _text(body.get("video_id"), "video_id", 64)
        if video_id:
            arguments["video_id"] = video_id
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
    panel_stream.notify("queue", action=op)
    return _respond(handler, payload)


def _provider_tracks(result: dict) -> list[dict]:
    """The tracks a provider search or library read hands over."""
    for key in ("tracks", "results", "items"):
        value = result.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _queue_track_fields(track: dict) -> dict:
    """The queue's own fields, lifted off a provider track record."""
    return {key: track.get(key) for key in queue_engine.TRACK_FIELDS if track.get(key) is not None}


def _queue_track_wanted(body: dict) -> dict:
    """One track to queue: named by id, or resolved through the provider search.

    A page that already knows the id (it came from a playlist listing) costs no
    provider call at all: the queue stores what it was given and the background
    fills in anything missing.
    """
    video_id = _text(body.get("video_id"), "video_id", 64)
    if video_id:
        wanted = {"video_id": video_id}
        wanted.update(_queue_track_fields({
            key: body.get(key) for key in ("title", "artist", "album", "duration_sec", "artwork_url")
        }))
        return wanted
    query = _text(body.get("query"), "query", 200)
    if not query:
        raise MusicApiError("Provide query=<search text> or video_id=<id>.", 400, "validation_error")
    _skill, result = _provider_call("search", {"query": query, "limit": 1})
    tracks = _provider_tracks(result)
    if not tracks:
        raise MusicApiError(f"No track matched “{query}”.", 404, "no_match")
    return _queue_track_fields(tracks[0])


def _queue_write_batches(tracks: list[dict], max_batch: int) -> list[list[dict]]:
    """A long playlist is written in the provider's own batch size.

    Not one call per song, and not one unbounded call either: the ceiling is
    the provider's, and a list longer than it is simply read in a few writes.
    """
    size = max(1, int(max_batch or 200))
    return [tracks[start:start + size] for start in range(0, len(tracks), size)]


def _queue_service_add_all(tracks: list[dict], playlist_id: str = "",
                          mode: str = "append") -> dict:
    """Queue a whole list, in as few writes as the provider's ceiling allows."""
    limits = _push_queue_limits()
    batches = _queue_write_batches(tracks, limits.get("max_batch", 200))
    reply: dict = {}
    accepted = refused = total = 0
    for index, batch in enumerate(batches):
        reply = _queue_service_add(batch, playlist_id=playlist_id,
                                   mode=mode if index == 0 else "append")
        accepted += int(reply.get("accepted") or 0)
        refused += int(reply.get("refused") or 0)
        total = int(reply.get("total") or total)
        if refused and index + 1 < len(batches):
            # The queue is full: the rest of the list would be refused too, so
            # the writes stop rather than being sent to be turned away.
            break
    return {"accepted": accepted, "refused": refused, "total": total,
            "revision": reply.get("revision"), "mode": mode,
            "max_tracks": limits.get("max_tracks")}


def _queue_service_play_from(result: dict, skill_id: str) -> dict | None:
    """Hand the service's current queued track to the speaker.

    The queue service decides *which* track plays; the provider skill is still
    what turns a track into a stream URL and what receives the hand-off, so a
    track started from the queue is played exactly like one played from search.
    """
    track = result.get("track") if isinstance(result.get("track"), dict) else None
    if not track or not track.get("video_id"):
        return None
    _provider_skill_id, play_result = _provider_call("play", {"video_id": track["video_id"]})
    _start_playback(play_result, skill_id, queue={"index": result.get("index")})
    return play_result


def _advance_through_queue_service() -> None:
    """The queue service's version of "the track ended, play the next one"."""
    try:
        result = _queue_service_op({"op": "next", "auto": True})
    except MusicApiError:
        return
    if not result.get("ok"):
        return
    skill_id = ""
    try:
        skill_id, _manifest, _provider = _provider_skill()
    except MusicApiError:
        return
    _queue_service_play_from(result, skill_id)


@_guarded
def _music_queue_service(handler, _match=None) -> bool:
    """The queue routes, answered by the independent queue service.

    A read is a window plus a revision, so a page that already holds the current
    revision is told "unchanged" and sends nothing back. A write is one call —
    including ``add_many``, which is how a whole playlist changes hands: the
    page posts the songs it already listed, and the service stores them and
    warms them ahead of the cursor at the provider's own rate.
    """
    if handler.command == "GET":
        query = _query(handler)
        offset = _int(query.get("from"), "from", 0, 1_000_000) or 0
        limit = _int(query.get("limit"), "limit", 1, 2000) or _push_queue_limits().get("window", 50)
        revision = _int(query.get("revision"), "revision", 0, 2**31 - 1)
        try:
            reply = _queue_service_state(offset, limit, revision)
        except MusicApiError as exc:
            return _fail(handler, exc)
        payload = _queue_service_payload(reply)
        if not payload["message"]:
            payload["message"] = (
                f"{payload['total']} track(s) queued." if payload["total"] else "The queue is empty."
            )
        return _respond(handler, payload)

    body = _body(handler)
    op = (_text(body.get("action") or body.get("op"), "action", 24) or "list").lower()
    if op not in QUEUE_OPS:
        return _fail(handler, MusicApiError(
            f"action must be one of: {', '.join(sorted(QUEUE_OPS))}.", 400, "validation_error"))
    try:
        if op == "add":
            track = _queue_track_wanted(body)
            mode = "play_next" if body.get("mode") == "play_next" else "append"
            reply = _queue_service_add([track], mode=mode)
        elif op == "add_many":
            tracks = body.get("tracks")
            if not isinstance(tracks, list) or not tracks:
                return _fail(handler, MusicApiError(
                    "Provide tracks=[{video_id, title, ...}, ...].", 400, "validation_error"))
            if len(tracks) > 20000:
                return _fail(handler, MusicApiError(
                    "That is more than one call may queue.", 413, "too_many"))
            mode = body.get("mode") if body.get("mode") in {"append", "play_next", "replace"} else "append"
            reply = _queue_service_add_all(
                [item for item in tracks if isinstance(item, dict)],
                playlist_id=_text(body.get("playlist_id"), "playlist_id", 100), mode=mode)
            reply["message"] = _queue_service_message(reply)
        elif op == "add_range":
            playlist_id = _text(body.get("playlist_id"), "playlist_id", 100)
            if not playlist_id:
                return _fail(handler, MusicApiError(
                    "Provide playlist_id=<the playlist the song came from>.", 400, "validation_error"))
            _skill, listing = _provider_call("library", {
                "kind": "playlist", "playlist_id": playlist_id, "limit": LIBRARY_LIMIT_MAX})
            songs = _provider_tracks(listing)
            if not songs:
                return _fail(handler, MusicApiError("That playlist has no songs to queue.",
                                                    404, "no_match"))
            anchor = _text(body.get("video_id"), "video_id", 64)
            start = 0
            if anchor:
                found = next((position for position, song in enumerate(songs)
                              if song.get("video_id") == anchor), None)
                if found is None:
                    return _fail(handler, MusicApiError(
                        "That song is not in this playlist any more.", 404, "no_match"))
                start = found
            reply = _queue_service_add_all([_queue_track_fields(song) for song in songs[start:]],
                                           playlist_id=playlist_id)
            reply["message"] = _queue_service_message(reply)
        elif op in {"remove", "move", "front", "next_up", "play", "next", "clear", "shuffle", "repeat"}:
            arguments: dict = {"op": op}
            index = _int(body.get("index"), "index", 0, QUEUE_INDEX_MAX)
            to = _int(body.get("to"), "to", 0, QUEUE_INDEX_MAX)
            if op in {"remove", "front", "next_up"} and index is None and not body.get("query") and not body.get("video_id"):
                return _fail(handler, MusicApiError("Provide index=<queue position>.", 400, "validation_error"))
            if op == "move" and (index is None or to is None):
                return _fail(handler, MusicApiError("Provide index=<from> and to=<to>.", 400, "validation_error"))
            if op == "play" and index is None:
                return _fail(handler, MusicApiError("Provide index=<queue position>.", 400, "validation_error"))
            if index is not None:
                arguments["index"] = index
            if to is not None and op == "move":
                arguments["to"] = to
            if op == "shuffle" and _flag(body.get("on"), "on") is not None:
                arguments["on"] = _flag(body.get("on"), "on")
            if op == "repeat":
                mode = _text(body.get("mode"), "mode", 8).lower()
                if mode and mode not in REPEAT_MODES:
                    return _fail(handler, MusicApiError(
                        f"mode must be one of: {', '.join(REPEAT_MODES)}.", 400, "validation_error"))
                if mode:
                    arguments["mode"] = mode
            if op == "next" and body.get("auto"):
                arguments["auto"] = True
            reply = _queue_service_op(arguments)
        else:  # ``list`` as a POST, which is the same read as GET
            reply = _queue_service_state(0, _push_queue_limits().get("window", 50), None)
    except MusicApiError as exc:
        return _fail(handler, exc)

    payload = _queue_service_payload(reply)
    if op in {"add", "add_many", "add_range"}:
        # A write says what it took: a caller that sent four hundred songs is
        # owed a number, not just the queue it produced.
        payload.update({key: reply[key] for key in ("accepted", "refused", "unqueable", "mode")
                        if key in reply})
    if op == "clear":
        payload["removed"] = int(reply.get("removed") or 0)
        payload["message"] = reply.get("message") or payload["message"]
    if op in {"play", "next"}:
        if not reply.get("ok"):
            state = {}
            try:
                state = _queue_service_state(0, _push_queue_limits().get("window", 50), None)
            except MusicApiError:
                pass
            payload.update(_queue_service_payload(state) if state else {})
            payload["ok"] = False
            payload["code"] = "queue_empty"
            payload["error"] = reply.get("message") or "The queue has no track to play."
            return _respond(handler, payload, 409)
        try:
            skill_id, _manifest, _provider = _provider_skill()
            played = _queue_service_play_from(reply, skill_id)
        except MusicApiError as exc:
            return _fail(handler, exc)
        if played is None:
            return _fail(handler, MusicApiError("The queue has no track to play.", 409, "queue_empty"))
        payload["track"] = _now_playing_payload()["track"]
        payload["message"] = payload["message"] or f"Playing “{played.get('title') or ''}”."
    elif not payload["message"]:
        total = payload["total"]
        payload["message"] = f"{total} track(s) queued." if total else "The queue is empty."
    panel_stream.notify("queue", action=op)
    return _respond(handler, payload)


@_guarded
def music_queue_resolve(handler, _match=None) -> bool:
    """Resolve one track's metadata without queueing or playing it.

    This is what the independent queue service calls back for, once per track
    whose metadata never arrived (a queue built from bare video ids): the
    service owns the rate limit and the cache, and this end knows how to ask the
    installed provider skill.
    """
    video_id = _text(_body(handler).get("video_id"), "video_id", 64)
    if not video_id:
        return _fail(handler, MusicApiError("Provide video_id=<id>.", 400, "validation_error"))
    try:
        _skill, result = _provider_call("queue", {"op": "resolve", "video_id": video_id})
    except MusicApiError as exc:
        return _fail(handler, exc)
    # The provider's result carries the track's own fields at the top level (its
    # result schema is closed), so the resolved track is read from there.
    track = _queue_track_fields(result)
    if not track.get("video_id") or not track.get("title"):
        return _respond(handler, {"ok": False, "video_id": video_id,
                                  "error": "That track is not known to the provider."}, 404)
    return _respond(handler, {"ok": True, "video_id": video_id, "track": track})


@_guarded
def music_queue_add_range(handler, _match=None) -> bool:
    """Add a playlist's songs from one row to the end to the queue.

    The page's button is a queue write, so it belongs on the queue route: this
    is the one place that knows the body's shape, and it forwards the same
    ``add_range`` op the provider skill declares. Kept as its own function so the
    route table (and the panel's copy of it) names it like every other handler.
    """
    return music_queue(handler, _match)


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
    named = {key: value for key, value in (
        ("video_id", _text(query.get("video_id"), "video_id", 64)),
        ("title", _text(query.get("title"), "title", 200)),
        ("artist", _text(query.get("artist"), "artist", 200)),
        ("album", _text(query.get("album"), "album", 200)),
    ) if value}
    # A caller that names a track is answered for exactly that track. Merging the
    # recorded track's own name over a named id used to make the provider look up
    # the *last* song it played and hand its words back as the words of the named
    # one - wrong lyrics, reported as a success, which is worse than none. The
    # recorded track fills in only when nothing is named at all ("show me the
    # lyrics" while music plays).
    arguments = dict(named) if named else _lyrics_arguments(track)
    duration = _int(query.get("duration_sec"), "duration_sec", 1, 60 * 60)
    if duration:
        arguments["duration_sec"] = duration
    if named and not duration:
        # Naming the track that is playing by its title is the same request as
        # naming nothing, and the console's own clock knows how long it runs: the
        # provider matches best (and an exact LRCLIB lookup needs) a duration.
        # A title that is not the recorded track's inherits nothing.
        recorded_title = _normalized_title(track.get("title"))
        if recorded_title and named.get("title") and _normalized_title(named["title"]) == recorded_title:
            recorded_duration = track.get("duration_sec")
            if isinstance(recorded_duration, (int, float)) and recorded_duration > 0:
                arguments["duration_sec"] = int(recorded_duration)
            for key in ("video_id", "artist", "album"):
                if not arguments.get(key) and track.get(key):
                    arguments[key] = track[key]
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
