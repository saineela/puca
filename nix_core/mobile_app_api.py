"""Small Mobile App API surface currently implemented by the console extension.

Only randomized home copy is live here; the future mobile facade is documented
in MOBILE_APP_API.md and is not an implemented mobile application or API suite.
"""
from __future__ import annotations

import json
import re
import secrets
from itertools import product
from urllib.parse import parse_qs, urlparse


CATEGORY = "Mobile App"
API_VERSION = "1"

_HEADLINES = (
    "What would you like to work through?",
    "What is on your mind today?",
    "Where would you like to begin?",
    "What would feel helpful right now?",
    "What would you like to make space for?",
    "What would you like to sort out?",
    "What can we explore together?",
    "What would you like to focus on?",
    "How can we make today a little easier?",
    "What would you like to take one step at a time?",
    "What is weighing on you right now?",
    "What would you like to think through?",
    "What can I help you carry today?",
    "What deserves a little attention today?",
    "What would feel like progress to you?",
    "Where is your head at today?",
    "What would you like to put into words?",
    "What can we untangle together?",
    "What would make today feel lighter?",
    "What has been on your heart lately?",
)

_SUBTITLES = (
    "No rush. We can talk, make a small plan, or simply take a breath together.",
    "We can talk it through, find a next step, or save a thought for later.",
    "Take your time. We can start wherever you are.",
    "A conversation, a plan, or a quiet place to begin—your choice.",
    "We can make room for the big picture or just the next small thing.",
    "Bring a question, a feeling, or something you would like to remember.",
    "There is no perfect place to start. Pick whatever feels useful.",
    "We can sort through today together, one piece at a time.",
    "Start with what matters to you; we will take it at your pace.",
    "Whether it is practical or personal, we can begin with one thought.",
    "You can ask, plan, reflect, or simply put something into words.",
    "A little clarity can start with a simple conversation.",
    "Tell me what is on your mind, and we can work from there.",
    "We can look for a gentle next step without rushing the rest.",
    "Let us begin with whatever you would most like help carrying today.",
    "Whatever you bring, we can take it slowly and kindly.",
    "One thought, one step, one small conversation at a time.",
    "We can sit with it, sort it, or set it down for later.",
    "Big things and small things are both welcome here.",
    "Start anywhere. We will find the thread together.",
    "There is space here for whatever today has been.",
    "We can be practical, reflective, or just quiet together.",
    "Nothing is too small or too tangled to start with.",
    "Bring the day as it is; we will work from there.",
    "Your pace, your topic, your space.",
)

_SECTION_TITLES = (
    "Choose a gentle beginning",
    "A few ways to get started",
    "Start wherever it feels useful",
    "Pick a small first step",
    "Some possible starting points",
    "What would help right now?",
    "Where shall we begin?",
    "Three gentle options",
)

_CHECK_INS = (
    ("Check in with me", "A little space to be heard", "Can you check in with me? I want to talk through how I’m feeling."),
    ("Talk through today", "Put the day into words", "I’d like to talk through how today has been for me."),
    ("Name what is on my mind", "Sort through a few thoughts", "Help me sort through what has been on my mind."),
    ("Pause for a moment", "Take this at an easy pace", "Can we pause for a moment and talk about what I need right now?"),
    ("Help me reflect", "Look back without judgment", "Help me reflect on something that happened today."),
    ("Talk it out", "Start with one thing", "I want to talk something through, one piece at a time."),
    ("Make room to think", "A calm place to begin", "Can you help me make a little space to think?"),
    ("Sort out a feeling", "Find words for it", "Help me put a feeling I’m having into words."),
    ("Find some perspective", "Look at it another way", "Help me look at something on my mind from another perspective."),
    ("Share a small win", "Notice what went well", "I’d like to share something that went well today."),
    ("Work through a worry", "Take one concern at a time", "Help me think through a worry without rushing to solve everything."),
    ("Have a quiet check-in", "Begin with how things are", "Can we have a simple check-in about how I’m doing?"),
    ("Untangle a thought", "Make it easier to see", "Help me untangle a thought that feels a little complicated."),
    ("Talk about what matters", "Start with what feels important", "I want to talk about something that matters to me."),
    ("Take a breath and begin", "One moment at a time", "Help me slow down and decide what would feel useful next."),
)

_NEXT_STEPS = (
    ("Help me reset", "Find a softer next step", "Help me reset and make the rest of today feel easier."),
    ("Make a small plan", "Keep it simple and doable", "Help me make a small, realistic plan for today."),
    ("Choose what comes next", "Focus on one next action", "Help me decide on one useful next step."),
    ("Break this into pieces", "Make a big task feel smaller", "Help me break something I need to do into smaller steps."),
    ("Plan the rest of today", "Leave a little room to breathe", "Help me plan the rest of today at a manageable pace."),
    ("Get organized", "Bring a few loose ends together", "Help me organize a few things I need to take care of."),
    ("Start a task", "Find an easy first move", "Help me figure out the easiest first step for a task."),
    ("Think through a choice", "Compare the options", "Help me think through a decision and the options I have."),
    ("Make a short list", "Keep the priorities clear", "Help me make a short list of what matters most today."),
    ("Prepare for something", "Get ready one step at a time", "Help me prepare for something coming up."),
    ("Set a gentle priority", "Choose what can wait", "Help me choose what to focus on and what can wait."),
    ("Find a practical next step", "Turn a concern into an action", "Help me find one practical next step for something on my mind."),
    ("Make the week feel clearer", "Look at what is ahead", "Help me look at the week ahead and make a simple plan."),
    ("Sort out a loose end", "Give one thing your attention", "Help me decide how to handle a loose end."),
    ("Keep the plan manageable", "Make space for breaks", "Help me make a plan that leaves room for breaks."),
)

_REMEMBER = (
    ("Save a thought", "Keep what matters close", "Remember that I want to make more time for the people I care about."),
    ("Remember a preference", "Keep a useful detail in mind", "Remember a preference I’d like you to keep in mind."),
    ("Save something important", "Hold onto a detail for later", "I’d like you to remember something important to me."),
    ("Keep track of a goal", "A small intention for later", "Remember a goal I’m working toward."),
    ("Remember what helps", "Keep a helpful reminder", "Remember something that tends to help me when I’m having a hard day."),
    ("Save a plan", "Keep the next step handy", "Remember the plan I’m making so I can come back to it."),
    ("Keep a personal note", "Store a detail I choose", "I want to share a personal note for you to remember."),
    ("Remember something I enjoy", "Keep a favorite close", "Remember something I enjoy doing in my free time."),
    ("Save a small intention", "Make room for what matters", "Remember an intention I want to come back to."),
    ("Keep a detail for next time", "Useful context for later", "Please remember this detail for a future conversation."),
    ("Remember a boundary", "Keep my preference in mind", "Remember a boundary or preference I want you to respect."),
    ("Save what I learned", "Keep a helpful takeaway", "Remember a useful thing I learned today."),
    ("Keep a relationship note", "Remember what I share", "Remember something I’d like you to know about someone important to me."),
    ("Save a routine", "Keep a helpful rhythm in mind", "Remember a routine that works well for me."),
    ("Hold onto this for me", "Save a thought in my words", "I’d like to tell you something and have you remember it."),
)

_ICON_SETS = ("♡", "☼", "⌁", "✧", "◷", "＋", "◇", "☁", "↗", "○", "▤", "✦", "⌂", "☷", "·")


def home_copy_catalog() -> tuple[dict, ...]:
    """Build exactly 150 distinct copy bundles from 10 × 15 combinations."""
    variants: list[dict] = []
    for index, (headline, subtitle) in enumerate(product(_HEADLINES, _SUBTITLES)):
        cards = []
        for slot, choices in enumerate((_CHECK_INS, _NEXT_STEPS, _REMEMBER)):
            title, description, prompt = choices[(index * (slot + 1) + slot * 5) % len(choices)]
            cards.append({"icon": _ICON_SETS[(index + slot * 5) % len(_ICON_SETS)], "title": title, "description": description, "prompt": prompt})
        variants.append({
            "copy_id": f"home-{index + 1:03d}",
            "headline": headline,
            "subtitle": subtitle,
            "section_title": _SECTION_TITLES[index % len(_SECTION_TITLES)],
            "suggestions": cards,
        })
    return tuple(variants)


HOME_COPY_VARIANTS = home_copy_catalog()
# 20 headlines x 25 subtitles = 500 premade home-screen combinations.
assert len(HOME_COPY_VARIANTS) == 500, len(HOME_COPY_VARIANTS)
assert len({(item["headline"], item["subtitle"]) for item in HOME_COPY_VARIANTS}) == 500
assert len({item["copy_id"] for item in HOME_COPY_VARIANTS}) == 500


def _json_no_store(handler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store, max-age=0")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def api_index(handler, _match) -> bool:
    _json_no_store(handler, {
        "ok": True,
        "category": CATEGORY,
        "api_version": API_VERSION,
        "live_endpoints": [
            "GET /api/mobile/v1",
            "GET /api/mobile/v1/home-copy",
            "GET /api/mobile/v1/music",
            "GET /api/mobile/v1/music/now-playing",
            "POST /api/mobile/v1/music/control",
            "GET /api/mobile/v1/music/search",
            "POST /api/mobile/v1/music/play",
            "GET /api/mobile/v1/music/library",
            "GET|POST /api/mobile/v1/music/queue",
            "POST /api/mobile/v1/music/save",
            "GET /api/mobile/v1/music/lyrics",
            "POST /api/mobile/v1/devices/report",
            "POST /api/mobile/v1/devices/discover",
            "POST /api/mobile/v1/devices/register",
            "POST /api/mobile/v1/devices/unregister",
            "GET /api/mobile/v1/devices",
            "GET /api/mobile/v1/devices/list",
            "GET /api/mobile/v1/devices/{mac}",
            "POST /api/mobile/v1/devices/mock/seed",
        ],
        "documentation": "nix_core/MOBILE_APP_API.md",
        "mobile_application_included": False,
        "status": "home-copy prototype + official device registration",
    })
    return True


# ----------------------------------------------------------------------
# Official device registration (NIX Home BLE pairing counterpart)
# ----------------------------------------------------------------------

import mobile_device_registry  # noqa: E402  (module-level, imported lazily-safe)


def _registry() -> "mobile_device_registry.DeviceRegistry":
    global _DEVICE_REGISTRY
    if _DEVICE_REGISTRY is None:
        _DEVICE_REGISTRY = mobile_device_registry.DeviceRegistry()
    return _DEVICE_REGISTRY


_DEVICE_REGISTRY: "mobile_device_registry.DeviceRegistry | None" = None


def _read_body(handler) -> dict:
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


def _device_error(handler, exc: Exception, status: int = 400) -> None:
    _json_no_store(handler, {"ok": False, "category": CATEGORY, "error": str(exc)}, status)


def devices_index(handler, _match) -> bool:
    _json_no_store(handler, {
        "ok": True,
        "category": CATEGORY,
        "api_version": API_VERSION,
        "resource": "devices",
        "flow": [
            "1. app scans BLE; names containing 'Nix Dev' are candidates",
            "2. POST /devices/report per advertisement (dedup by MAC)",
            "3. POST /devices/discover -> claim token",
            "4. POST /devices/register with claim token + device secret -> official",
            "5. GET /devices or GET /devices/{mac} for status",
            "6. POST /devices/unregister to retire",
        ],
        "mock_testing": (
            "POST /devices/report with \"mock\": true and POST "
            "/devices/mock/seed exercise the whole lifecycle with fake "
            "addresses and no Bluetooth hardware."
        ),
    })
    return True


def devices_report(handler, _match) -> bool:
    body = _read_body(handler)
    try:
        device = _registry().report(
            mac=body.get("mac"),
            advertised_name=body.get("advertised_name"),
            rssi=body.get("rssi", -60),
            mock=bool(body.get("mock", False)),
        )
    except mobile_device_registry.DeviceRegistryError as exc:
        _device_error(handler, exc)
        return True
    _json_no_store(handler, {"ok": True, "category": CATEGORY, "device": device})
    return True


def devices_discover(handler, _match) -> bool:
    body = _read_body(handler)
    try:
        result = _registry().discover(mac=body.get("mac"))
    except mobile_device_registry.DeviceRegistryError as exc:
        _device_error(handler, exc)
        return True
    _json_no_store(handler, {"ok": True, "category": CATEGORY, **result})
    return True


def devices_register(handler, _match) -> bool:
    body = _read_body(handler)
    try:
        result = _registry().register(
            mac=body.get("mac"),
            claim_token=body.get("claim_token"),
            device_secret=body.get("device_secret"),
            label=str(body.get("label") or ""),
            kind=body.get("kind", "other"),
            mock=bool(body.get("mock", False)),
        )
    except mobile_device_registry.DeviceRegistryError as exc:
        _device_error(handler, exc)
        return True
    _json_no_store(handler, {"ok": True, "category": CATEGORY, **result})
    return True


def devices_unregister(handler, _match) -> bool:
    body = _read_body(handler)
    try:
        result = _registry().unregister(mac=body.get("mac"))
    except mobile_device_registry.DeviceRegistryError as exc:
        _device_error(handler, exc)
        return True
    _json_no_store(handler, {"ok": True, "category": CATEGORY, **result})
    return True


def devices_list(handler, _match) -> bool:
    _json_no_store(handler, {"category": CATEGORY, **_registry().status()})
    return True


def devices_get(handler, match) -> bool:
    try:
        result = _registry().status(match.group(1))
    except mobile_device_registry.DeviceRegistryError as exc:
        _device_error(handler, exc, 404)
        return True
    _json_no_store(handler, {"category": CATEGORY, **result})
    return True


def devices_mock_seed(handler, _match) -> bool:
    """Seed one fake 'Nix Dev' advertisement for mock testing (no hardware)."""
    body = _read_body(handler)
    suffix = re.sub(r"[^0-9A-F]", "", str(body.get("suffix") or "4A2F").upper())[:4]
    suffix = (suffix + "4A2F")[:4]
    mac = f"04:52:C7:{suffix[0:2]}:{suffix[2:4]}:{secrets.token_hex(1).upper()}"
    try:
        device = _registry().report(
            mac=mac,
            advertised_name=f"Nix Dev {suffix}",
            rssi=body.get("rssi", -55),
            mock=True,
        )
        secret = _registry()._devices[device["mac"]]["device_secret"]  # mock only
    except mobile_device_registry.DeviceRegistryError as exc:
        _device_error(handler, exc)
        return True
    _json_no_store(handler, {
        "ok": True,
        "category": CATEGORY,
        "mock": True,
        "device": device,
        "device_secret": secret,
        "note": "Mock test device with its secret; never exists as real hardware.",
    })
    return True


def home_copy(handler, _match) -> bool:
    options = parse_qs(urlparse(handler.path).query).get("catalog", [])
    if options and options != ["1"]:
        _json_no_store(handler, {"ok": False, "category": CATEGORY, "error": "catalog must be 1 when supplied"}, 400)
        return True
    if options == ["1"]:
        _json_no_store(handler, {
            "ok": True,
            "category": CATEGORY,
            "api_version": API_VERSION,
            "catalog_size": len(HOME_COPY_VARIANTS),
            "selection": "caller-selects-by-copy_id",
            "variants": HOME_COPY_VARIANTS,
        })
        return True
    selected = secrets.choice(HOME_COPY_VARIANTS)
    _json_no_store(handler, {
        "ok": True,
        "category": CATEGORY,
        "api_version": API_VERSION,
        "catalog_size": len(HOME_COPY_VARIANTS),
        "selection": "random-per-request",
        "data": selected,
    })
    return True
