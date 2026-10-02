"""Small Mobile App API surface currently implemented by the console extension.

Only randomized home copy is live here; the future mobile facade is documented
in MOBILE_APP_API.md and is not an implemented mobile application or API suite.
"""
from __future__ import annotations

import json
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
)

_SECTION_TITLES = (
    "Choose a gentle beginning",
    "A few ways to get started",
    "Start wherever it feels useful",
    "Pick a small first step",
    "Some possible starting points",
    "What would help right now?",
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
assert len(HOME_COPY_VARIANTS) == 150
assert len({(item["headline"], item["subtitle"]) for item in HOME_COPY_VARIANTS}) == 150


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
        "live_endpoints": ["GET /api/mobile/v1", "GET /api/mobile/v1/home-copy"],
        "documentation": "nix_core/MOBILE_APP_API.md",
        "mobile_application_included": False,
        "status": "home-copy-prototype; remaining mobile facade is documented, not implemented",
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
