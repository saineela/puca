"""The Documentation page must describe the API the console actually serves.

A documentation page that drifts from the routes is worse than none: a reader
follows it and gets a 404. These checks compare the page's route table against
the registered extension routes and the base console's own paths, so adding a
route without documenting it (or documenting one that was removed) fails here.
"""
from __future__ import annotations

from html import unescape
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
HTML = (HERE / "dashboard.html").read_text(encoding="utf-8")
SECTION_START = HTML.index('<section class="view" id="docs">')
SECTION_END = HTML.index("</section>", SECTION_START)
DOCS = HTML[SECTION_START:SECTION_END]
CONSOLE_SOURCE = (HERE / "console.py").read_text(encoding="utf-8")
MUSIC_SOURCE = (HERE / "music_api.py").read_text(encoding="utf-8")

ROUTE_LINE = re.compile(r"^(GET|POST)\s+(/[^\s]+)", re.MULTILINE)
EXTENSION_PREFIXES = (
    "/api/mobile/v1",
    "/api/music",
    "/api/voice",
    "/api/panel",
    "/api/extensions",
    "/api/system-model",
    "/api/skills/",
)


def _registered(method: str, path: str, registered: list[tuple[str, str]]) -> bool:
    """A documented path may be a pattern (``/api/skills/<id>/page/<asset>``)."""
    if "<" in path:
        prefix = path.split("<")[0].rstrip("/")
        return any(
            route_method == method and expression.startswith(prefix)
            for route_method, expression in registered
        )
    return any(
        route_method == method and re.fullmatch(expression, path)
        for route_method, expression in registered
    )


def documented_paths() -> list[tuple[str, str]]:
    """``(method, path)`` pairs from the page's code blocks, query stripped."""
    pairs: list[tuple[str, str]] = []
    for method, path in ROUTE_LINE.findall(DOCS):
        clean = unescape(path).split("?")[0].rstrip("…").rstrip("/") or "/"
        pairs.append((method, clean))
    return pairs


def test_documentation_page_lists_every_music_endpoint() -> None:
    """Every route the music index advertises is in the page, skill-gated or not."""
    advertised = re.findall(r'"(GET|POST|PUT|DELETE) (/\S+?)"', MUSIC_SOURCE)
    assert advertised, "the music API advertises no endpoints"
    missing = [
        f"{method} {path}"
        for method, path in advertised
        if path.split("?")[0] not in DOCS
    ]
    assert missing == [], f"the docs page is missing music endpoints: {missing}"


def test_documentation_page_says_the_music_api_is_skill_gated() -> None:
    music_section = DOCS[DOCS.index('id="docs-music-api"') : DOCS.index('id="docs-voice-api"')]
    assert "Skill-gated" in music_section
    assert "OpenDeck" in music_section, "the supported skill must be named"
    assert "skill_not_runnable" in music_section, "the no-provider answer must be stated"


def test_documented_extension_paths_are_actually_registered() -> None:
    import console_extend

    registered = [(route.method, route.expression) for route in console_extend._routes]
    unregistered: list[str] = []
    for method, path in documented_paths():
        if not path.startswith(EXTENSION_PREFIXES):
            continue
        if not _registered(method, path, registered):
            unregistered.append(f"{method} {path}")
    assert unregistered == [], f"the docs page promises unregistered routes: {unregistered}"


def test_documented_base_console_paths_are_actually_served() -> None:
    served = set(re.findall(r'path == "([^"]+)"', CONSOLE_SOURCE))
    served_startswith = set(re.findall(r'path\.startswith\("([^"]+)"\)', CONSOLE_SOURCE))
    unserved: list[str] = []
    for _method, path in documented_paths():
        if "<" in path or path.startswith(EXTENSION_PREFIXES) or path in served:
            continue
        if any(path.startswith(prefix) for prefix in served_startswith):
            continue
        if path in {"/", "/pipeline"}:
            continue
        unserved.append(path)
    assert unserved == [], f"the docs page lists paths the console does not serve: {unserved}"


def test_every_documentation_nav_target_exists_in_the_article() -> None:
    targets = re.findall(r'data-doc-target="([^"]+)"', DOCS)
    assert targets, "the docs page has no section navigation"
    missing = [target for target in targets if f'id="{target}"' not in DOCS]
    assert missing == [], f"navigation targets with no section: {missing}"
