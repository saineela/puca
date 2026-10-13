"""Tests for the PUCA Panel: the wall display's own listener and its question.

The panel is a second HTTP surface over the same music API, so what it needs
pinned is: the page comes from the installed provider's package and nothing else
does, the music routes behave exactly as they do on the console, the theme
follows the console's clock, and a play request Nix cannot decide on is asked as
a question instead of being played as a guess.
"""
import json
from datetime import datetime, timezone

import pytest

import brain
import console
import music_api
import puca_panel
from test_music_api import FakeRuntime, PROVIDER_MANIFEST, install, music_env, read_json, read_raw  # noqa: E402


PANEL_FILES = ["PUCA-Panel/panel.html", "PUCA-Panel/panel.css", "PUCA-Panel/panel.js"]

# The provider manifest the music tests use, plus the panel page it carries and
# the fields an unclear play answers with (the real manifest declares both).
PROVIDER = {
    **PROVIDER_MANIFEST,
    "version": "1.2.0",
    "files": [*PROVIDER_MANIFEST["files"], *PANEL_FILES],
    "tools": [
        {
            **PROVIDER_MANIFEST["tools"][0],
            "result_schema": {
                **PROVIDER_MANIFEST["tools"][0]["result_schema"],
                "properties": {
                    **PROVIDER_MANIFEST["tools"][0]["result_schema"]["properties"],
                    "ambiguous": {"type": "boolean"},
                    "query": {"type": "string", "maxLength": 200},
                    "options": {
                        "type": "array",
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "properties": {
                                "video_id": {"type": "string", "maxLength": 64},
                                "title": {"type": "string", "maxLength": 300},
                                "artist": {"type": "string", "maxLength": 200},
                                "duration_sec": {"type": "integer", "minimum": 0, "maximum": 100000},
                                "artwork_url": {"type": "string", "maxLength": 600},
                            },
                            "required": ["video_id", "title", "artist", "duration_sec", "artwork_url"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [*PROVIDER_MANIFEST["tools"][0]["result_schema"]["required"], "ambiguous", "query", "options"],
            },
        }
    ],
}


def play_result(*, ambiguous=False, options=None, video_id="vid-1", title="A Song"):
    """What the provider answers a play with, in its declared result fields."""
    filled = {
        "ok": True,
        "action": "play",
        "message": "Playing “A Song”." if not ambiguous else "2 tracks match “that song”: which one did you mean?",
        "connected": True,
        "instance": "http://opendeck.test",
        "user": "Ada",
        "video_id": video_id if not ambiguous else "",
        "title": title if not ambiguous else "",
        "artist": "Someone",
        "album": "",
        "duration_sec": 200,
        "artwork_url": "",
        "mime_type": "audio/mp4",
        "play_url": (video_id if not ambiguous else ""),
        "tracks": [],
        "playlists": [],
        "queue": [],
        "queue_index": -1,
        "count": 1,
        "lyrics": [],
        "lyrics_plain": "",
        "lyrics_timing": "",
        "lyrics_source": "",
        "shuffle": False,
        "repeat": "off",
        "ambiguous": ambiguous,
        "query": "that song" if ambiguous else "",
        "options": options or [],
    }
    return filled


def option(video_id, title):
    return {
        "video_id": video_id,
        "title": title,
        "artist": "Someone",
        "duration_sec": 200,
        "artwork_url": "",
    }


@pytest.fixture(autouse=True)
def clean_question():
    """No question may leak from one test into the next."""
    puca_panel.clear_question()
    yield
    puca_panel.clear_question()


@pytest.fixture
def panel(music_env, monkeypatch):
    """A running panel listener on an ephemeral port, with a provider installed."""
    runtime = install(monkeypatch, music_env, [PROVIDER], FakeRuntime(
        results={"opendeck_music": play_result()},
    ))
    server = puca_panel.create_server("127.0.0.1", 0)
    from threading import Thread

    Thread(target=server.serve_forever, daemon=True).start()
    yield server, runtime
    server.shutdown()
    server.server_close()


# ── day and night ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "hour,expected",
    [(5, "dark"), (6, "light"), (12, "light"), (21, "light"), (22, "dark"), (23, "dark"), (0, "dark")],
)
def test_the_theme_follows_the_console_clock(monkeypatch, hour, expected):
    monkeypatch.setattr(console, "TIMEZONE", "UTC")
    moment = datetime(2026, 5, 1, hour, 30, tzinfo=timezone.utc).timestamp()
    assert puca_panel.daypart(moment) == expected


def test_the_theme_reports_how_the_display_should_paint(monkeypatch):
    monkeypatch.setattr(console, "TIMEZONE", "UTC")
    reported = puca_panel.theme()
    assert reported["mode"] in {"dark", "light"}
    assert (reported["dark_from_hour"], reported["dark_until_hour"]) == (22, 6)
    assert reported["timezone"] == "UTC"
    assert len(reported["local_time"]) == 5


def test_an_unknown_timezone_falls_back_instead_of_breaking_the_display(monkeypatch):
    monkeypatch.setattr(console, "TIMEZONE", "Not/AZone")
    assert puca_panel.daypart() in {"dark", "light"}


# ── the question ─────────────────────────────────────────────────────────


def test_a_question_keeps_only_playable_options(monkeypatch):
    question = puca_panel.ask_choice(
        "Which did you mean?",
        [
            option("a", "First"),
            {"video_id": "", "title": "No id"},
            {"video_id": "b", "title": ""},
            "not an option",
            option("c", "Third"),
        ],
        source="play",
        query="first",
    )
    assert [item["video_id"] for item in question["options"]] == ["a", "c"]
    assert question["query"] == "first"
    assert question["source"] == "play"


def test_a_question_with_nothing_to_choose_between_is_refused():
    with pytest.raises(puca_panel.PanelError) as raised:
        puca_panel.ask_choice("Which did you mean?", [])
    assert raised.value.code == "no_options"


def test_a_question_is_bounded_and_expires(monkeypatch):
    asked = puca_panel.ask_choice(
        "Which did you mean?",
        [option("v%d" % index, "Track %d" % index) for index in range(10)],
    )
    assert len(asked["options"]) == puca_panel.MAX_OPTIONS
    assert puca_panel.pending_question()["id"] == asked["id"]

    # A question left on the wall overnight is not a question any more.
    moment = [asked["asked_at"] + puca_panel.QUESTION_TTL_SECONDS + 1]
    monkeypatch.setattr(puca_panel.time, "time", lambda: moment[0])
    assert puca_panel.pending_question() is None
    assert puca_panel.clear_question() is False


def test_answering_plays_the_chosen_track_once(monkeypatch):
    played = []
    monkeypatch.setattr(puca_panel.music_api, "_provider_call", lambda key, extra: ("opendeck", play_result(video_id="chosen-1")))
    monkeypatch.setattr(puca_panel.music_api, "_start_playback", lambda result, skill_id: played.append(result["video_id"]))

    puca_panel.ask_choice("Which did you mean?", [option("a", "First"), option("b", "Second")])
    answered = puca_panel.answer_question(1)

    assert answered["chosen"]["video_id"] == "b"
    assert played == ["chosen-1"], "the chosen track is what the provider is asked to play"
    assert puca_panel.pending_question() is None
    with pytest.raises(puca_panel.PanelError) as raised:
        puca_panel.answer_question(0)
    assert raised.value.code == "no_question", "a second tap cannot start a second track"


def test_answering_something_that_was_not_offered_is_refused():
    puca_panel.ask_choice("Which did you mean?", [option("a", "First")])
    for bad in (1, -1, "0", None, True):
        with pytest.raises(puca_panel.PanelError) as raised:
            puca_panel.answer_question(bad)
        assert raised.value.code == "bad_option"
    assert puca_panel.pending_question() is not None, "a refused answer leaves the question up"


def test_dismissing_closes_the_question():
    puca_panel.ask_choice("Which did you mean?", [option("a", "First")])
    assert puca_panel.clear_question() is True
    assert puca_panel.pending_question() is None
    assert puca_panel.clear_question() is False


# ── the panel's own pages ────────────────────────────────────────────────


def test_the_page_comes_from_the_installed_music_provider(panel):
    server, _runtime = panel
    status, body, headers = read_raw(server, "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert body == "// PUCA-Panel/panel.html\n"
    assert "no-store" not in headers.get("Cache-Control", "")

    skill_id, _manifest = puca_panel.panel_skill()
    assert skill_id == "opendeck"


def test_only_the_panel_folder_of_the_package_is_served(panel):
    server, _runtime = panel
    for asset in ("PUCA-Panel/panel.js", "PUCA-Panel/panel.css"):
        status, _body, headers = read_raw(server, "/" + asset)
        assert status == 200, asset
        assert headers["Content-Type"] in {"application/javascript; charset=utf-8", "text/css; charset=utf-8"}
    # Declared files outside the panel folder stay unservable, and so does a path
    # that tries to climb out of the package.
    for asset in ("skill.json", "opendeck_worker.py", "ui/music.js", "PUCA-Panel/../skill.json", "PUCA-Panel/missing.js"):
        status, _body, _headers = read_json(server, "/" + asset)
        assert status == 404, asset


def test_the_panel_says_so_when_no_provider_ships_a_page(music_env, monkeypatch):
    install(monkeypatch, music_env, [{**PROVIDER_MANIFEST, "files": PROVIDER_MANIFEST["files"]}], FakeRuntime())
    server = puca_panel.create_server("127.0.0.1", 0)
    from threading import Thread

    Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, payload, _headers = read_json(server, "/api/panel")
        assert status == 200
        assert payload["available"] is False
        assert "no_panel" in json.dumps(payload) or payload["skill_id"] is None
        status, payload, _headers = read_json(server, "/")
        assert status == 404 and payload["code"] == "no_panel"
    finally:
        server.shutdown()
        server.server_close()


# ── the music API on the panel's own origin ──────────────────────────────


def test_the_panel_serves_the_same_music_api_as_the_console(panel):
    server, runtime = panel
    status, payload, _headers = read_json(server, "/api/music/queue")
    assert status == 200 and payload["ok"] is True
    assert payload["queue"] == []

    status, payload, _headers = read_json(server, "/api/music/search?q=test")
    assert status == 200 and payload["ok"] is True

    status, payload, _headers = read_json(server, "/api/music/play", {"video_id": "vid-1"})
    assert status == 200 and payload["ok"] is True
    assert payload["track"]["video_id"] == "vid-1"
    assert [call["arguments"]["action"] for call in runtime.calls if call.get("arguments")] != []


def test_a_route_the_console_does_not_serve_is_not_invented_here(panel):
    server, _runtime = panel
    status, payload, _headers = read_json(server, "/api/music/save_everything")
    assert status == 404 and payload["ok"] is False
    status, payload, _headers = read_json(server, "/api/not-a-route")
    assert status == 404


def test_the_state_call_carries_what_one_paint_needs(panel):
    server, _runtime = panel
    status, payload, _headers = read_json(server, "/api/panel/state")
    assert status == 200 and payload["ok"] is True
    assert set(payload) >= {"panel", "theme", "question", "now_playing", "queue", "server_time"}
    assert payload["panel"]["port"] == puca_panel.PANEL_PORT
    assert payload["panel"]["page"] == "PUCA-Panel/panel.html"
    assert payload["panel"]["skill_id"] == "opendeck"
    assert payload["question"] is None
    assert payload["queue"]["queue"] == []


def test_a_hardware_hiccup_does_not_blank_the_display(monkeypatch, panel):
    server, _runtime = panel

    def explode(*_args, **_kwargs):
        raise RuntimeError("the device is asleep")

    monkeypatch.setattr(puca_panel.music_api, "_provider_call", explode)
    status, payload, _headers = read_json(server, "/api/panel/state")
    assert status == 200 and payload["ok"] is True
    assert payload["queue"] is None and "asleep" in payload["queue_error"]
    assert payload["now_playing"] is not None, "the recorded track is still shown"


# ── the question over HTTP ───────────────────────────────────────────────


def test_the_question_round_trips_over_the_panel_routes(panel, monkeypatch):
    server, _runtime = panel
    played = []
    monkeypatch.setattr(puca_panel.music_api, "_start_playback", lambda result, skill_id: played.append(result["video_id"]))

    status, payload, _headers = read_json(server, "/api/panel/question", {
        "prompt": "Which did you mean?",
        "options": [option("a", "First"), option("b", "Second")],
        "source": "play",
        "query": "midnight",
    })
    assert status == 200 and payload["question"]["query"] == "midnight"

    _status, state, _headers = read_json(server, "/api/panel/state")
    assert state["question"]["prompt"] == "Which did you mean?"

    status, payload, _headers = read_json(server, "/api/panel/answer", {"index": 5})
    assert status == 400 and payload["code"] == "bad_option"

    status, payload, _headers = read_json(server, "/api/panel/answer", {"index": 1})
    assert status == 200 and payload["chosen"]["video_id"] == "b"
    assert played == ["vid-1"]
    assert payload["question"] is None

    status, payload, _headers = read_json(server, "/api/panel/dismiss", {})
    assert status == 200 and payload["dismissed"] is False


def test_an_unclear_play_becomes_the_question_the_panel_asks(music_env, monkeypatch):
    options = [option("a", "Midnight City"), option("b", "Midnight Sun")]
    runtime = install(monkeypatch, music_env, [PROVIDER], FakeRuntime(
        results={"opendeck_music": play_result(ambiguous=True, options=options)},
    ))
    started = []
    monkeypatch.setattr(music_api, "_start_playback", lambda result, skill_id: started.append(result))
    before = music_api._read_music_state()

    import console_extend
    from http.server import ThreadingHTTPServer
    from threading import Thread

    server = ThreadingHTTPServer(("127.0.0.1", 0), console_extend.ExtendedHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, payload, _headers = read_json(server, "/api/music/play", {"query": "midnight"})
        assert status == 200
        assert payload["ok"] is True and payload["ambiguous"] is True
        assert [item["video_id"] for item in payload["options"]] == ["a", "b"]
        assert payload["question"]["prompt"].startswith("2 tracks match")
        assert payload["panel_available"] is True
        assert payload["panel_url"].endswith(":%d/" % puca_panel.PANEL_PORT)
        # The reply carries the current now-playing snapshot (here: nothing),
        # and the one thing that matters is that this request wrote no state.
        assert payload["track"] is None
        assert started == [], "an unclear request must never start a track"
        assert music_api._read_music_state() == before, "and must not record one either"
        assert puca_panel.pending_question()["query"] == "midnight"
        assert runtime.calls[-1]["arguments"]["action"] == "play"
    finally:
        server.shutdown()
        server.server_close()


def test_the_console_points_at_the_panel(music_env, monkeypatch):
    install(monkeypatch, music_env, [PROVIDER], FakeRuntime())
    import console_extend
    from http.server import ThreadingHTTPServer
    from threading import Thread

    server = ThreadingHTTPServer(("127.0.0.1", 0), console_extend.ExtendedHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        puca_panel.ask_choice("Which did you mean?", [option("a", "First")])
        status, payload, _headers = read_json(server, "/api/panel")
        assert status == 200
        assert payload["port"] == puca_panel.PANEL_PORT
        assert payload["available"] is True
        assert payload["question"]["options"][0]["video_id"] == "a"
    finally:
        server.shutdown()
        server.server_close()


# ── what the assistant says ──────────────────────────────────────────────


def test_an_unclear_play_is_reported_as_the_choice_it_is(monkeypatch):
    monkeypatch.setattr(puca_panel, "panel_info", lambda: {"available": True})
    question = puca_panel.ask_choice(
        "2 tracks match “that song”: which one did you mean?",
        [option("a", "Midnight City"), option("b", "Midnight Sun")],
        source="play",
        query="that song",
    )
    handoff = {
        "supported": False,
        "ambiguous": True,
        "panel": True,
        "question": question,
        "options": question["options"],
    }
    reply = brain._playback_reply_with_handoff("Playing something.", handoff)
    assert "Midnight City" in reply and "Midnight Sun" in reply
    assert "PUCA Panel" in reply
    assert "Playing something." not in reply, "the provider's wording would claim it played"

    handoff["panel"] = False
    assert "Say which one you meant" in brain._playback_reply_with_handoff("Playing something.", handoff)


def test_the_media_handoff_parks_the_question_instead_of_playing(monkeypatch):
    options = [option("a", "Midnight City"), option("b", "Midnight Sun")]
    execution = {
        "outcomes": [
            {"skill_id": "opendeck", "result": play_result(ambiguous=True, options=options)},
        ]
    }
    started = []
    monkeypatch.setattr(music_api, "_start_playback", lambda result, skill_id: started.append(result))
    instance = brain.Brain.__new__(brain.Brain)

    handoff = instance._hand_over_media_playback(execution)

    assert handoff["ambiguous"] is True
    assert handoff["options"][0]["title"] == "Midnight City"
    assert started == []
    assert puca_panel.pending_question()["options"][1]["video_id"] == "b"


def test_a_play_with_no_candidates_is_reported_honestly(monkeypatch):
    execution = {"outcomes": [{"skill_id": "opendeck", "result": play_result(ambiguous=True, options=[])}]}
    instance = brain.Brain.__new__(brain.Brain)
    handoff = instance._hand_over_media_playback(execution)
    assert handoff.get("ambiguous") is not True
    assert handoff["supported"] is False
    assert "which one did you mean" in handoff["error"]


def test_a_played_track_is_still_handed_over(monkeypatch):
    execution = {"outcomes": [{"skill_id": "opendeck", "result": play_result()}]}
    monkeypatch.setattr(music_api, "_start_playback", lambda result, skill_id: {"handoff": {"supported": True, "skill_id": skill_id}})
    instance = brain.Brain.__new__(brain.Brain)
    handoff = instance._hand_over_media_playback(execution)
    assert handoff == {"supported": True, "skill_id": "opendeck"}


def test_play_options_reads_only_a_refusal_to_choose():
    assert music_api.play_options({"play_url": "x"}) is None
    assert music_api.play_options({"ambiguous": False, "options": [option("a", "First")]}) is None
    assert music_api.play_options({"ambiguous": True}) == []
    assert music_api.play_options({"ambiguous": True, "options": [option("a", "First"), "junk"]}) == [option("a", "First")]
