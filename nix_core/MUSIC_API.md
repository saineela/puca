# NIX music page and music control API

**Status:** implemented in `console_extend.py` + `music_api.py`. The page shell
and routes are the console's; the page contents, the catalog and the lyrics come
from an installed skill. Transport control is resolved from the declared device
controls of whichever installed skill owns the speaker.

## How a skill fills the page

A trusted, installed skill contributes a page and a provider by declaring two
blocks in its `skill.json` (validated by `console._parse_skill_manifest`):

```json
"device_ui": {
  "page": {"id": "music", "title": "Music", "script": "ui/music.js", "styles": ["ui/music.css"]},
  "media": {"provider": {
    "search":  {"tool": "opendeck_music", "arguments": {"action": "search"}},
    "play":    {"tool": "opendeck_music", "arguments": {"action": "play"}},
    "library": {"tool": "opendeck_music", "arguments": {"action": "library", "kind": "playlists"}},  "lyrics":  {"tool": "opendeck_music", "arguments": {"action": "lyrics"}},
  "queue":   {"tool": "opendeck_music", "arguments": {"action": "queue", "op": "list"}},
  "save":    {"tool": "opendeck_music", "arguments": {"action": "save"}}
  }}
}
```

- `page.id` must be a console page this build ships (`music`). A package cannot
  invent a dashboard route, and only its **declared** `.js`/`.css` files are
  served, from `/api/skills/<skill_id>/page/<path>`.
- Declared provider `arguments` are defaults the API completes per request (a
  search adds the query, a play adds the track). Every declared value must match
  its own property in the tool's input schema, and the merged call is validated
  against the tool schema before it reaches the worker.
- The page script is loaded into the console page and runs at the console
  origin — the same explicit package-digest trust that governs the worker
  applies to it. The script defines `window.NixMusicPage = { mount(root, api) }`
  and receives the API surface below.

## Page (web)

`GET /music` serves the console page shell, which appears in the sidebar as soon
as a music provider skill is installed and trusted. The shell renders the
heading, the provider name, which player commands are available, and mounts the
skill's page script inside `.music-app`.

## JSON API

`/api/music` is the primary surface. Every mobile operation is also served under
`/api/mobile/v1/music…` with identical payloads, so the NIX Home app renders the
same page against the same state.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/music` | Capabilities: the page, its assets, the provider skill and which player commands exist. Answers `ok:false`, `code:"no_music_provider"` when no provider is installed. |
| `GET` | `/api/music/now-playing[?lyrics=1]` | The recorded track, the speaker's reported state, an estimated position and the hand-off result. |
| `POST` | `/api/music/control` | `{"action":"play"\|"pause"\|"resume"\|"stop"\|"mute"\|"unmute"\|"volume","value":0..1}` — routed to the installed skill's declared control, with a plain-language `message` for the result. |
| `GET` | `/api/music/search?q=&limit=[&refresh=1]` | Provider catalog search, answered from the console's catalog cache. |
| `POST` | `/api/music/play` | `{"query"}` or `{"video_id"}` — resolves the stream through the provider, remembers the track, and hands the URL to a `play_url` control when one is declared. |
| `GET` | `/api/music/queue[?from=&limit=&revision=]` | The play queue: `{queue:[…], index, count, total, offset, revision, unchanged, message}`. `queue` is a window of the line-up and `total` is how long it is; with `revision` set to the one the caller holds, an unchanged queue answers `unchanged:true` with no tracks at all. |
| `POST` | `/api/music/queue` | `{"action":"add"\|"add_many"\|"add_range"\|"remove"\|"move"\|"front"\|"next_up"\|"clear"\|"next"\|"play", …}` — changes the queue, or resolves a queued track and hands it to the speaker. `add_many` takes `tracks:[{video_id,title,…},…]`, which is how a whole playlist changes hands in one call. |
| `POST` | `/api/music/queue/resolve` | `{"video_id"}` — the track's metadata, read-only: nothing is queued and nothing plays. This is what the queue service calls back for. |
| `POST` | `/internal/music/queue/play` | `{"video_id","index"}` — the queue service's player callback: resolves one track it has already chosen and hands it to the speaker. It never chooses and never reads the queue. |
| `GET` | `/api/music/library?kind=&limit=&playlist_id=[&refresh=1]` | Provider library read (`playlists`, `playlist`, `likes`, `history`, `frequent`, `stats`), answered from the console's catalog cache. |
| `POST` | `/api/music/save` | `{"kind":"like"\|"unlike"\|"playlist_add"\|"playlist_create", …}` — writes to the user's own library through the declared `save` provider action. |
| `GET` | `/api/music/lyrics[?video_id=|title=|artist=|album=|duration_sec=]` | Synchronized lyrics for the current or named track, plus the same position estimate. |
| `GET` | `/api/music/artwork?src=<image url>` | One cover, from the console's own artwork cache: served from disk, fetched and stored on a miss, safe to keep for a month (`ETag`, `304`). |
| `GET` | `/api/skills/<skill_id>/page/<path>` | A declared page asset of an installed skill. |

**The queue has a service of its own.** When `NIX_MUSIC_QUEUE_URL` is set — or
when the service has published its address in the console's data directory
(`<NIX_DATA_DIR>/music-queue.json`, which is how a box finds its own queue with
nothing configured) — the queue lives in `nix_core/music_queue_service.py` rather
than in the provider skill: a playlist of any length is one write (`add_many`), reads are windows
carrying a revision, and the tracks ahead of the cursor are warmed in the
background at the rate the OpenDeck instance declares. The provider is still what
resolves a stream and receives the hand-off, so a queued song plays exactly like
a searched one. With neither an address nor a published one, every route above
goes back to the provider skill's own queue, exactly as it did before. See
`MUSIC_QUEUE.md`.

Playback is a single call by design: the provider resolves a stream that can be
fetched whole, the player skill prepares it for its own speaker, and the console
returns one payload — there is no job to poll and no second round trip to start
the music. The play queue below is a different thing: a list of tracks the user
lined up, not an asynchronous job.

**What the account plays most is the provider's too.** `kind=frequent` asks the
provider for the songs this account plays most often, ranked by play count, each
track carrying that count as `plays` (`history` is the same log in the order it
happened, and `stats` is the provider's own summary). The OpenDeck provider
ranks them from its play log, so a wall display can offer "Frequently played"
as a row of its own instead of guessing at a recommendation.

**What a playlist is belongs to the provider.** `kind=playlists` returns whatever
the installed provider calls a playlist; the OpenDeck provider reads the linked
YouTube Music account, so the list is that account's playlists plus its **Liked
Music** auto playlist (`playlist_id: "LM"`) and then any playlist created through
OpenDeck itself. Opening `kind=playlist&playlist_id=LM` returns the liked songs,
the same tracks `kind=likes` returns — liked songs are a playlist in YouTube
Music too, so the page and the app offer them as one.

**Each playlist carries its cover.** Every entry of `kind=playlists` has an
`artwork_url` — the thumbnail the provider resolved for that playlist, the cover
a stored playlist was saved with, or, for **Liked Music**, the first liked
song's cover when the playlist itself has none. It may be an empty string, which
is a playlist with no picture rather than a missing field: the page draws its own
tinted tile for it instead of a broken image.

**Cover art is cached too.** Every `artwork_url` the page paints goes through
`GET /api/music/artwork`, which serves the image from the console's own disk
cache: the first paint fetches the cover once, every later paint is a local file
that the browser also keeps (`max-age` a month, `ETag`, `304` on a re-request). A
provider's CDN is therefore contacted once per cover, not once per render — the
music page shell rewrites the URLs it hands a page script, so a skill page paints
cached covers without changing anything.

**Reads are cached in three layers, each one in front of the next.** The
OpenDeck provider keeps each library read (playlists, a playlist's tracks, likes,
history) on disk for ten minutes and drops the whole cache on any write, so a
repeat read answers instantly instead of browsing a linked account again
(measured: about 5.7s cold, under 10ms warm). The console keeps its own copy of
the catalog behind `/api/music/library` and `/api/music/search`, so a read that
the assistant, the mobile app or a second browser makes never pays the
skill-worker round trip either: a cached copy is replayed immediately and a
background refresh asks the provider again, whose reply becomes the new cached
copy when it arrives. `X-Music-Cache` (`hit`, `stale`, `miss`, `refresh`,
`disabled`) and `X-Music-Cache-Age` say which copy answered, `?refresh=1` skips
the cached copy, and any successful write (`POST /api/music/save`) drops the
catalog it invalidates. The page adds its own copy in local storage on top, which
is what makes opening the Music page instant: it paints what it had and refreshes
from here behind it.

### now-playing payload

```json
{"ok":true,
 "track":{"video_id":"BSTsnWoslP4","title":"Bohemian Rhapsody","artist":"Queen","duration_sec":354,
          "artwork_url":"…","mime_type":"audio/mp4","play_url":"http://…/api/media/stream?…"},
 "playing":true,
 "player":{"state":"playing","volume":0.53,"muted":false,"device":"Light","confirmed":true},
 "position":{"elapsed_sec":42.5,"duration_sec":354,"estimated":true,"basis":"console_started_at","paused":false},
 "handoff":{"supported":true,"skill_id":"…:echo-connect","result":{…}},
 "updated_at":1791700000.0,"source":"github:saineela/opendeck-skill:opendeck"}
```

**Position is an estimate.** The Dot's ESPHome media player reports state,
volume and mute but no playback position, so the console records when it handed
the track over and both the page and the app derive progress from that
(`started_at` minus paused time) and keep their own clock between polls. The
clock is anchored when the speaker is *commanded*, not when the provider started
resolving the track: resolving and converting a track can take a second or two,
and anchoring at the earlier moment made progress and lyrics run ahead of the
audio. Lyrics highlighting uses the same clock, which is why the page and the
app stay in step without a media-progress channel.

### The play queue

**The queue belongs to the provider skill, not to the console.** A provider that
declares a `queue` action owns the list, its ordering and its persistence; the
console reads and drives it and performs the speaker hand-off exactly as
`POST /api/music/play` does. That is why the assistant can fill the same queue
through the provider's own tool and the page sees it immediately — there is one
list, in one place.

| Method | Body | Effect |
| --- | --- | --- |
| `GET` | — | The provider's current list, normalized to `{ok, action:"queue", queue, index, count, message}`. |
| `POST` | `{"action":"add","query"}` or `{"action":"add","video_id"}` | Resolves the track through the provider and appends it. The speaker is not touched. |
| `POST` | `{"action":"remove","index"}` | Drops one entry. |
| `POST` | `{"action":"remove","query":"Believer"}` | Drops the entry that name resolves to — “remove Believer from the queue” without the caller having to know its position. A name that is not queued is refused (`queue_entry_missing`) instead of dropping a different track. |
| `POST` | `{"action":"move","index","to"}` | Moves the entry at `index` to position `to` — one reorder, which is what dropping a dragged row sends. Both are 0-based queue positions. |
| `POST` | `{"action":"front"\|"next_up","index"\|\"query"\|\"video_id"}` | Prioritises a track: `front` puts it first, `next_up` puts it behind the playing one. A named track that is not queued yet is inserted at that spot. Neither starts playback. |
| `POST` | `{"action":"clear"}` | Empties the queue. |
| `POST` | `{"action":"play","index"}` / `{"action":"next"}` | Resolves that entry and, when the provider returns a `play_url`, hands it to the speaker and records it as now-playing. |
| `POST` | `{"action":"shuffle"}` / `{"action":"shuffle","on":true\|false}` | Shuffles the tracks waiting behind the playing one, or puts them back in queued order. No `on` toggles. |
| `POST` | `{"action":"repeat","mode":"off"\|\"all\"\|\"one\"}` | How the queue behaves when a track ends. No `mode` cycles off → all → one. |

```json
{"ok":true,"skill_id":"…:opendeck","action":"queue",
 "queue":[{"video_id":"…","title":"Second","artist":"…","duration_sec":200,"artwork_url":"…"}],
 "index":0,"count":1,"shuffle":false,"repeat":"off",
 "message":"Playing queued track 1 of 1: “Second”.",
 "track":{…},"playing":true,"player":{…},"position":{…},"handoff":{…}}
```

**Shuffle and repeat travel with the queue.** Every queue reply carries
`shuffle` and `repeat`, so a page renders its two toggles from the same answer
that changed or listed the queue — a toggle never has to guess, and a second
read is not needed to learn what it became. Turning shuffle off restores the
order the tracks were queued in (a track added while shuffled keeps its place at
the end), so the control is reversible rather than one-way.

**Repeat-one needs to tell an ending from a skip.** The console's auto-advance
sends `{"action":"next","auto":true}`; a page's Next button sends `next`
without it. Under `repeat: "one"` the first restarts the track that just
finished, while the second still moves on — which is what a skip means.

`play` and `next` answer with the same track/player/position/handoff fields a
play answers, so a client has one handling path. An exhausted queue is not an
error in the provider — it simply carries no `play_url`, and the console turns
that into `409 queue_empty` so a caller never mistakes "nothing left" for
"playing".

**A queued track advances by itself.** The speaker emits no end-of-track event,
so the console notices on the `now-playing` poll: the current track is at or
past its length *and* the speaker has gone idle, and only then does it ask the
provider for the next entry. A one-off play never auto-advances (the recorded
state marks queue-sourced playback), and a paused speaker is not "finished", so
the poll cannot hijack a deliberate pause.

**One speaker, one decider.** When the queue service is running, the poll is a
report rather than a decision: the console sends the speaker's state to the
service's `POST /playback`, and the service advances the line-up at most once
per finished track — however many devices (page, wall panel, phone) are polling
the same speaker. The service remembers the ending it acted on (the track, and
how far into it the speaker was), so a report below that mark is progress, and
the end itself is the one report that advances. Its reply carries `executed`,
which says whether the service has already started the next track through `POST
/internal/music/queue/play`; if it has, the console must not hand the same track
over again. A report of a track the queue did not choose, or of a track stopped
before its end, moves nothing, and a position back at the start of the track
clears the mark — one ending per pass, which is what repeat-one needs. The same
rule holds when the service runs without a player: it still decides, answers
`executed:false`, and the console performs that one hand-off.

### Saving to the user's own library

Liking a track and putting one into a playlist are writes to the user's own
OpenDeck account, so they go through a declared `save` provider action and never
through the console's own storage. The console fills in the track: with nothing
named, `POST /api/music/save` saves what is playing, so the page does not have
to know a video id it never showed.

| Body | Effect |
| --- | --- |
| `{"kind":"like"}` / `{"kind":"unlike"}` | Likes or unlikes the playing track (or `video_id`/`title`/`artist` when given). Answers `liked: true\|false`. |
| `{"kind":"playlist_add","playlist_id":"…"}` | Appends the track to that playlist. |
| `{"kind":"playlist_add","name":"Late Night"}` | The provider finds the playlist by name (case-insensitively, across the linked account and the instance's own store) and creates it when the name is new, so a spoken "add this to my Late Night playlist" needs no lookup first. |
| `{"kind":"playlist_create","name":"…"}` | Creates a playlist and answers its `playlist_id`, so a caller can add to what the provider actually created. |

Nothing playing and no track named is a `409 no_track`, and an unknown `kind`
is a `400 validation_error`. The provider decides what a write means; the
console only completes and validates the call.

### Errors

`{"ok":false,"code":"…","error":"…"}` with a stable code: `no_music_provider`
(409), `skill_not_runnable` (409), `player_action_unavailable` (501),
`provider_action_unsupported` (501), `skill_call_failed` (502),
`validation_error` (400), `no_track` (409), `queue_empty` (409),
`artwork_unavailable` (502, an image that could not be fetched or is not an
image), `artwork_too_large` (502, beyond the 2 MiB cover cap). A missing player
command is a normal state, not a crash: install or update the skill that owns
the speaker.

## Transport resolution

`/api/music/control` does not name a skill. It looks for an installed skill that
declares a device control for the action — `{"action":"volume"}` with a single
`volume` input, `{"action":"mute"}`, `{"action":"stop_speaking"}`, and so on —
and calls that skill's tool with the merged arguments. A `play_url` control
(with a `url` input) is what makes `/api/music/play` actually start playback on
the speaker; without one the API still resolves and records the track and
reports `handoff.supported = false` with the reason.

A **spoken** play (`"play Fate of Ophelia by Taylor Swift"`) goes through the
assistant and the provider's own tool, so the console makes the same hand-off
there: after a confirmed execution whose result carries a `play_url`, Core calls
this same `_start_playback`, which commands the speaker and records now-playing.
The reply says so plainly when the hand-off could not happen (`"no installed
skill can play a stream URL"`, or the speaker's refusal), instead of leaving the
provider's `"Playing …"` sentence to read as playback that never started.

A **spoken** transport command is Core's too. `"pause"`, `"stop the music"`,
`"resume"`, `"mute"` and their forms are resolved by Core (`brain`) and run
through this same resolution step — one declared control each, no model picks
between actions — and the confirmation is phrased from what the speaker reports,
so a pause the device did not take is never reported as paused. Two details are
deliberate: `"stop the music"` maps onto the **pause** control, because the
speaker has no stop, while `stop` itself stays the speaker's own voice control;
and a request with a second action in it (`"pause the music and turn on the ring
light"`) or one that means the voice (`"stop speaking"`) is not read as
transport at all.

Transport controls answer as soon as the command is accepted (tens of
milliseconds). Pause/resume confirm by polling the speaker's reported state
with a short bound instead of waiting for a pushed event that some firmware
never emits, so a transport tap never blocks for seconds. `/api/music/play` is
the one slow call, and only because the speaker's format may need conversion.

### Playback: what is measured (reference Dot, live instance)

Starting playback is the only step that depends on the provider's stream URL,
and both halves of it were timed on the reference setup:

- **The resolved format matters.** A DASH (`adaptiveFormats`) audio URL serves
  only the opening of the file — the CDN refuses a request past roughly its
  first 256 KiB, which is about 42 s at 48 kbps — so the OpenDeck provider
  resolves playback to a *progressive* (`formats`) rendition, whose URL streams
  end to end. Full tracks therefore play, not just their first minute.
- **The whole hand-off takes about a second** for a 5½-minute track: provider
  resolve ~0.2 s, one bounded fetch of the 6 MB progressive stream ~0.2 s,
  ffmpeg conversion to the speaker's WAV ~0.4 s, then the device command. The
  reference Dot advertises 48 kHz stereo WAV only, so audio is always converted
  for it; a speaker that advertises the container it is handed gets the URL
  directly and skips the conversion.
- **A stream that cannot be fetched completely becomes a shorter track, not an
  error.** Whatever the origin will serve is converted, and the seconds of audio
  actually prepared are reported in the skill's message rather than implied.

These are provider/environment facts, not API defects: `handoff.result` or
`handoff.error` always states what actually happened.

**A dropped device session recovers by itself.** A skill worker holds its own
connection to its device, so a reboot, a Wi-Fi blip or an idle timeout makes
later calls fail with a transport error while the device is still reachable. A
control call that fails that way restarts the worker once and tries again, so a
speaker that is merely reconnecting does not surface as a `502`.

## Unclear play requests

`POST /api/music/play` with a `query` is answered by the provider, and a provider
that cannot tell which catalog match the request meant does not pick one. It
answers with the candidates instead:

```json
{"ok": true, "ambiguous": true, "query": "that song from the film",
 "options": [{"video_id": "…", "title": "…", "artist": "…", "duration_sec": 214,
             "album": "…", "artwork_url": "…"}],
 "question": {"id": "…", "prompt": "3 tracks match …: which one did you mean?",
              "source": "play", "options": ["…"], "asked_at": 0, "expires_at": 0},
 "panel_available": true, "panel_url": "http://hub:49170/", "track": null}
```

Nothing plays: the guess the user did not make is the whole point of asking. The
reply still carries the current now-playing snapshot (`track`, `playing`,
`player`, `position`) — that is whatever was already playing, not something this
request started — and it carries no `result`, because the provider resolved no
stream. The
question is held in Core (`puca_panel`), not in a page, so it survives a reload
and every surface can read it — the panel asks it on screen, the Music page names
the options in its own notice, and a spoken reply says the options and where to
choose them. A question expires after 15 minutes.

`GET /api/panel` reports the question and where the display lives. The panel's own
routes are on its own origin: `GET /api/panel/state` (theme, question, now
playing, the queue and the `build` of the panel being served in one round trip),
`POST /api/panel/answer` with
`{"index": 0}` (plays that option and closes the question), `POST
/api/panel/dismiss`, and `POST /api/panel/question`.

The display is also *told*, rather than only asking: `GET /api/panel/socket`
upgrades the same connection to a websocket (`panel_stream.py`, the page's own
origin, so `wss://` and the TLS listener come for free). The console sends one
small `{"type": "event", "event": "playing"|"question"|"question-cleared"|...}`
frame when a track starts, a question is parked or cleared, the transport is
driven or the queue moves, and the page answers it by reading
`/api/panel/state` — the socket is latency, not a second implementation of the
state. A plain (non-upgrade) `GET` of the socket path is refused with `426`.

## PUCA Panel (wall display)

The panel is a second listener (`puca_panel.py`, port `49170`, `NIX_PANEL_PORT`)
for a screen that is read from across a room. It serves the page out of the
installed music provider's package — `PUCA-Panel/panel.html`, `panel.css`,
`panel.js`, all declared in the skill's `files` — and the same `/api/music`
handlers as the console, so the music API is not implemented twice and the page
needs no cross-origin setup.

- The theme follows the console's clock (dark from 22:00, light from 06:00, in
  the console's own timezone), not the screen's.
- The console **pushes** a change to the display over a websocket on the page's
  own origin (`/api/panel/socket`), so a track that starts or a question that
  arrives is on the wall immediately rather than at its next poll; the page
  reconnects with a growing wait when the channel drops and still polls on a slow
  fallback, so a display is never left stale. `?stream=0` turns the channel off,
  and `window.__puca.stream()` reports `{enabled, up, url, attempts}`.
- The lyrics of whatever is playing open by themselves and stay open until the
  user closes them; the next track opens them again.
- Spoken requests are transcribed on the box (see *Hearing* below): a phrase that
  names a command (`pause`, `next`, `volume up`) is obeyed, `play <song>` is the
  same `play` call the assistant makes — including the unclear case above — and
  anything else becomes a search the user picks from.
- The playlists row ends in the **library page**: every playlist the provider
  lists, each wearing its own cover, and a tap opens that playlist's **whole song
  list** — `kind=playlist` with the reader's own `limit`.
- A shelf the user scrolls to is read at the **whole-library ceiling** (500), not
  as a short preview: the liked-songs shelf renders every song the provider lists,
  so a 430-song *Liked Music* playlist arrives whole instead of stopping at 15.
  `kind=library` keeps the ceiling honest — the provider caps a read it cannot
  fill and the console's own catalog cache is keyed by the `limit` asked for, so
  a short read and a whole read never share an entry.
- A search answers with more than songs: the page matches the playlists it already
  holds by name and the artists the songs on it are by, so “Miles” offers the
  artist as a doorway (a tap searches the catalog for them) and any playlist that
  goes by the word, alongside the songs `GET /api/music/search` returned.
- Every song row carries a **queue** control beside it — `POST /api/music/queue`
  `{"action":"add","video_id":…}` — so a list can be lined up behind what is
  playing, not only sent to the speaker.
- After a minute of nobody touching it the wall hands itself over to the package's
  own idle page (`PUCA-Panel/screensaver.html`, in an iframe so its stylesheet stays
  out of the panel's), and any touch brings the panel back to what it was showing.
  `?saver=off` and `?saverAfter=<seconds>` change that. What counts as somebody
  being there is deliberately narrow — music playing, a question waiting for an
  answer, the lyrics (while a song is playing, or when they were opened by hand)
  and a list opened within that same minute; a wall left on the last song's
  auto-opened lyrics is idle, which is the state a live panel sat in for days.
  `window.__puca.idle()` on the page reports the decision (`saving`, `idle_ms`,
  `something_to_show`, `lyrics_open`, `lyrics_by_user`, `sheet_open`) so a wall
  nobody can stand in front of can still be asked why it is awake.

- The page asks **no font CDN** for its typography: the faces it draws with ship
  in the package (`PUCA-Panel/fonts.css` and its `fonts/*.woff2`, both declared in
  the skill's `files`) and this listener serves them from its own origin, so a
  display on a wall with no internet still paints the design it was given.

Only the panel folder is servable from that listener, path-normalised inside the
package; a package's other files (`skill.json`, the worker, the Music page's own
assets) are not reachable there. A declared asset answers both by the path the
manifest gives it (`/PUCA-Panel/panel.css`) and by the name the page inside that
folder uses (`/panel.css`, `/fonts/outfit-latin.woff2`) — a page links its files
the way a page in its own folder does, and a folder of its own is a normal thing
for it to have.

### The wall talks to the box, not to the internet

The page is served by this console, so every read it makes — the state, the
music, the covers, the recogniser — is a relative path on the origin it was
opened at. Pointed at the box's own address the whole wall is local already;
pointed at a tunnel or a tailnet name, every poll leaves the LAN and comes back,
and it stays that way for as long as the page is open because nothing in it ever
asks whether there is a closer way to reach the box — a wall display is not
reloaded by hand, so "it will be local next time somebody opens the right URL"
never happens.

So `panel_info()` reports where the box lives on its own network (`url` and
`url_https`, both built from the box's LAN address), and the panel uses it:
opened anywhere else, it checks whether that address answers (a bare `no-cors`
reachability probe of `/api/panel/state`, so nothing of the console's is read
across origins) and, if it does, hands the wall over to it — the same page and
the same query, once per session, in the page's own direction of travel (a secure
page moves to the box's secure listener, so the microphone it had is not what the
move costs). An address that does not answer leaves the display exactly where it
is: a wall showing music is never traded for a blank page. `?local=0` pins the
wall where it was opened, for a display deliberately watched from outside the
house, and `window.__puca.local()` on the page reports `{auto, here, box_http,
box_https, on_the_box, looked, moved, moved_from, moved_to, reason}` — which is
how a wall nobody can stand in front of is asked which address it is talking to.

### The display keeps itself current

A wall display is pointed at its port once and never reloaded by hand, so the
page that is open is the page that stays open: measured live, the glass kept
drawing an older panel for hours after the console had started serving the next
one, and what the served page did have (the size chips on the lyrics page, the
card folding away as the music is scrolled) was reported as *missing*, because
the page in memory predated it. (That fold is the music column's own scrolling —
the card is the first thing in the column's scroller, so it goes away on the
compositor with no height to lay out per frame; `?fold=0` pins the card and
scrolls the music underneath it instead, and `window.__puca.scroll()` reports
where the column is.) The console therefore reports **`build`** in
`panel_info()` — and so in every `GET /api/panel/state` a display polls —
computed from the size and write time of the panel files it is serving (a stat
per file, read fresh, so re-deploying the skill changes it and nothing else
does). The page compares it with the build it booted into — the first state a
page sees is by definition its own — and reloads itself once per build per
browsing session when they differ, so a console that keeps changing its answer,
or a reload that does not take, cannot turn the wall into a reload loop. The
music carries on: the panel seeks back to the console's position as soon as it
is up. `?refresh=0` pins a page where it is, and `window.__puca.builds()` on the
page reports `{page, served, reload_asked}`.

### Over https, for the microphone

The same display is served over TLS on `49171` (`NIX_PANEL_TLS_PORT`), because a
browser only lends a page a microphone on a **secure** origin — over plain http on
a LAN address the mic button can only say where the working address is. The
certificate is self-signed, generated once into the console's data directory
(`panel-cert.pem`, `panel-key.pem`) and regenerated when the names and addresses
it has to be valid for change (**the box's own LAN address is one of them**, so
the address the wall is handed to is not a name mismatch on top of the self-signed
note); the browser asks to be trusted once. `GET /api/panel` reports both
addresses (`port`/`url` and `tls_port`/`url_https`).

A tap on the microphone on a page the browser will not lend one to is the one
moment the wall can move itself somewhere better: the panel hands the display to
that secure address and arrives listening (`?listen=1`, the microphone opens once
the page is up), so the tap that moved the wall is the tap being listened to —
rather than an address put on screen for somebody standing in front of a wall to
type in. That move is attempted once per session and without a reachability probe
(a probe of a self-signed address fails until the certificate has been accepted
once, which is the very thing the move is for); if the wall is back on a page it
cannot record from, it says where the voice works instead, as it always did.

## Hearing

Speech to text on this machine (`voice.py`). The page records through the
browser's microphone and posts the samples to its own origin; nothing is uploaded
to a speech service, and the audio is discarded when the answer is written.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/voice` | Whether this console can hear: `{available, engine, model, device, loaded, max_seconds}`. The model is warmed in the background when a page asks. |
| `POST` | `/api/voice/transcribe` | A raw WAV body (`audio/wav`, 16-bit PCM, mono or stereo, up to 8 MB / 30 s) → `{"ok":true,"text":"play Bohemian Rhapsody","seconds":2.1,"silent":false}`. |

The model is Whisper (`NIX_STT_MODEL`, default `openai/whisper-small.en`) on
`NIX_STT_DEVICE` (`auto` uses the graphics card only when it has ~2 GB free, so
recognising does not evict the chat model), and the clip is resampled to 16 kHz on
numpy/scipy. A clip that is only room noise is answered with `silent: true` and
empty text rather than a hallucinated phrase, and a malformed body is refused
before the model is touched. Both routes are served by the console **and** by the
panel's own listener: the page that records the audio has to post it back to the
origin that asked for the microphone.

The Music page uses the same two routes with the same helper, and falls back to
the browser's own recogniser when no recogniser is installed here.

## Security notes

- The page must be installed, configured and explicitly trusted at its current
  digest; installing a package still runs nothing.
- Player and provider calls go through `skill_runtime`'s declared schemas, so
  the API cannot reach a tool the skill does not declare.
- Page assets are served only from the declared file list of an installed
  package, path-normalised inside that package, with `nosniff`.
- `GET /api/music/artwork` fetches a caller-supplied URL, so it only accepts
  `http(s)` images and refuses hosts that resolve to a non-public address (a
  loopback, a LAN, a link-local or any other private range), re-checking the
  address a redirect lands on. That keeps the console from being used to reach
  services on its own network, and only `image/*` responses inside the size cap
  are stored.
- The routes are unauthenticated on the console listener, exactly like the
  existing device routes: keep the console on a trusted LAN or bind it to
  loopback. They are not a public mobile backend — see `MOBILE_APP_API.md` for
  the authentication requirements a real mobile deployment needs.
