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
| `POST` | `/api/music/control` | `{"action":"play"\|"pause"\|"resume"\|"stop"\|"mute"\|"unmute"\|"volume","value":0..1}` — routed to the installed skill's declared control. |
| `GET` | `/api/music/search?q=&limit=[&refresh=1]` | Provider catalog search, answered from the console's catalog cache. |
| `POST` | `/api/music/play` | `{"query"}` or `{"video_id"}` — resolves the stream through the provider, remembers the track, and hands the URL to a `play_url` control when one is declared. |
| `GET` | `/api/music/queue` | The play queue as the provider holds it: `{queue:[…], index, count, message}`. |
| `POST` | `/api/music/queue` | `{"action":"add"\|"remove"\|"clear"\|"next"\|"play", …}` — changes the queue, or resolves a queued track and hands it to the speaker. |
| `GET` | `/api/music/library?kind=&limit=&playlist_id=[&refresh=1]` | Provider library read (`playlists`, `playlist`, `likes`, `history`, `stats`), answered from the console's catalog cache. |
| `POST` | `/api/music/save` | `{"kind":"like"\|"unlike"\|"playlist_add"\|"playlist_create", …}` — writes to the user's own library through the declared `save` provider action. |
| `GET` | `/api/music/lyrics[?video_id=|title=|artist=|album=|duration_sec=]` | Synchronized lyrics for the current or named track, plus the same position estimate. |
| `GET` | `/api/music/artwork?src=<image url>` | One cover, from the console's own artwork cache: served from disk, fetched and stored on a miss, safe to keep for a month (`ETag`, `304`). |
| `GET` | `/api/skills/<skill_id>/page/<path>` | A declared page asset of an installed skill. |

Playback is a single call by design: the provider resolves a stream that can be
fetched whole, the player skill prepares it for its own speaker, and the console
returns one payload — there is no job to poll and no second round trip to start
the music. The play queue below is a different thing: a list of tracks the user
lined up, not an asynchronous job.

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
