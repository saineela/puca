/* jsdom behaviour harness for the mini playbar (static/playbar.js).
 *
 * The strip is JavaScript: it builds itself into whatever page it is included
 * by, reads the console's now-playing route, and turns its controls into the
 * same calls the Music page makes. That is checked by running the real script
 * in a DOM and driving its NixPlaybar surface, not by matching strings in the
 * source file.
 *
 * jsdom is a test-only dependency: this looks for it in the usual places and
 * then wherever NIX_JSDOM_ROOT points, and reports a skip (exit code 77) where
 * it is not installed instead of failing a build for a missing dev tool.
 */
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);

function loadJsdom() {
  const roots = [
    process.env.NIX_JSDOM_ROOT,
    ...(process.env.NODE_PATH || "").split(path.delimiter),
    path.join(process.cwd(), "node_modules"),
    "/tmp/node_modules",
    path.join(process.env.npm_config_prefix || process.env.PREFIX || "/usr/local", "lib", "node_modules"),
    path.join(path.dirname(process.execPath), "..", "lib", "node_modules"),
  ].filter(Boolean);
  let failure = "";
  for (const candidate of ["jsdom", ...roots.map((root) => path.join(root, "jsdom"))]) {
    try {
      return { jsdom: require(candidate), failure: "" };
    } catch (error) {
      if (error && error.code !== "MODULE_NOT_FOUND") failure = String(error.message).split("\n")[0];
    }
  }
  return { jsdom: null, failure };
}

const { jsdom, failure } = loadJsdom();
if (!jsdom) {
  process.stdout.write(
    `skip the playbar DOM checks: jsdom is not installed (set NIX_JSDOM_ROOT)${failure ? `: ${failure}` : ""}\n`,
  );
  process.exit(77);
}
const { JSDOM } = jsdom;

const here = path.dirname(fileURLToPath(import.meta.url));
const source = readFileSync(path.join(here, "static", "playbar.js"), "utf8");

const dom = new JSDOM(
  "<!doctype html><html><body><nav><button class=\"nav-item\" data-page=\"music\" id=\"navMusicItem\">Music</button></nav></body></html>",
  { runScripts: "outside-only", url: "http://127.0.0.1:49117/home" },
);
const { window } = dom;

const COVER = "/api/music/artwork?src=https%3A%2F%2Fexample.test%2Fcover.jpg";
let nowPlaying = {
  ok: true,
  track: { video_id: "a1", title: "So What", artist: "Miles Davis", album: "Kind of Blue", duration_sec: 545, artwork_url: COVER },
  playing: true,
  player: { state: "playing", muted: false, volume: 0.5 },
  position: { elapsed_sec: 65 },
  queue_active: true,
};
const calls = [];

window.fetch = async function (url, options) {
  const method = (options && options.method) || "GET";
  const body = options && options.body ? JSON.parse(options.body) : null;
  calls.push(`${method} ${url}${body ? " " + JSON.stringify(body) : ""}`);
  const answer = (payload) => ({ ok: true, status: 200, text: async () => JSON.stringify(payload) });
  if (String(url) === "/api/music/now-playing") return answer(nowPlaying);
  if (String(url).startsWith("/api/music/queue") && method === "GET") {
    return answer({ ok: true, queue: [], index: 2, count: 0, total: 5, shuffle: false, repeat: "off", revision: 7 });
  }
  return answer({ ok: true, message: "done" });
};

window.eval(source);
const bar = window.document.getElementById("nixPlaybar");
const strip = window.document.getElementById("playbarStrip");

const failures = [];
function check(name, condition, detail) {
  if (condition) process.stdout.write(`  ok   ${name}\n`);
  else {
    failures.push(name);
    process.stdout.write(`  FAIL ${name}${detail === undefined ? "" : ` — ${detail}`}\n`);
  }
}
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

check("the strip builds itself, so no page has to declare it", Boolean(bar) && Boolean(strip));
await flush();
await window.NixPlaybar.refresh();

const state = window.NixPlaybar.state();
check("the track's name is on the strip", state.title === "So What" && state.artist === "Miles Davis", JSON.stringify(state));
check("the strip paints the position in the song", state.elapsed >= 65 && state.duration === 545, JSON.stringify(state));
check("it is only shown when there is something to show", state.shown === true);
check("it says the track came from the queue", state.queueActive === true);
check("the cover is drawn from the console's own artwork route",
  window.document.getElementById("playbarArt").innerHTML.includes(COVER),
  window.document.getElementById("playbarArt").innerHTML);

/* ── the controls ──────────────────────────────────────────────────────── */

window.NixPlaybar.act("toggle");
await flush();
await flush();
check("pause and resume go through the console's transport route",
  calls.includes('POST /api/music/control {"action":"pause"}'), calls.join(" | "));

nowPlaying = { ...nowPlaying, playing: false };
await window.NixPlaybar.refresh();
window.NixPlaybar.act("toggle");
await flush();
check("a paused track resumes rather than pausing again",
  calls.includes('POST /api/music/control {"action":"resume"}'), calls.join(" | "));

window.NixPlaybar.act("next");
await flush();
check("next is a queue skip", calls.includes('POST /api/music/queue {"action":"next"}'), calls.join(" | "));

window.NixPlaybar.act("mute");
await flush();
check("mute is the console's own transport command",
  calls.includes('POST /api/music/control {"action":"mute"}'), calls.join(" | "));

nowPlaying = { ...nowPlaying, player: { state: "playing", muted: true, volume: 0.5 }, playing: true };
await window.NixPlaybar.refresh();
window.NixPlaybar.act("mute");
await flush();
check("a muted speaker is unmuted by the same button",
  calls.includes('POST /api/music/control {"action":"unmute"}'), calls.join(" | "));

/* "Previous" is the track before this one when there is one, and this one from
   the top when there is not. */
const before = calls.length;
window.NixPlaybar.act("prev");
await flush();
await flush();
check("previous plays the queue's earlier track",
  calls.slice(before).some((call) => call === 'POST /api/music/queue {"action":"play","index":1}'),
  calls.slice(before).join(" | "));

window.fetch = async function (url, options) {
  const method = (options && options.method) || "GET";
  const body = options && options.body ? JSON.parse(options.body) : null;
  calls.push(`${method} ${url}${body ? " " + JSON.stringify(body) : ""}`);
  const answer = (payload) => ({ ok: true, status: 200, text: async () => JSON.stringify(payload) });
  if (String(url) === "/api/music/now-playing") return answer(nowPlaying);
  if (String(url).startsWith("/api/music/queue") && method === "GET") {
    return answer({ ok: true, queue: [], index: 0, count: 0, total: 5, revision: 8 });
  }
  return answer({ ok: true });
};
const atStart = calls.length;
window.NixPlaybar.act("prev");
await flush();
await flush();
check("with nothing before it, previous starts this track again",
  calls.slice(atStart).some((call) => call === 'POST /api/music/play {"video_id":"a1"}'),
  calls.slice(atStart).join(" | "));

/* ── the strip gets out of the way when there is nothing to show ───────── */

nowPlaying = { ok: true, track: null, playing: false, player: { state: "idle", muted: false, volume: 0.5 }, position: { elapsed_sec: 0 }, queue_active: false };
await window.NixPlaybar.refresh();
check("nothing playing hides the strip instead of showing an empty one",
  bar.hidden === true && window.NixPlaybar.state().shown === false);

/* A console that cannot be reached must not leave a stale song on every page. */
nowPlaying = {
  ok: true,
  track: { video_id: "b2", title: "Blue in Green", artist: "Miles Davis", duration_sec: 337, artwork_url: "" },
  playing: true, player: { state: "playing", muted: false, volume: 0.5 },
  position: { elapsed_sec: 10 }, queue_active: false,
};
await window.NixPlaybar.refresh();
window.fetch = async () => ({ ok: false, status: 503, text: async () => '{"ok":false}' });
let reported = false;
try { await window.NixPlaybar.refresh(); } catch (error) { reported = true; }
check("a read that fails is reported rather than swallowed", reported);
check("the strip keeps the track it last knew",
  bar.hidden === false && window.NixPlaybar.state().title === "Blue in Green",
  window.NixPlaybar.state().title);

if (failures.length) {
  process.stdout.write(`\n${failures.length} playbar check(s) failed\n`);
  process.exit(1);
}
process.stdout.write("\nplaybar checks passed\n");
// The strip re-arms its poll on purpose (that is how it follows the speaker),
// so this ends the process rather than waiting for an event loop that never
// empties.
process.exit(0);
