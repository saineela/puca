/* jsdom behaviour harness for the console's music page shell.
 *
 * The shell's job with covers is JavaScript: it rewrites the provider's absolute
 * artwork URLs to the console's own artwork route before a page script ever sees
 * them, leaves non-http artwork alone, and keeps the provider's URL as
 * artwork_source. That is checked by running the real static/music-page.js in a
 * DOM and driving its NixMusic surface, not by matching strings in the source.
 *
 * jsdom is a test-only dependency of this harness. It is looked up in the usual
 * places, then wherever NIX_JSDOM_ROOT points, so the check runs on a machine
 * that has it installed and reports a skip (exit code 77) where it is absent
 * instead of failing a build for a missing dev tool.
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
    // A global npm install lives in <prefix>/lib/node_modules, and the prefix
    // is not always the one the running node binary was built with (this box
    // runs /usr/bin/node with npm's packages under /usr/local).
    path.join(process.env.npm_config_prefix || process.env.PREFIX || "/usr/local", "lib", "node_modules"),
    path.join(path.dirname(process.execPath), "..", "lib", "node_modules"),
  ].filter(Boolean);
  let failure = "";
  for (const candidate of ["jsdom", ...roots.map((root) => path.join(root, "jsdom"))]) {
    try {
      return { jsdom: require(candidate), failure: "" };
    } catch (error) {
      // No jsdom at that location is the ordinary skip; a jsdom that is there
      // but cannot load means these checks did not run, so carry the reason.
      if (error && error.code !== "MODULE_NOT_FOUND") {
        failure = String(error.message).split("\n")[0];
      }
    }
  }
  return { jsdom: null, failure };
}

const { jsdom, failure } = loadJsdom();
if (!jsdom) {
  process.stdout.write(
    `skip the music shell DOM checks: jsdom is not installed (set NIX_JSDOM_ROOT)${failure ? `: ${failure}` : ""}\n`,
  );
  process.exit(77);
}
const { JSDOM } = jsdom;

const here = path.dirname(fileURLToPath(import.meta.url));
const source = readFileSync(path.join(here, "static", "music-page.js"), "utf8");
const dom = new JSDOM("<!doctype html><div id='music'></div>", {
  runScripts: "outside-only",
  url: "http://127.0.0.1:49117/",
});
const { window } = dom;

const COVER = "https://lh3.googleusercontent.com/cover=w544-h544-l90-rj";
const LOCAL = "/api/music/artwork?src=" + encodeURIComponent(COVER);
const answers = {
  "/api/music": { ok: true, page: { id: "music" }, provider: { skill_id: "opendeck" } },
  "/api/music/library?kind=playlists": {
    ok: true,
    playlists: [{ playlist_id: "LM", name: "Liked Music", track_count: 3, artwork_url: COVER }],
  },
  "/api/music/search?q=Miles": {
    ok: true,
    tracks: [{ video_id: "a1", title: "So What", artist: "Miles", duration_sec: 545, artwork_url: COVER }],
    playlists: [],
  },
  "/api/music/now-playing": {
    ok: true,
    track: { video_id: "a1", title: "So What", artist: "Miles", artwork_url: COVER },
    player: { state: "playing" },
  },
};

window.fetch = async (url) => {
  const body = answers[String(url)];
  if (!body) return { ok: false, status: 404, text: async () => '{"ok":false}' };
  return { ok: true, status: 200, text: async () => JSON.stringify(body) };
};

window.eval(source);
const api = window.NixMusic;
const failures = [];
function check(name, condition, detail) {
  if (condition) process.stdout.write(`  ok   ${name}\n`);
  else {
    failures.push(name);
    process.stdout.write(`  FAIL ${name}${detail === undefined ? "" : ` — ${detail}`}\n`);
  }
}

const playlists = await api.library("playlists");
const search = await api.search("Miles");
const snapshot = await api.nowPlaying();

check("a playlist cover is served by the console", playlists.playlists[0].artwork_url === LOCAL, playlists.playlists[0].artwork_url);
check("the provider's playlist URL is kept as artwork_source", playlists.playlists[0].artwork_source === COVER);
check("a search result cover is served by the console", search.tracks[0].artwork_url === LOCAL, search.tracks[0].artwork_url);
check("the now-playing cover is served by the console", snapshot.track.artwork_url === LOCAL, snapshot.track.artwork_url);
check("api.artwork builds the same local URL", api.artwork(COVER) === LOCAL);
check("artwork that is not an http URL is left alone", api.artwork("") === "" && api.artwork("data:image/png;base64,AA") === "data:image/png;base64,AA");
check("a payload without artwork is unchanged", JSON.stringify(await api.capabilities()) === JSON.stringify(answers["/api/music"]));

if (failures.length) {
  process.stdout.write(`\n${failures.length} music shell check(s) failed\n`);
  process.exit(1);
}
process.stdout.write("\nmusic shell checks passed\n");
