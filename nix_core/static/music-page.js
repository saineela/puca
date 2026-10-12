/* NIX PUCA music page shell.
 *
 * The console owns this page, the route and the API. An installed skill that
 * declares device_ui.page fills it: the shell loads that skill's declared
 * script and hands it the NixMusic API surface below, which is the same JSON
 * API the NIX Home app calls under /api/mobile/v1/music.
 *
 * Skill page contract:
 *   window.NixMusicPage = { mount(rootElement, api) }
 *   api.nowPlaying(), api.control(action, value), api.search(query),
 *   api.play({query|video_id}), api.library(kind), api.lyrics(),
 *   api.poll(handler) -> unsubscribe
 */
(function () {
  "use strict";
  const API = "/api/music";
  const state = { page: null, provider: null, player: null, mounted: false, scriptLoaded: false, pollTimer: null };
  const listeners = new Set();

  /* One call, read as text and parsed here. A reply that is not JSON (the
     console restarting mid-request, a truncated or empty body, something else
     answering on the port) then reports its status and a snippet of what
     actually arrived, instead of one opaque sentence that hides the cause. */
  async function once(path, options) {
    const response = await fetch(API + path, Object.assign({ cache: "no-store" }, options || {}));
    const text = await response.text();
    let payload = null;
    try { payload = text ? JSON.parse(text) : null; } catch (_error) { payload = null; }
    if (!payload || typeof payload !== "object") {
      const error = new Error(
        "The music API answered " + response.status + (response.statusText ? " " + response.statusText : "") +
        " with a body this page could not read" +
        (text.trim() ? ": " + text.trim().slice(0, 140) : " (it was empty)") + "."
      );
      error.code = "unreadable_reply";
      error.status = response.status;
      error.unreadable = true;
      throw error;
    }
    if (!response.ok || payload.ok === false) {
      const error = new Error(payload.error || "The music request failed.");
      error.code = payload.code || "music_error";
      error.status = response.status;
      throw error;
    }
    return payload;
  }

  // Reads are idempotent, so a transport failure (a console restart, a dropped
  // connection, a half-written reply) is retried once before it is reported.
  // Writes are never retried: a command may already have reached the speaker.
  async function request(path, options) {
    const signal = options && options.signal;
    try {
      return await once(path, options);
    } catch (error) {
      const isRead = !options || !options.method || String(options.method).toUpperCase() === "GET";
      if (!isRead || error.name === "AbortError" || (signal && signal.aborted)) throw error;
      if (!(error.unreadable || error instanceof TypeError)) throw error;
      await new Promise((resolve) => setTimeout(resolve, 200));
      return once(path, options);
    }
  }

  // Reads accept fetch options so a page can abort a superseded request (for
  // example a search the user typed past) instead of queueing it.
  function get(path, options) { return request(path, options); }
  function post(path, body) {
    return request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  }

  const NixMusic = {
    base: API,
    state,
    get: get,
    post: post,
    capabilities: () => get(""),
    nowPlaying: (options) => get("/now-playing" + (options && options.lyrics ? "?lyrics=1" : ""), options),
    control: (action, value) => post("/control", { action: action, value: value }),
    search: (query, limit, options) => get("/search?q=" + encodeURIComponent(query) + (limit ? "&limit=" + limit : ""), options),
    play: (target) => post("/play", target || {}),
    library: (kind, extra, options) => {
      const params = new URLSearchParams(Object.assign({ kind: kind || "playlists" }, extra || {}));
      return get("/library?" + params.toString(), options);
    },
    lyrics: (extra, options) => {
      const params = new URLSearchParams(extra || {});
      const query = params.toString();
      return get("/lyrics" + (query ? "?" + query : ""), options);
    },
    poll(handler) {
      listeners.add(handler);
      return () => listeners.delete(handler);
    },
    broadcast(snapshot) { listeners.forEach((handler) => { try { handler(snapshot); } catch (_error) { /* a page handler must not break the shell */ } }); },
    formatTime(seconds) {
      if (typeof seconds !== "number" || !isFinite(seconds) || seconds < 0) return "--:--";
      const total = Math.floor(seconds);
      return Math.floor(total / 60) + ":" + String(total % 60).padStart(2, "0");
    }
  };
  window.NixMusic = NixMusic;

  function byId(id) { return document.getElementById(id); }

  function setMessage(root, text, isError) {
    root.innerHTML = "";
    const card = document.createElement("div");
    card.className = "card";
    const message = document.createElement("p");
    message.className = "meta";
    message.textContent = text;
    if (isError) message.style.color = "#904b4b";
    card.appendChild(message);
    root.appendChild(card);
  }

  function loadStyles(urls) {
    (urls || []).forEach((url) => {
      if (document.querySelector('link[data-music-style="' + url + '"]')) return;
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = url;
      link.dataset.musicStyle = url;
      document.head.appendChild(link);
    });
  }

  function loadScript(url) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector('script[data-music-script="' + url + '"]');
      if (existing) { existing.dataset.loaded === "1" ? resolve() : existing.addEventListener("load", () => resolve(), { once: true }); return; }
      const script = document.createElement("script");
      script.src = url;
      script.dataset.musicScript = url;
      script.addEventListener("load", () => { script.dataset.loaded = "1"; resolve(); }, { once: true });
      script.addEventListener("error", () => reject(new Error("The music skill page script could not be loaded.")), { once: true });
      document.head.appendChild(script);
    });
  }

  /* One snapshot endpoint serves everything: reads are de-duplicated into a
     single in-flight request, the poll runs every 1.5s only while something is
     actually playing, backs off to 5s when the speaker is idle, and stops
     completely while another page is open or the tab is hidden. A visible tab
     that regains focus refreshes at once instead of waiting for the next tick. */
  const POLL_PLAYING_MS = 1500;
  const POLL_IDLE_MS = 5000;
  let snapshotInFlight = null;

  function musicViewVisible() {
    const view = byId("music");
    return Boolean(view) && view.classList.contains("active") && !document.hidden;
  }

  function readSnapshot() {
    if (snapshotInFlight) return snapshotInFlight;
    snapshotInFlight = NixMusic.nowPlaying().then((snapshot) => {
      state.player = snapshot.player;
      state.track = snapshot.track;
      NixMusic.broadcast(snapshot);
      return snapshot;
    });
    snapshotInFlight.catch(() => {}).then(() => { snapshotInFlight = null; });
    return snapshotInFlight;
  }

  function reschedulePolling() {
    if (state.pollTimer) clearTimeout(state.pollTimer);
    const playing = Boolean(state.player && state.player.state === "playing");
    state.pollTimer = setTimeout(async () => {
      state.pollTimer = null;
      if (musicViewVisible()) {
        try { await readSnapshot(); } catch (_error) { /* keep the last good snapshot */ }
      }
      reschedulePolling();
    }, playing ? POLL_PLAYING_MS : POLL_IDLE_MS);
  }

  NixMusic.refresh = function refresh() {
    reschedulePolling();
    return readSnapshot();
  };

  async function startPolling() {
    if (state.pollTimer) return;
    reschedulePolling();
  }

  /* The Music entry belongs to an installed music skill: it stays hidden until a
     provider skill is installed, and re-checks itself whenever the skills
     catalog changes (the dashboard calls window.refreshMusicNav). */
  function setMusicNav(visible) {
    const nav = byId("navMusicItem");
    if (nav) nav.hidden = !visible;
  }

  async function refreshMusicNav() {
    try {
      const capabilities = await NixMusic.capabilities();
      setMusicNav(Boolean(capabilities.page && capabilities.provider));
    } catch (_error) {
      setMusicNav(false);
    }
  }
  window.refreshMusicNav = refreshMusicNav;

  /* `play` is reported for a control that starts playback by itself. A speaker
     that plays a stream URL declares `play_url`, and this console never calls
     that as a control - tapping a track goes through POST /api/music/play - so
     calling it "unavailable" would read as a broken player. */
  function playerStateLabel(capabilities) {
    const player = capabilities.player || {};
    const commands = player.commands || {};
    const declared = Object.keys(commands);
    if (!declared.some((name) => commands[name])) {
      return player.skill_id
        ? "This speaker skill declares no player control"
        : "No speaker skill is installed; OpenDeck can still serve a stream";
    }
    const missing = declared.filter((name) => !commands[name] && name !== "play");
    return missing.length
      ? "Player control unavailable for: " + missing.join(", ")
      : "Player control ready";
  }

  async function loadMusicPage(force) {
    const root = byId("musicShell");
    if (!root) return;
    if (state.mounted && !force) return;
    let capabilities;
    try {
      capabilities = await NixMusic.capabilities();
    } catch (error) {
      setMusicNav(false);
      setMessage(root, "The music API is not available on this console: " + error.message, true);
      return;
    }
    state.page = capabilities.page;
    state.provider = capabilities.provider;
    state.player = capabilities.player;
    if (!capabilities.page || !capabilities.provider) {
      setMusicNav(false);
      setMessage(root, "Install a music skill (Skills → install, then configure and trust it) to fill this page. " + (capabilities.error || ""));
      return;
    }
    setMusicNav(true);
    if (byId("musicHeading")) byId("musicHeading").textContent = capabilities.page.title || "Music";
    if (byId("musicBlurb")) {
      byId("musicBlurb").textContent = "Playback on your speaker, provided by " + (capabilities.provider.skill_name || capabilities.provider.skill_id) + ".";
    }
    if (byId("musicSourceState")) {
      byId("musicSourceState").textContent = playerStateLabel(capabilities);
    }
    loadStyles(capabilities.page.styles);
    try {
      await loadScript(capabilities.page.script);
    } catch (error) {
      setMessage(root, error.message, true);
      return;
    }
    if (typeof window.NixMusicPage === "object" && typeof window.NixMusicPage.mount === "function") {
      root.innerHTML = "";
      const container = document.createElement("div");
      container.className = "music-app";
      root.appendChild(container);
      try {
        await window.NixMusicPage.mount(container, NixMusic);
        state.mounted = true;
      } catch (error) {
        setMessage(root, "The music skill page failed to start: " + error.message, true);
      }
    } else {
      setMessage(root, "The installed music skill (" + capabilities.page.skill_id + ") ships a page script but no window.NixMusicPage.mount(). Update the skill.", true);
    }
    startPolling();
  }

  /* Opening the page refreshes immediately: the 13ms snapshot call means the
     user never looks at a stale card while the next poll tick waits. */
  function showMusic(force) {
    loadMusicPage(force);
    if (state.mounted) NixMusic.refresh().catch(() => {});
  }

  window.loadMusicPage = loadMusicPage;
  window.showMusic = showMusic;

  document.addEventListener("DOMContentLoaded", () => {
    const refresh = byId("musicRefresh");
    if (refresh) refresh.addEventListener("click", () => showMusic(true));
    const nav = byId("navMusicItem");
    if (nav) nav.addEventListener("click", () => { showMusic(false); });
    // Returning to a visible tab should not show a snapshot from before it was
    // hidden, so refresh at once rather than waiting for the next poll tick.
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && musicViewVisible()) NixMusic.refresh().catch(() => {});
    });
    // The nav entry appears as soon as a music skill is installed, without a reload.
    refreshMusicNav();
  });
})();
