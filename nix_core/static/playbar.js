/* The mini playbar — the track that is playing, on every page of the console.
 *
 * The console is a set of pages (Home, Models, Skills, the Music page the
 * installed skill fills…). Music does not belong to any one of them: once
 * something is playing, you want its name and a way to skip it wherever you
 * are. This script builds that strip and looks after it — it is included once,
 * by the dashboard shell, and it draws itself, so no page has to declare it.
 *
 *   collapsed   a small pill: the track's name, and a thin line for the
 *               position in the song
 *   expanded    the pill opens upward when the cursor reaches it (or the strip
 *               is focused): pause/resume, next, mute, and the way back to the
 *               full Music page
 *
 * Reads:  GET  /api/music/now-playing      what is playing, and where we are
 * Writes: POST /api/music/control          pause · resume · mute · unmute
 *         POST /api/music/queue            next
 *
 * Nothing here decides how the console looks: sizes, colours and the expand
 * animation are playbar.css's. The polling is deliberately modest — five
 * seconds while a song is playing, and it stops entirely on a hidden tab.
 */
(function () {
  'use strict';

  var MUSIC = '/api/music';
  // How often the console is asked what is playing. The position is drawn from
  // the local clock between reads, so this only has to catch a track changing.
  var POLL_PLAYING_MS = 5000;
  var POLL_IDLE_MS = 15000;
  var TICK_MS = 1000;

  var S = {
    shown: false,
    payload: null,
    track: null,
    playing: false,
    muted: false,
    duration: 0,
    elapsed: 0,
    startedAt: 0,
    queueActive: false,
    lastRead: 0,
    timer: null,
    tick: null,
    busy: false
  };

  var nodes = {};

  function $(id) { return document.getElementById(id); }

  function esc(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function fmtTime(seconds) {
    if (!isFinite(seconds) || seconds < 0) seconds = 0;
    var m = Math.floor(seconds / 60), s = Math.floor(seconds % 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
  }

  function request(path, options) {
    return window.fetch(MUSIC + path, options).then(function (response) {
      return response.text().then(function (text) {
        var payload = {};
        try { payload = text ? JSON.parse(text) : {}; } catch (error) { payload = {}; }
        if (!response.ok) {
          var failure = new Error(payload.error || payload.message || ('HTTP ' + response.status));
          failure.payload = payload;
          throw failure;
        }
        return payload;
      });
    });
  }

  function get(path) { return request(path, { cache: 'no-store' }); }

  function post(path, body) {
    return request(path, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body || {})
    });
  }

  /* ── the strip ─────────────────────────────────────────────────────────── */

  function build() {
    if (nodes.bar) return;
    var bar = document.createElement('div');
    bar.className = 'playbar';
    bar.id = 'nixPlaybar';
    bar.hidden = true;
    // A strip the cursor can reach without having to hit a small target: the
    // whole bar expands while the pointer is anywhere over it.
    bar.setAttribute('role', 'region');
    bar.setAttribute('aria-label', 'Now playing');
    bar.innerHTML =
      '<div class="playbar-panel">' +
      '  <button class="playbar-btn" type="button" data-playbar="prev" title="Restart this track" aria-label="Restart this track">' +
      '    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 5v14L7 12zM6 5v14" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>' +
      '  </button>' +
      '  <button class="playbar-btn playbar-play" type="button" data-playbar="toggle" title="Play or pause" aria-label="Play or pause">' +
      '    <svg class="ic-play" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4.5v15l13-7.5z" fill="currentColor"/></svg>' +
      '    <svg class="ic-pause" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4h4v16H7zM13 4h4v16h-4z" fill="currentColor"/></svg>' +
      '  </button>' +
      '  <button class="playbar-btn" type="button" data-playbar="next" title="Next track" aria-label="Next track">' +
      '    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5v14l11-7zM18 5v14" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>' +
      '  </button>' +
      '  <button class="playbar-btn" type="button" data-playbar="mute" title="Mute" aria-label="Mute">' +
      '    <svg class="ic-vol" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9.5h4L13 5v14l-5-4.5H4z" fill="currentColor"/><path d="M16.5 9a4.5 4.5 0 0 1 0 6" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/><path d="M19 6.5a8 8 0 0 1 0 11" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>' +
      '    <svg class="ic-muted" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9.5h4L13 5v14l-5-4.5H4z" fill="currentColor"/><path d="M17 9.5l4.5 5M21.5 9.5l-4.5 5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>' +
      '  </button>' +
      '</div>' +
      '<button class="playbar-strip" id="playbarStrip" type="button" title="Open the Music page" aria-label="Now playing — open the Music page">' +
      '  <span class="playbar-art" id="playbarArt" aria-hidden="true">♪</span>' +
      '  <span class="playbar-copy">' +
      '    <span class="playbar-title" id="playbarTitle">Nothing playing</span>' +
      '    <span class="playbar-artist" id="playbarArtist"></span>' +
      '  </span>' +
      '  <span class="playbar-where" id="playbarWhere"></span>' +
      '  <span class="playbar-times" id="playbarTimes">0:00</span>' +
      '</button>' +
      '<span class="playbar-progress" id="playbarProgress"><i id="playbarFill"></i></span>';
    document.body.appendChild(bar);
    nodes.bar = bar;
    nodes.panel = bar.querySelector('.playbar-panel');
    nodes.strip = bar.querySelector('#playbarStrip');
    nodes.art = bar.querySelector('#playbarArt');
    nodes.title = bar.querySelector('#playbarTitle');
    nodes.artist = bar.querySelector('#playbarArtist');
    nodes.where = bar.querySelector('#playbarWhere');
    nodes.times = bar.querySelector('#playbarTimes');
    nodes.fill = bar.querySelector('#playbarFill');

    bar.addEventListener('click', function (event) {
      var control = event.target.closest ? event.target.closest('[data-playbar]') : null;
      if (control) {
        event.preventDefault();
        act(control.getAttribute('data-playbar'));
        return;
      }
      if (event.target.closest('#playbarStrip')) openMusic();
    });
    // Focus arriving from the keyboard opens the controls too: a strip that only
    // expands under a cursor is unreachable with a Tab key.
    bar.addEventListener('focusin', function () { bar.classList.add('open'); });
    bar.addEventListener('focusout', function (event) {
      if (!bar.contains(event.relatedTarget)) bar.classList.remove('open');
    });
  }

  function openMusic() {
    var nav = document.getElementById('navMusicItem');
    if (nav && !nav.hidden) { nav.click(); return; }
    window.location.href = '/music';
  }

  function act(name) {
    if (S.busy) return;
    var call;
    if (name === 'toggle') {
      call = post('/control', { action: S.playing ? 'pause' : 'resume' });
    } else if (name === 'next') {
      call = post('/queue', { action: 'next' });
    } else if (name === 'mute') {
      call = post('/control', { action: S.muted ? 'unmute' : 'mute' });
    } else if (name === 'prev') {
      // "Previous" means the track before this one in the queue — and this one
      // from the top when there is nothing before it. The queue is read at the
      // moment it is needed rather than on every poll, so the strip still costs
      // one request per turn.
      call = get('/queue?limit=1').then(function (state) {
        var index = state && typeof state.index === 'number' ? state.index : -1;
        if (state && state.ok && index > 0) {
          return post('/queue', { action: 'play', index: index - 1 });
        }
        return post('/play', { video_id: (S.track || {}).video_id || '' });
      });
    } else {
      return;
    }
    S.busy = true;
    nodes.bar.classList.add('busy');
    call.then(function () {
      return refresh();
    }).catch(function () {
      // A failed control leaves the strip as it was; the next poll is the truth.
    }).then(function () {
      S.busy = false;
      nodes.bar.classList.remove('busy');
    });
  }

  /* ── painting ──────────────────────────────────────────────────────────── */

  function position() {
    if (!S.playing) return S.elapsed;
    var elapsed = S.elapsed + (Date.now() - S.lastRead) / 1000;
    if (S.duration) elapsed = Math.min(elapsed, S.duration);
    return Math.max(0, elapsed);
  }

  function paint() {
    if (!nodes.bar) return;
    var track = S.track;
    if (!track || !(track.title || track.video_id)) {
      nodes.bar.hidden = true;
      S.shown = false;
      return;
    }
    nodes.bar.hidden = false;
    S.shown = true;
    nodes.title.textContent = track.title || track.video_id;
    nodes.artist.textContent = track.artist || '';
    var art = track.artwork_url;
    if (art) {
      nodes.art.innerHTML = '<img alt="" src="' + esc(art) + '" />';
    } else {
      nodes.art.textContent = '♪';
    }
    nodes.where.textContent = S.queueActive ? 'Queue' : '';
    nodes.bar.classList.toggle('playing', S.playing);
    nodes.bar.classList.toggle('muted', S.muted);
    paintClock();
  }

  function paintClock() {
    if (!nodes.bar) return;
    var elapsed = position();
    nodes.times.textContent = fmtTime(elapsed);
    var percent = S.duration ? Math.min(100, (elapsed / S.duration) * 100) : 0;
    if (nodes.fill) nodes.fill.style.width = percent.toFixed(2) + '%';
    nodes.bar.setAttribute('aria-label',
      (S.playing ? 'Playing ' : 'Paused ') + (nodes.title.textContent || '') +
      (nodes.artist.textContent ? ' by ' + nodes.artist.textContent : '') +
      (S.duration ? ' — ' + fmtTime(elapsed) + ' of ' + fmtTime(S.duration) : ''));
  }

  function apply(payload) {
    S.payload = payload || {};
    S.track = S.payload.track && S.payload.track.video_id ? S.payload.track : null;
    S.playing = Boolean(S.payload.playing);
    var player = S.payload.player || {};
    S.muted = Boolean(player.muted);
    S.duration = Number((S.track || {}).duration_sec) || 0;
    var where = S.payload.position || {};
    S.elapsed = Number(where.elapsed_sec) || 0;
    S.lastRead = Date.now();
    S.queueActive = Boolean(S.payload.queue_active);
    paint();
  }

  function refresh() {
    return get('/now-playing').then(function (payload) {
      apply(payload);
      return payload;
    });
  }

  /* ── the poll ──────────────────────────────────────────────────────────── */

  function arm() {
    if (S.timer) window.clearTimeout(S.timer);
    var wait = S.playing ? POLL_PLAYING_MS : POLL_IDLE_MS;
    S.timer = window.setTimeout(function () {
      refresh().catch(function () { /* a console that is restarting is not an error to show */ })
        .then(arm);
    }, wait);
    if (!S.tick) {
      S.tick = window.setInterval(function () {
        if (S.playing && !document.hidden) paintClock();
      }, TICK_MS);
    }
  }

  function awake() {
    // Coming back to the tab asks straight away: the song may have changed
    // several times while the page was in the background.
    if (!document.hidden) { refresh().catch(function () {}).then(arm); }
  }

  build();
  refresh().catch(function () { /* nothing playing yet is the ordinary case */ }).then(arm);
  document.addEventListener('visibilitychange', awake);

  // For the behaviour harness: the same surface the music page's shell exposes.
  window.NixPlaybar = {
    refresh: refresh,
    state: function () {
      return {
        shown: S.shown,
        playing: S.playing,
        muted: S.muted,
        queueActive: S.queueActive,
        title: S.track ? (S.track.title || S.track.video_id) : '',
        artist: S.track ? (S.track.artist || '') : '',
        elapsed: position(),
        duration: S.duration
      };
    },
    act: act
  };
})();
