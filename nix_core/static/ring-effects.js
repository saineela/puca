/**
 * ring-effects.js — the Echo Dot's ring effect engine for the web.
 * Ported motion-for-motion from the NIX Home Android app's ui/RingEffects.kt
 * (which was ported from the device firmware). Every advertised effect, with
 * its real palettes, periods and frame maths. `frame()` is pure; renderers
 * sample it at the screen's frame rate.
 */
(function (root) {
  const SEGMENTS = 12;
  const EFFECTS = [
    'Pulse', 'Heartbeat', 'Ripple', 'Standing Wave', 'Twinkle',
    'Crimson Heartbeat', 'Aurora Pulse', 'Candle', 'Fireplace', 'Embers',
    'Aurora', 'Sunset Drift', 'Ocean Ripple', 'Rainbow Twinkle', 'Forest Twinkle',
    'Comet', 'Chase', 'Scanner', 'Pinwheel', 'Spiral', 'Wipe',
    'Helix', 'Orbits', 'Bounce', 'Spring',
    'Rainbow', 'Fire Comet', 'Ice Comet', 'Rainbow Chase', 'Ice Scanner',
    'Rainbow Pinwheel', 'Sunset Spiral', 'Rainbow Orbits', 'DNA', 'Pac-Man',
    'Alert', 'Beacon',
  ];

  function rgb(hex) {
    const n = parseInt(String(hex).replace('#', ''), 16);
    return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
  }
  function css(c, alpha) {
    const r = Math.round(clamp01(c[0]) * 255);
    const g = Math.round(clamp01(c[1]) * 255);
    const b = Math.round(clamp01(c[2]) * 255);
    return alpha === undefined ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},${alpha})`;
  }
  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
  function scale(c, f) { return [clamp01(c[0] * f), clamp01(c[1] * f), clamp01(c[2] * f)]; }
  function add(a, b) {
    return [Math.min(1, a[0] + b[0]), Math.min(1, a[1] + b[1]), Math.min(1, a[2] + b[2])];
  }
  function blend(a, b, f) {
    const g = clamp01(f);
    return [a[0] + (b[0] - a[0]) * g, a[1] + (b[1] - a[1]) * g, a[2] + (b[2] - a[2]) * g];
  }
  function hue(h) {
    const x = (((h * 6) % 6) + 6) % 6;
    const ramp = clamp01;
    switch (Math.floor(x)) {
      case 0: return [1, ramp(x), 0];
      case 1: return [ramp(2 - x), 1, 0];
      case 2: return [0, 1, ramp(x - 2)];
      case 3: return [0, ramp(4 - x), 1];
      case 4: return [ramp(x - 4), 0, 1];
      default: return [1, 0, ramp(6 - x)];
    }
  }

  const wheel = Array.from({ length: 6 }, (_, i) => scale(hue(i / 6), 0.6));
  const flame = [rgb('#FF9329')];
  const fire = ['#FFE0A0', '#FF9020', '#D03000', '#400400'].map(rgb);
  const crimson = ['#280004', '#800008', '#D00014', '#FF4048'].map(rgb);
  const aurora = ['#00C850', '#00B0B0', '#3040C0', '#7020A0', '#109080'].map(rgb);
  const ocean = ['#000C40', '#0050A0', '#00B0C0', '#90F0E0'].map(rgb);
  const ice = ['#081850', '#1060C0', '#60C0F0', '#E0F8FF'].map(rgb);
  const sunset = ['#300860', '#A01060', '#E05020', '#FFB030'].map(rgb);
  const forest = ['#042810', '#107020', '#40B020', '#A0E040'].map(rgb);
  const alarm = ['#300000', '#C00000', '#FF3018', '#FFB0A0'].map(rgb);
  const duo = ['#00E0C0', '#E000A0'].map(rgb);
  const pellets = ['#FFD000', '#303038'].map(rgb);

  function palAt(p, x) {
    if (p.length === 0) return [0, 0, 0];
    if (p.length === 1) return p[0];
    const xf = (((x % 1) + 1) % 1) * p.length;
    const i = Math.min(p.length - 1, Math.floor(xf));
    return blend(p[i], p[(i + 1) % p.length], xf - i);
  }
  function palAlong(p, x) {
    if (p.length < 2) return palAt(p, x);
    const xf = clamp01(x) * (p.length - 1);
    const i = Math.floor(xf);
    if (i >= p.length - 1) return p[p.length - 1];
    return blend(p[i], p[i + 1], xf - i);
  }
  function palShade(p, x, dim) {
    const d = p.length > 1 ? 0.5 + 0.5 * dim : dim;
    return scale(palAlong(p, x), d);
  }
  function palNth(p, i) { return p[((i % p.length) + p.length) % p.length]; }

  function around(p, f) {
    return Array.from({ length: SEGMENTS }, (_, i) => scale(palAt(p, i / SEGMENTS), f));
  }
  function shaded(p, f) {
    return Array.from({ length: SEGMENTS }, () => palShade(p, f, f));
  }
  function ringDist(i, at) {
    const norm = ((at % SEGMENTS) + SEGMENTS) % SEGMENTS;
    const d = Math.abs(i - norm);
    return Math.min(d, SEGMENTS - d);
  }
  function dot(out, at, c, width) {
    for (let i = 0; i < SEGMENTS; i++) {
      const d = ringDist(i, at);
      if (d >= width) continue;
      const f = 1 - d / width;
      out[i] = add(out[i], scale(c, f * f));
    }
  }
  function wander(t, offset) {
    const s =
      Math.sin(2 * Math.PI * (1.7 * t + offset)) +
      Math.sin(2 * Math.PI * (2.9 * t + 1.3 * offset)) +
      Math.sin(2 * Math.PI * (4.7 * t + 0.6 * offset));
    return (s / 3 + 1) / 2;
  }
  function hash(input) {
    let x = input | 0;
    x ^= x >>> 16;
    x = Math.imul(x, 0x7FEB352D);
    x ^= x >>> 15;
    x = Math.imul(x, 0x846CA68B);
    x ^= x >>> 16;
    return x;
  }
  function arc(fraction, c) {
    const f = clamp01(fraction);
    const lit = f * SEGMENTS;
    const out = Array.from({ length: SEGMENTS }, () => [0, 0, 0]);
    for (let i = 0; i < SEGMENTS; i++) {
      const remaining = lit - i;
      if (remaining <= 0) break;
      out[(SEGMENTS - 1 + i) % SEGMENTS] = scale(c, Math.min(1, remaining));
    }
    return out;
  }

  function breathe(p, t) {
    const period = 1200;
    const f = (1 - Math.cos(2 * Math.PI * (t % period) / period)) / 2;
    return around(p, f);
  }
  function heartbeat(p, t) {
    const period = 1400, decay = 170;
    const beats = [[0, 1], [200, 0.6]];
    const tt = t % period;
    let f = 0;
    for (const [at, amp] of beats) {
      if (tt < at) continue;
      f = Math.max(f, amp * Math.exp(-(tt - at) / decay));
    }
    return shaded(p, f);
  }
  function ripple(p, t) {
    const revolution = 3200, crests = 2, glow = 0.12;
    const phase = (t % revolution) / revolution;
    return Array.from({ length: SEGMENTS }, (_, i) => {
      let c = (1 + Math.cos(2 * Math.PI * crests * (i / SEGMENTS - phase))) / 2;
      c = c * c * c;
      return palShade(p, c, glow + (1 - glow) * c);
    });
  }
  function standing(p, t) {
    const period = 2600, crests = 2, glow = 0.08;
    const swell = Math.cos(2 * Math.PI * (t % period) / period);
    return Array.from({ length: SEGMENTS }, (_, i) => {
      const f = Math.abs(Math.sin(2 * Math.PI * crests * i / SEGMENTS) * swell);
      return palShade(p, f, glow + (1 - glow) * f);
    });
  }
  function twinkle(p, t) {
    const shortest = 1700, spread = 1300, duty = 0.35, dark = 0.05;
    return Array.from({ length: SEGMENTS }, (_, i) => {
      const h = hash(i - 1640531527);
      const period = shortest + spread * (h & 0xFFFF) / 65536;
      const phase = (h >>> 16) / 65536;
      const color = palAt(p, phase);
      let f = dark;
      const x = ((t / period) + phase) % 1;
      if (x < duty) f = Math.max(dark, Math.sin(Math.PI * x / duty));
      return scale(color, f);
    });
  }
  function flicker(p, t) {
    const ts = t / 1000;
    const whole = 0.75 + 0.25 * wander(ts, 0);
    return Array.from({ length: SEGMENTS }, (_, i) => {
      const w = wander(ts, i * 0.7);
      const f = whole * (0.8 + 0.2 * w);
      return scale(palAlong(p, 1 - w), Math.min(1, f));
    });
  }
  function embers(p, t) {
    const out = flicker(p, t);
    const every = 9000, lasts = 260;
    for (let i = 0; i < SEGMENTS; i++) {
      const h = hash(i + 1374496513);
      const phase = (h >>> 0) / 4294967296;
      const x = ((t / every) + phase) % 1;
      const window = lasts / every;
      if (x >= window) continue;
      let f = x / window;
      f = Math.sin(Math.PI * Math.pow(f, 0.7));
      out[i] = add(out[i], scale(palAlong(p, 0), 0.7 * f));
    }
    return out;
  }
  function drift(p, t) {
    const ts = t / 1000;
    return Array.from({ length: SEGMENTS }, (_, i) => {
      const pos = i / SEGMENTS;
      const x = pos + 0.03 * ts +
        0.10 * Math.sin(2 * Math.PI * (0.07 * ts + pos)) +
        0.06 * Math.sin(2 * Math.PI * (0.13 * ts + 2 * pos));
      const f = 0.45 + 0.55 * (0.5 + 0.5 * Math.sin(2 * Math.PI * (0.11 * ts + 1.5 * pos)));
      return scale(palAt(p, x), f);
    });
  }
  function comet(p, t) {
    const perSegment = 80;
    const tail = [1.0, 0.55, 0.3, 0.16, 0.08];
    const head = Math.floor(t / perSegment) % SEGMENTS;
    const out = Array.from({ length: SEGMENTS }, () => [0, 0, 0]);
    for (let i = 0; i < tail.length; i++) {
      const c = palShade(p, i / (tail.length - 1), tail[i]);
      out[((head - i) % SEGMENTS + SEGMENTS) % SEGMENTS] = c;
    }
    return out;
  }
  function chase(p, t) {
    const step = 110, spacing = 3;
    const out = Array.from({ length: SEGMENTS }, () => [0, 0, 0]);
    const off = Math.floor(t / step) % spacing;
    for (let lamp = 0; lamp < SEGMENTS / spacing; lamp++) {
      out[off + lamp * spacing] = palNth(p, lamp);
    }
    return out;
  }
  function scanner(p, t) {
    const sweep = 900, width = 2.2;
    let x = (t / sweep) % 2;
    if (x > 1) x = 2 - x;
    const head = x * (SEGMENTS - 1);
    return Array.from({ length: SEGMENTS }, (_, i) => {
      const d = ringDist(i, head);
      if (d >= width) return [0, 0, 0];
      const f = 1 - d / width;
      return palShade(p, d / width, f * f);
    });
  }
  function pinwheel(p, t) {
    const revolution = 1600, arms = 4;
    const span = SEGMENTS / arms;
    const at = (t % revolution) / revolution * SEGMENTS;
    return Array.from({ length: SEGMENTS }, (_, i) => {
      const turns = (i - at) / span;
      const arm = Math.round(turns);
      const d = Math.abs(turns - arm) * span;
      return d < 1 ? scale(palNth(p, arm), 1 - d) : [0, 0, 0];
    });
  }
  function spiral(p, t) {
    const revolution = 2200;
    const phase = (t % revolution) / revolution;
    return Array.from({ length: SEGMENTS }, (_, i) => {
      const behind = (((i / SEGMENTS - phase + 1) % 1) + 1) % 1;
      const f = 1 - behind;
      return palShade(p, f, f);
    });
  }
  function wipe(p, t) {
    const period = 2400;
    let x = 2 * (t % period) / period;
    if (x > 1) x = 2 - x;
    return arc(x, palAlong(p, x));
  }
  function helix(p, t) {
    const revolution = 2400;
    const phase = (t % revolution) / revolution * SEGMENTS;
    const out = Array.from({ length: SEGMENTS }, () => [0, 0, 0]);
    dot(out, phase, palNth(p, 0), 1.6);
    dot(out, -phase, palNth(p, 1), 1.6);
    return out;
  }
  function orbits(p, t) {
    const speeds = [1.0, -0.62, 0.37];
    const revolution = 2600;
    const tt = t / revolution;
    const out = Array.from({ length: SEGMENTS }, () => [0, 0, 0]);
    for (let i = 0; i < speeds.length; i++) {
      const pos = (((tt * speeds[i]) % 1) + 1) % 1 * SEGMENTS;
      dot(out, pos, palNth(p, i), 1.4);
    }
    return out;
  }
  const bounceArcs = (() => {
    const arcs = [];
    const first = 900, keeps = 0.55, spent = 0.04;
    let cycle = 0, h = 1.0;
    while (h > spent) {
      const d = first * Math.sqrt(h);
      arcs.push([cycle, d, h]);
      cycle += d;
      h *= keeps;
    }
    return arcs;
  })();
  const bounceCycle = bounceArcs.reduce((s, a) => s + a[1], 0) + 700;
  function bounce(p, t) {
    const tt = t % bounceCycle;
    let at = 0;
    for (const [start, d, height] of bounceArcs) {
      if (tt < start || tt >= start + d) continue;
      const x = 2 * (tt - start) / d - 1;
      at = height * (1 - x * x) * (SEGMENTS - 1);
      break;
    }
    const out = Array.from({ length: SEGMENTS }, () => [0, 0, 0]);
    dot(out, at, palAlong(p, at / (SEGMENTS - 1)), 1.5);
    return out;
  }
  function spring(p, t) {
    const half = 1300, rings = 1.5, decay = 3.0;
    const stretched = 0.72, slack = 0.12;
    const step = (from, to, x) =>
      to + (from - to) * Math.exp(-decay * x) * Math.cos(2 * Math.PI * rings * x);
    const tt = t % (2 * half);
    const raw = tt < half
      ? step(slack, stretched, tt / half)
      : step(stretched, slack, (tt - half) / half);
    const x = clamp01(raw);
    return arc(x, palAlong(p, x));
  }
  function spin(p, t) {
    const revolution = 1500;
    const phase = (t % revolution) / revolution;
    return Array.from({ length: SEGMENTS }, (_, i) => palAt(p, i / SEGMENTS + phase));
  }
  function pacman(p, t) {
    const lap = 3600, chomp = 220;
    const at = (t % lap) / lap * SEGMENTS;
    const out = Array.from({ length: SEGMENTS }, () => [0, 0, 0]);
    for (let i = 0; i < SEGMENTS; i++) {
      if (i % 2 === 0 && i > at) out[i] = palNth(p, 1);
    }
    const open = Math.abs(Math.sin(Math.PI * (t % chomp) / chomp));
    dot(out, at, scale(palNth(p, 0), 0.55 + 0.45 * (1 - open)), 1.3);
    return out;
  }
  function alert(p, t) {
    const period = 2000, pulse = 130, gap = 90;
    const tt = t % period;
    let f = 0;
    for (let i = 0; i < 3; i++) {
      const at = i * (pulse + gap);
      if (tt >= at && tt < at + pulse) f = Math.sin(Math.PI * (tt - at) / pulse);
    }
    return shaded(p, f);
  }
  function beacon(p, t) {
    const revolution = 900, width = 2.5;
    const at = (t % revolution) / revolution * SEGMENTS;
    return Array.from({ length: SEGMENTS }, (_, i) => {
      const d = ringDist(i, at);
      if (d >= width) return [0, 0, 0];
      const f = 1 - d / width;
      return palShade(p, f, f * f);
    });
  }

  function frame(effect, elapsedMs, base, brightness) {
    const name = String(effect == null ? '' : effect).trim();
    const p1 = [base];
    const t = elapsedMs;
    let out;
    switch (name.toLowerCase()) {
      case '': case 'none':
        out = Array.from({ length: SEGMENTS }, () => base); break;
      case 'pulse': out = breathe(p1, t); break;
      case 'heartbeat': out = heartbeat(p1, t); break;
      case 'ripple': out = ripple(p1, t); break;
      case 'standing wave': out = standing(p1, t); break;
      case 'twinkle': out = twinkle(p1, t); break;
      case 'crimson heartbeat': out = heartbeat(crimson, t); break;
      case 'aurora pulse': out = breathe(aurora, t); break;
      case 'candle': out = flicker(flame, t); break;
      case 'fireplace': out = flicker(fire, t); break;
      case 'embers': out = embers(fire, t); break;
      case 'aurora': out = drift(aurora, t); break;
      case 'sunset drift': out = drift(sunset, t); break;
      case 'ocean ripple': out = ripple(ocean, t); break;
      case 'rainbow twinkle': out = twinkle(wheel, t); break;
      case 'forest twinkle': out = twinkle(forest, t); break;
      case 'comet': out = comet(p1, t); break;
      case 'chase': out = chase(p1, t); break;
      case 'scanner': out = scanner(p1, t); break;
      case 'pinwheel': out = pinwheel(p1, t); break;
      case 'spiral': out = spiral(p1, t); break;
      case 'wipe': out = wipe(p1, t); break;
      case 'helix': out = helix(p1, t); break;
      case 'orbits': out = orbits(p1, t); break;
      case 'bounce': out = bounce(p1, t); break;
      case 'spring': out = spring(p1, t); break;
      case 'rainbow': out = spin(wheel, t); break;
      case 'fire comet': out = comet(fire, t); break;
      case 'ice comet': out = comet(ice, t); break;
      case 'rainbow chase': out = chase(wheel, t); break;
      case 'ice scanner': out = scanner(ice, t); break;
      case 'rainbow pinwheel': out = pinwheel(wheel, t); break;
      case 'sunset spiral': out = spiral(sunset, t); break;
      case 'rainbow orbits': out = orbits(wheel, t); break;
      case 'dna': out = helix(duo, t); break;
      case 'pac-man': out = pacman(pellets, t); break;
      case 'alert': out = alert(alarm, t); break;
      case 'beacon': out = beacon(alarm, t); break;
      default: out = breathe(p1, t);
    }
    const b = clamp01(brightness === undefined ? 1 : brightness);
    return b >= 0.999 ? out : out.map(c => scale(c, b));
  }
  function meanColor(segs) {
    let r = 0, g = 0, b = 0;
    for (const c of segs) { r += c[0]; g += c[1]; b += c[2]; }
    const n = Math.max(1, segs.length);
    return [clamp01(r / n), clamp01(g / n), clamp01(b / n)];
  }
  function meanLuma(segs) {
    let l = 0;
    for (const c of segs) l += 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
    return clamp01(l / Math.max(1, segs.length));
  }
  function isEffect(name) {
    return name != null && String(name).trim() !== '' && !/^none$/i.test(String(name).trim());
  }
  function createClock() {
    const t0 = performance.now();
    return () => performance.now() - t0;
  }

  const DEG = Math.PI / 180;
  function drawSegmentRing(ctx, segs, cx, cy, radius, stroke, gapDeg) {
    const gap = gapDeg === undefined ? 5 : gapDeg;
    const sweep = 360 / SEGMENTS;
    for (let i = 0; i < SEGMENTS; i++) {
      ctx.beginPath();
      ctx.strokeStyle = css(segs[i] || [0, 0, 0]);
      ctx.lineWidth = stroke;
      ctx.arc(cx, cy, radius,
        (-90 + i * sweep + gap / 2) * DEG,
        (-90 + i * sweep + sweep - gap / 2) * DEG);
      ctx.stroke();
    }
  }
  function fitCanvas(canvas) {
    const dpr = Math.min(3, (root.devicePixelRatio || 1));
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    return { w, h, dpr };
  }
  function render(canvas, state, clock) {
    const { w, h } = fitCanvas(canvas);
    const ctx = canvas.getContext('2d');
    const on = state.connected !== false && state.on !== false;
    const base = rgb(state.base || '#7CFF6B');
    const brightness = state.brightness === undefined ? 1 : state.brightness;
    const effect = on && isEffect(state.effect) ? state.effect : null;
    const segs = effect ? frame(effect, clock(), base, brightness) : null;
    ctx.fillStyle = '#0B0B0F';
    ctx.fillRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2;
    const pad = Math.min(cx, cy) * 0.15;
    const radius = Math.min(cx, cy) - pad;
    if (on) {
      const glowColor = segs ? meanColor(segs) : base;
      const glowLuma = segs ? meanLuma(segs) : 1;
      const alpha = 0.30 * (0.18 + 0.82 * glowLuma);
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius * 1.5);
      g.addColorStop(0, css(glowColor, alpha));
      g.addColorStop(1, css(glowColor, 0));
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(cx, cy, radius * 1.5, 0, Math.PI * 2); ctx.fill();
    }
    if (segs) {
      drawSegmentRing(ctx, segs, cx, cy, radius, Math.max(8, radius * 0.12));
    } else {
      ctx.beginPath();
      ctx.strokeStyle = on ? css(base) : '#2A2A31';
      ctx.lineWidth = Math.max(8, radius * 0.12);
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();
    }
    const bodyR = radius * 0.82;
    const bg = ctx.createRadialGradient(cx - radius * 0.3, cy - radius * 0.35, bodyR * 0.1, cx, cy, radius * 1.05);
    bg.addColorStop(0, '#2E2E36');
    bg.addColorStop(1, '#131318');
    ctx.fillStyle = bg;
    ctx.beginPath(); ctx.arc(cx, cy, bodyR, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = on ? css(base) : '#3A3A44';
    ctx.beginPath(); ctx.arc(cx, cy - radius * 0.55, Math.max(3, radius * 0.045), 0, Math.PI * 2); ctx.fill();
  }
  function renderGlyph(canvas, state, clock) {
    const { w, h } = fitCanvas(canvas);
    const ctx = canvas.getContext('2d');
    const on = state.connected !== false && state.on !== false;
    const base = rgb(state.base || '#7CFF6B');
    const brightness = state.brightness === undefined ? 1 : state.brightness;
    const effect = on && isEffect(state.effect) ? state.effect : null;
    const segs = effect ? frame(effect, clock(), base, brightness) : null;
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2;
    const r = Math.min(w, h) / 2 - 2;
    if (segs) {
      const glow = meanColor(segs);
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 1.35);
      g.addColorStop(0, css(glow, 0.40 * meanLuma(segs)));
      g.addColorStop(1, css(glow, 0));
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(cx, cy, r * 1.35, 0, Math.PI * 2); ctx.fill();
      drawSegmentRing(ctx, segs, cx, cy, r, Math.max(3, r * 0.11));
    } else {
      if (on) {
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 1.35);
        g.addColorStop(0, css(base, 0.28));
        g.addColorStop(1, css(base, 0));
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.arc(cx, cy, r * 1.35, 0, Math.PI * 2); ctx.fill();
      }
      ctx.beginPath();
      ctx.strokeStyle = on ? css(base) : '#D5D5DA';
      ctx.lineWidth = Math.max(3, r * 0.11);
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
    }
    const bg = ctx.createRadialGradient(cx - r * 0.3, cy - r * 0.35, r * 0.1, cx, cy, r);
    bg.addColorStop(0, '#2E2E36');
    bg.addColorStop(1, '#131318');
    ctx.fillStyle = bg;
    ctx.beginPath(); ctx.arc(cx, cy, r * 0.8, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = on ? css(base) : '#B9B9C1';
    ctx.beginPath(); ctx.arc(cx, cy - r * 0.52, Math.max(2, r * 0.07), 0, Math.PI * 2); ctx.fill();
  }
  function animate(canvas, getState, renderer) {
    const clock = createClock();
    const draw = renderer || render;
    let running = true;
    (function loop() {
      if (!running || !canvas.isConnected) return;
      try { draw(canvas, getState() || {}, clock); } catch (e) { /* keep looping */ }
      requestAnimationFrame(loop);
    })();
    return () => { running = false; };
  }

  const api = {
    SEGMENTS, EFFECTS, frame, meanColor, meanLuma, isEffect, rgb, css,
    createClock, drawSegmentRing, render, renderGlyph, animate,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.RingFx = api;
})(typeof window !== 'undefined' ? window : globalThis);
