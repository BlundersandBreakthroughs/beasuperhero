// Bending — Phase 2 (fire + water/ice + air)
//
// Hand tracking: MediaPipe Tasks Vision `HandLandmarker` (the current, actively
// maintained API — the older `@mediapipe/hands` "solutions" package is in
// maintenance mode). Loaded straight from a CDN as an ES module, no build step.
//
// Rendering: two stacked full-viewport <canvas> elements sharing one coordinate
// space — a video canvas (mirrored webcam feed) underneath, and a transparent FX
// canvas on top for particles. All effects are plain Canvas 2D. A future upgrade
// path would be to swap the FX canvas for WebGL/PixiJS for cheaper additive
// blending at very high particle counts, but that's out of scope here.
//
// TODO (nice-to-have, not built): record & share — a button that captures a
// short clip via MediaRecorder on fxCanvas/videoCanvas's captureStream() and
// offers it as a download.

// MediaPipe is imported dynamically inside initHandLandmarker(), not as a
// static top-level import: a static import blocks execution of this entire
// module until it resolves, so a CDN hiccup would silently kill the mobile
// gate, element toggle, and start button along with hand tracking. Deferring
// it keeps page wiring independent of that network dependency and routes any
// failure through the normal try/catch -> friendly error screen in begin().
const MEDIAPIPE_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const screens = {
  mobileGate: document.getElementById("mobile-gate"),
  landing: document.getElementById("landing"),
  loading: document.getElementById("loading"),
  error: document.getElementById("error-screen"),
  experience: document.getElementById("experience"),
};

const startBtn = document.getElementById("start-btn");
const retryBtn = document.getElementById("retry-btn");
const landingError = document.getElementById("landing-error");
const errorMessage = document.getElementById("error-message");
const coachEl = document.getElementById("coach");
const noHandEl = document.getElementById("no-hand-hint");
const showTrackingEl = document.getElementById("show-tracking");
const elementButtons = Array.from(document.querySelectorAll(".element-btn"));

const video = document.getElementById("webcam");
const videoCanvas = document.getElementById("video-canvas");
const fxCanvas = document.getElementById("fx-canvas");
const videoCtx = videoCanvas.getContext("2d");
const fxCtx = fxCanvas.getContext("2d", { alpha: true });

function showScreen(name) {
  for (const key of Object.keys(screens)) {
    screens[key].hidden = key !== name;
  }
}

// ---------------------------------------------------------------------------
// Mobile / small-screen gate — this is a laptop/desktop experience, so bail
// out early with a friendly message instead of letting hand tracking fail
// silently on a phone.
// ---------------------------------------------------------------------------

function isUnsupportedScreen() {
  return (
    window.matchMedia("(max-width: 820px)").matches ||
    window.matchMedia("(pointer: coarse)").matches
  );
}

function checkMobileGate() {
  if (isUnsupportedScreen()) {
    showScreen("mobileGate");
    return true;
  }
  return false;
}

const gatedOnLoad = checkMobileGate();
if (!gatedOnLoad) {
  showScreen("landing");
}

// Re-check on resize, but never yank the user out of a session already in
// progress (e.g. a dev-tools panel toggling `pointer: coarse` briefly).
window.addEventListener("resize", () => {
  if (!screens.experience.hidden) return;
  checkMobileGate();
});

// ---------------------------------------------------------------------------
// Element theming — one source of truth per element. Swapping
// `document.body.dataset.element` re-themes the CSS (panels, accents,
// background). These JS-side tables mirror the same palette for canvas
// drawing, which CSS custom properties can't reach into.
// ---------------------------------------------------------------------------

let currentElement = "fire";

const ELEMENT_ORB_STOPS = {
  fire: ["#ffffff", "#ffe28a", "#ff8a1e", "rgba(255,59,0,0)"],
  water: ["#ffffff", "#dff6ff", "#61a5c2", "rgba(42,111,151,0)"],
  air: ["#ffffff", "#eef0f2", "#b7bcc4", "rgba(138,145,156,0)"],
};

const ELEMENT_TRACK_COLOR = {
  fire: "rgba(255,150,60,0.55)",
  water: "rgba(120,190,220,0.55)",
  air: "rgba(210,214,220,0.5)",
};

function setElement(name) {
  currentElement = name;
  document.body.dataset.element = name;
  for (const btn of elementButtons) {
    const active = btn.dataset.element === name;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-pressed", active ? "true" : "false");
  }
}

for (const btn of elementButtons) {
  btn.addEventListener("click", () => setElement(btn.dataset.element));
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const MAX_PARTICLES = 2000;
const CHARGE_TIME_MS = 1100; // fist-held time to reach full charge
const CHARGE_DECAY_MS = 500; // charge lost per second while hand is missing
const HAND_LOST_GRACE_MS = 350; // keep hand state alive this long after tracking drops it
const IDLE_SPAWN_RATE = 4; // particles/sec from an idle open palm

// ---------------------------------------------------------------------------
// Glow sprites — generated once at startup, then stamped per-particle instead
// of drawing hard circles. This is what gives the particles their soft,
// luminous look. Each element has its own color ramp; particles pick the
// nearest ramp sprite by life ratio (fresh -> old) each frame rather than
// building a gradient per particle per frame. Ice gets its own faceted sprite
// so thrown water reads as shards, not droplets.
// ---------------------------------------------------------------------------

const SPRITE_SIZE = 128;

const RAMPS = {
  fire: ["#ffffff", "#fff2b0", "#ffb703", "#ff6a00", "#ff3b00", "#7a1600"],
  water: ["#eaf7fb", "#c7e8f3", "#a9d6e5", "#61a5c2", "#2a6f97", "#01314f"],
  air: ["#ffffff", "#eef0f2", "#d7dbe0", "#b7bcc4", "#8a919c"],
};

function makeGlowSprite(hexColor) {
  const c = document.createElement("canvas");
  c.width = c.height = SPRITE_SIZE;
  const ctx = c.getContext("2d");
  const r = SPRITE_SIZE / 2;
  const g = ctx.createRadialGradient(r, r, 0, r, r, r);
  g.addColorStop(0, hexColor + "ff");
  g.addColorStop(0.35, hexColor + "cc");
  g.addColorStop(1, hexColor + "00");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, SPRITE_SIZE, SPRITE_SIZE);
  return c;
}

function makeIceSprite() {
  // A soft-edged diamond instead of a circle, so ice shards read as
  // angular/crystalline rather than round droplets.
  const c = document.createElement("canvas");
  c.width = c.height = SPRITE_SIZE;
  const ctx = c.getContext("2d");
  const r = SPRITE_SIZE / 2;
  ctx.save();
  ctx.translate(r, r);
  ctx.rotate(Math.PI / 4);
  const g = ctx.createRadialGradient(0, 0, 0, 0, 0, r * 0.9);
  g.addColorStop(0, "#eafcffff");
  g.addColorStop(0.5, "#c8f1ffcc");
  g.addColorStop(1, "#a9e4f500");
  ctx.fillStyle = g;
  ctx.fillRect(-r * 0.65, -r * 0.65, r * 1.3, r * 1.3);
  ctx.restore();
  return c;
}

const sprites = {
  fire: RAMPS.fire.map(makeGlowSprite),
  water: RAMPS.water.map(makeGlowSprite),
  air: RAMPS.air.map(makeGlowSprite),
  ice: makeIceSprite(),
};

function spriteForRamp(element, ratio) {
  // ratio: 1 = freshly spawned, 0 = about to die
  const ramp = sprites[element];
  const idx = Math.min(ramp.length - 1, Math.floor((1 - ratio) * ramp.length));
  return ramp[idx];
}

// ---------------------------------------------------------------------------
// Particle system
// ---------------------------------------------------------------------------

/**
 * @typedef {{x:number,y:number,vx:number,vy:number,life:number,maxLife:number,
 *   size:number,kind:string,element:string,blend:string,maxAlpha:number,spin?:number}} Particle
 */

/** @type {Particle[]} */
const particles = [];

function spawnParticle(p) {
  if (particles.length >= MAX_PARTICLES) {
    particles.shift(); // recycle oldest rather than growing unbounded
  }
  particles.push(p);
}

function applyPhysics(p, dt) {
  switch (p.element) {
    case "fire":
      // Buoyant rise + flicker.
      p.vy -= 40 * dt;
      p.vx += (Math.random() - 0.5) * 40 * dt;
      p.vx *= 0.98;
      break;

    case "water":
      if (p.kind === "ice") {
        // Ice shards travel straight, no gravity, just gentle drag.
        p.vx *= 0.995;
        p.vy *= 0.995;
      } else {
        // Real gravity so thrown water arcs downward like a splash.
        p.vy += 260 * dt;
        p.vx *= 0.995;
      }
      break;

    case "air":
      // A light curling gust: rotate the velocity vector slightly each
      // frame so the stream swirls rather than flying dead straight.
      if (p.spin === undefined) {
        p.spin = (Math.random() < 0.5 ? -1 : 1) * (2 + Math.random() * 2);
      }
      {
        const ang = p.spin * dt;
        const cos = Math.cos(ang);
        const sin = Math.sin(ang);
        const nvx = p.vx * cos - p.vy * sin;
        const nvy = p.vx * sin + p.vy * cos;
        p.vx = nvx;
        p.vy = nvy;
      }
      p.vx *= 0.985;
      p.vy *= 0.985;
      break;
  }
  p.x += p.vx * dt;
  p.y += p.vy * dt;
}

function updateParticles(dt) {
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.life -= dt;
    if (p.life <= 0) {
      particles.splice(i, 1);
      continue;
    }
    applyPhysics(p, dt);
  }
}

function spriteForParticle(p, ratio) {
  if (p.element === "water" && p.kind === "ice") return sprites.ice;
  return spriteForRamp(p.element, ratio);
}

function drawParticles() {
  for (const p of particles) {
    const ratio = p.life / p.maxLife;
    const sprite = spriteForParticle(p, ratio);
    const size = p.size * (0.4 + 0.6 * ratio); // shrink as it ages
    fxCtx.globalCompositeOperation = p.blend;
    fxCtx.globalAlpha = Math.min(p.maxAlpha, ratio * 1.4 * p.maxAlpha);
    fxCtx.drawImage(sprite, p.x - size / 2, p.y - size / 2, size, size);
  }
  fxCtx.globalAlpha = 1;
  fxCtx.globalCompositeOperation = "source-over";
}

// ---------------------------------------------------------------------------
// Hand state — persisted across frames so charge/gesture survives brief
// tracking jitter. Matched to newly detected landmarks by nearest palm
// position each frame (tracking gives up to 2 hands, order isn't stable).
// ---------------------------------------------------------------------------

let handStates = []; // { id, palm, pointDir, mode, charge, lastSeen, landmarks }
let nextHandId = 0;

function palmCenter(landmarks) {
  const idxs = [0, 5, 9, 13, 17];
  let x = 0,
    y = 0;
  for (const i of idxs) {
    x += landmarks[i].x;
    y += landmarks[i].y;
  }
  return { x: x / idxs.length, y: y / idxs.length };
}

function countExtendedFingers(landmarks) {
  // A finger is "extended" if its tip is farther from the wrist than its
  // middle (PIP) joint is. Robust to hand rotation without needing depth.
  const wrist = landmarks[0];
  const pairs = [
    [8, 6], // index: tip, pip
    [12, 10], // middle
    [16, 14], // ring
    [20, 18], // pinky
  ];
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  let count = 0;
  for (const [tipIdx, pipIdx] of pairs) {
    if (dist(landmarks[tipIdx], wrist) > dist(landmarks[pipIdx], wrist)) {
      count++;
    }
  }
  // Thumb: compare tip (4) to its MCP (2) distance from wrist, weighted less
  // since thumb geometry is noisier.
  if (dist(landmarks[4], wrist) > dist(landmarks[2], wrist) * 1.05) {
    count += 0.5;
  }
  return count;
}

function pointingDirection(landmarks) {
  const wrist = landmarks[0];
  const middleTip = landmarks[12];
  const dx = middleTip.x - wrist.x;
  const dy = middleTip.y - wrist.y;
  const len = Math.hypot(dx, dy) || 1;
  return { x: dx / len, y: dy / len };
}

function classifyMode(extendedCount) {
  if (extendedCount <= 1) return "fist";
  if (extendedCount >= 3) return "open";
  return "neutral";
}

function updateHandStates(detectedHands, dt, canvasW, canvasH) {
  const now = performance.now();
  const usedStateIds = new Set();

  for (const landmarks of detectedHands) {
    const normPalm = palmCenter(landmarks);
    // Mirror X to match the mirrored video (selfie mode).
    const palm = {
      x: (1 - normPalm.x) * canvasW,
      y: normPalm.y * canvasH,
    };
    const rawDir = pointingDirection(landmarks);
    const pointDir = { x: -rawDir.x, y: rawDir.y }; // mirror X here too

    // Match to nearest existing (unused) state within a reasonable radius.
    let best = null;
    let bestDist = Infinity;
    for (const state of handStates) {
      if (usedStateIds.has(state.id)) continue;
      const d = Math.hypot(state.palm.x - palm.x, state.palm.y - palm.y);
      if (d < bestDist) {
        bestDist = d;
        best = state;
      }
    }

    const matchThreshold = Math.max(canvasW, canvasH) * 0.25;
    let state;
    if (best && bestDist < matchThreshold) {
      state = best;
    } else {
      state = {
        id: nextHandId++,
        mode: "open",
        charge: 0,
        justThrew: false,
      };
      handStates.push(state);
    }

    usedStateIds.add(state.id);
    state.palm = palm;
    state.pointDir = pointDir;
    state.landmarks = landmarks;
    state.lastSeen = now;
    state.tracked = true;

    const extendedCount = countExtendedFingers(landmarks);
    const classification = classifyMode(extendedCount);

    state.justThrew = false;
    if (classification !== "neutral") {
      const prevMode = state.mode;
      state.mode = classification;
      if (prevMode === "fist" && classification === "open") {
        state.justThrew = true;
      }
    }

    if (state.mode === "fist") {
      state.charge = Math.min(1, state.charge + (dt * 1000) / CHARGE_TIME_MS);
    }
  }

  // Age out / decay states that weren't matched this frame.
  handStates = handStates.filter((state) => {
    if (usedStateIds.has(state.id)) return true;
    state.tracked = false;
    const missingFor = now - state.lastSeen;
    state.charge = Math.max(0, state.charge - (dt * 1000) / CHARGE_DECAY_MS);
    return missingFor < HAND_LOST_GRACE_MS;
  });

  return handStates;
}

// ---------------------------------------------------------------------------
// Per-element gesture effects: charge orb, throw burst, idle stream
// ---------------------------------------------------------------------------

const idleSpawnAccumulator = new Map();

function rotate(vx, vy, angle) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return { x: vx * cos - vy * sin, y: vx * sin + vy * cos };
}

function throwFire(state) {
  const power = 0.35 + state.charge * 0.65; // even an unheld "open" throw has a little juice
  const count = Math.round(18 + state.charge * 90);
  const baseSpeed = 220 + state.charge * 480;
  const { x, y } = state.palm;
  const dir = state.pointDir;

  for (let i = 0; i < count; i++) {
    const spread = (Math.random() - 0.5) * 0.5;
    const speed = baseSpeed * (0.7 + Math.random() * 0.6);
    const v = rotate(dir.x, dir.y, spread);
    spawnParticle({
      x,
      y,
      vx: v.x * speed,
      vy: v.y * speed,
      life: 0.5 + Math.random() * 0.4 + power * 0.3,
      maxLife: 0.5 + power * 0.5,
      size: 22 + Math.random() * 20 + power * 20,
      kind: "throw",
      element: "fire",
      blend: "lighter",
      maxAlpha: 1,
    });
  }
}

function spawnIdleFire(state, dt) {
  const acc = (idleSpawnAccumulator.get(state.id) || 0) + dt * IDLE_SPAWN_RATE;
  let toSpawn = Math.floor(acc);
  idleSpawnAccumulator.set(state.id, acc - toSpawn);

  while (toSpawn-- > 0) {
    const { x, y } = state.palm;
    spawnParticle({
      x: x + (Math.random() - 0.5) * 14,
      y: y + (Math.random() - 0.5) * 14,
      vx: (Math.random() - 0.5) * 20,
      vy: -30 - Math.random() * 30,
      life: 0.6 + Math.random() * 0.3,
      maxLife: 0.9,
      size: 14 + Math.random() * 10,
      kind: "idle",
      element: "fire",
      blend: "lighter",
      maxAlpha: 1,
    });
  }
}

function throwWater(state) {
  const power = 0.35 + state.charge * 0.65;
  const count = Math.round(20 + state.charge * 70);
  const baseSpeed = 200 + state.charge * 380;
  const { x, y } = state.palm;
  const dir = state.pointDir;
  const ICE_FRACTION = 0.25;

  for (let i = 0; i < count; i++) {
    const isIce = Math.random() < ICE_FRACTION;
    const spread = (Math.random() - 0.5) * (isIce ? 0.25 : 0.45);
    const speed = baseSpeed * (0.7 + Math.random() * 0.6);
    const v = rotate(dir.x, dir.y, spread);

    if (isIce) {
      spawnParticle({
        x,
        y,
        vx: v.x * speed,
        vy: v.y * speed,
        life: 0.4 + Math.random() * 0.35 + power * 0.2,
        maxLife: 0.5 + power * 0.3,
        size: 16 + Math.random() * 14 + power * 12,
        kind: "ice",
        element: "water",
        blend: "lighter",
        maxAlpha: 1,
      });
    } else {
      // A small upward kick so gravity pulls it into a real arc rather than
      // an instant straight drop.
      spawnParticle({
        x,
        y,
        vx: v.x * speed,
        vy: v.y * speed - (60 + Math.random() * 60),
        life: 0.6 + Math.random() * 0.5 + power * 0.3,
        maxLife: 0.6 + power * 0.5,
        size: 12 + Math.random() * 14 + power * 10,
        kind: "drop",
        element: "water",
        blend: Math.random() < 0.3 ? "lighter" : "source-over",
        maxAlpha: 1,
      });
    }
  }
}

function spawnIdleWater(state, dt) {
  const acc = (idleSpawnAccumulator.get(state.id) || 0) + dt * IDLE_SPAWN_RATE;
  let toSpawn = Math.floor(acc);
  idleSpawnAccumulator.set(state.id, acc - toSpawn);

  while (toSpawn-- > 0) {
    const { x, y } = state.palm;
    spawnParticle({
      x: x + (Math.random() - 0.5) * 14,
      y: y + (Math.random() - 0.5) * 10,
      vx: (Math.random() - 0.5) * 15,
      vy: 10 + Math.random() * 20,
      life: 0.5 + Math.random() * 0.3,
      maxLife: 0.8,
      size: 8 + Math.random() * 8,
      kind: "drop",
      element: "water",
      blend: "source-over",
      maxAlpha: 1,
    });
  }
}

function throwAir(state) {
  const power = 0.3 + state.charge * 0.7;
  const count = Math.round(30 + state.charge * 60);
  const baseSpeed = 260 + state.charge * 300;
  const { x, y } = state.palm;
  const dir = state.pointDir;

  for (let i = 0; i < count; i++) {
    const spread = (Math.random() - 0.5) * 0.8;
    const speed = baseSpeed * (0.7 + Math.random() * 0.6);
    const v = rotate(dir.x, dir.y, spread);
    spawnParticle({
      x,
      y,
      vx: v.x * speed,
      vy: v.y * speed,
      life: 0.25 + Math.random() * 0.25 + power * 0.15,
      maxLife: 0.4 + power * 0.2,
      size: 18 + Math.random() * 16 + power * 10,
      kind: "throw",
      element: "air",
      blend: "lighter",
      maxAlpha: 0.55,
    });
  }
}

function spawnIdleAir(state, dt) {
  const acc = (idleSpawnAccumulator.get(state.id) || 0) + dt * IDLE_SPAWN_RATE;
  let toSpawn = Math.floor(acc);
  idleSpawnAccumulator.set(state.id, acc - toSpawn);

  while (toSpawn-- > 0) {
    const { x, y } = state.palm;
    spawnParticle({
      x: x + (Math.random() - 0.5) * 14,
      y: y + (Math.random() - 0.5) * 14,
      vx: (Math.random() - 0.5) * 60,
      vy: (Math.random() - 0.5) * 60,
      life: 0.3 + Math.random() * 0.2,
      maxLife: 0.5,
      size: 12 + Math.random() * 8,
      kind: "idle",
      element: "air",
      blend: "lighter",
      maxAlpha: 0.4,
    });
  }
}

const THROW_FN = { fire: throwFire, water: throwWater, air: throwAir };
const IDLE_FN = { fire: spawnIdleFire, water: spawnIdleWater, air: spawnIdleAir };

function drawChargeOrb(state) {
  if (state.charge <= 0.02) return;
  const { x, y } = state.palm;
  const radius = 14 + state.charge * 46;
  const stops = ELEMENT_ORB_STOPS[currentElement];

  fxCtx.save();
  fxCtx.globalCompositeOperation = "lighter";
  const g = fxCtx.createRadialGradient(x, y, 0, x, y, radius);
  g.addColorStop(0, stops[0]);
  g.addColorStop(0.3, stops[1]);
  g.addColorStop(0.65, stops[2]);
  g.addColorStop(1, stops[3]);
  fxCtx.fillStyle = g;
  fxCtx.globalAlpha = 0.85;
  fxCtx.beginPath();
  fxCtx.arc(x, y, radius, 0, Math.PI * 2);
  fxCtx.fill();
  fxCtx.restore();
}

// ---------------------------------------------------------------------------
// Tracking-dot overlay (subtle, toggleable)
// ---------------------------------------------------------------------------

function drawTrackingDots(state, canvasW, canvasH) {
  if (!state.tracked || !state.landmarks) return;
  fxCtx.save();
  fxCtx.fillStyle = ELEMENT_TRACK_COLOR[currentElement];
  for (const lm of state.landmarks) {
    const x = (1 - lm.x) * canvasW;
    const y = lm.y * canvasH;
    fxCtx.beginPath();
    fxCtx.arc(x, y, 2.5, 0, Math.PI * 2);
    fxCtx.fill();
  }
  fxCtx.restore();
}

// ---------------------------------------------------------------------------
// Camera + MediaPipe setup
// ---------------------------------------------------------------------------

let handLandmarker = null;
let rafId = null;
let lastFrameTime = 0;
let lastDetectTime = -1;

async function initHandLandmarker() {
  const { HandLandmarker, FilesetResolver } = await import(MEDIAPIPE_URL);
  const filesetResolver = await FilesetResolver.forVisionTasks(WASM_URL);
  handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

function resizeCanvases() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  for (const c of [videoCanvas, fxCanvas]) {
    c.width = w;
    c.height = h;
  }
}

function drawMirroredVideo() {
  const w = videoCanvas.width;
  const h = videoCanvas.height;
  videoCtx.save();
  videoCtx.translate(w, 0);
  videoCtx.scale(-1, 1);
  // Cover-fit the video into the canvas.
  const vw = video.videoWidth || w;
  const vh = video.videoHeight || h;
  const scale = Math.max(w / vw, h / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  const dx = (w - dw) / 2;
  const dy = (h - dh) / 2;
  videoCtx.drawImage(video, dx, dy, dw, dh);
  videoCtx.restore();
}

function updateCoachAndHints(states) {
  const anyTracked = states.some((s) => s.tracked);
  noHandEl.hidden = anyTracked;
  const anyFist = states.some((s) => s.tracked && s.mode === "fist");
  coachEl.classList.toggle("active", anyFist);
}

function frame(timestampMs) {
  rafId = requestAnimationFrame(frame);

  const dt = lastFrameTime ? Math.min(0.05, (timestampMs - lastFrameTime) / 1000) : 0;
  lastFrameTime = timestampMs;

  // Run hand detection every frame; detectForVideo requires strictly
  // increasing timestamps and is decoupled from the particle render below in
  // spirit (detection could be throttled independently if it ever becomes a
  // bottleneck — particles/rendering always run at full rAF rate regardless).
  let states = handStates;
  if (handLandmarker && video.readyState >= 2 && timestampMs > lastDetectTime) {
    lastDetectTime = timestampMs;
    const result = handLandmarker.detectForVideo(video, timestampMs);
    states = updateHandStates(result.landmarks || [], dt, fxCanvas.width, fxCanvas.height);
  }

  drawMirroredVideo();

  // Fade FX canvas toward transparent for glowing motion trails.
  fxCtx.save();
  fxCtx.globalCompositeOperation = "destination-out";
  fxCtx.fillStyle = "rgba(0,0,0,0.18)";
  fxCtx.fillRect(0, 0, fxCanvas.width, fxCanvas.height);
  fxCtx.restore();

  for (const state of states) {
    if (state.justThrew) {
      THROW_FN[currentElement](state);
    }
    if (state.tracked && state.mode === "open") {
      IDLE_FN[currentElement](state, dt);
    }
  }

  updateParticles(dt);
  drawParticles();

  for (const state of states) {
    if (state.tracked) drawChargeOrb(state);
    if (showTrackingEl.checked) drawTrackingDots(state, fxCanvas.width, fxCanvas.height);
  }

  updateCoachAndHints(states);
}

// ---------------------------------------------------------------------------
// Boot sequence
// ---------------------------------------------------------------------------

async function begin() {
  landingError.hidden = true;
  showScreen("loading");

  try {
    resizeCanvases();
    window.addEventListener("resize", resizeCanvases);

    // Camera must be requested from a user gesture (the button click) — it's
    // requested here, inside begin(), which only ever runs from the click
    // handler below.
    await startCamera();
    await initHandLandmarker();

    showScreen("experience");
    lastFrameTime = 0;
    rafId = requestAnimationFrame(frame);
  } catch (err) {
    console.error(err);
    if (rafId) cancelAnimationFrame(rafId);
    showFriendlyError(err);
  }
}

function showFriendlyError(err) {
  let message =
    "Something went wrong starting the camera or the tracking model. Please try again.";
  if (err && (err.name === "NotAllowedError" || err.name === "PermissionDeniedError")) {
    message =
      "Camera access was denied. Allow camera permission for this site in your browser's settings, then try again.";
  } else if (err && (err.name === "NotFoundError" || err.name === "DevicesNotFoundError")) {
    message = "No camera was found. Connect a webcam and try again.";
  }
  errorMessage.textContent = message;
  showScreen("error");
}

startBtn.addEventListener("click", begin);
retryBtn.addEventListener("click", begin);
