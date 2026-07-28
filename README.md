# Bending

A browser-based, webcam-driven "elemental bending" toy. Make a fist to charge
a glowing orb in your palm, open your hand to throw it — fire, water/ice, or
air. Everything runs client-side; no video ever leaves your device.

Vanilla HTML/CSS/JS, no build step, no backend. See `main.js` for
implementation notes (which MediaPipe API is used and why, rendering
approach, etc).

## Run it locally

Camera access requires a served origin (not `file://`) in most browsers, so
use any static file server, e.g.:

npx serve .

or

python3 -m http.server 8000

Then open the printed URL (e.g. http://localhost:3000 or
http://localhost:8000) in Chrome or Edge on a laptop/desktop with a webcam.
Click "Allow camera & begin" and grant camera permission when prompted.

## Deploy to Vercel

vercel

or connect the GitHub repo in the Vercel dashboard. No environment variables
or secrets needed — it's fully static.

## What to test

- Fist charges a glowing orb at your palm — hold longer, it grows (capped).
- Opening your hand throws the charge in the direction your hand points
  (wrist → middle finger).
- An open, non-charging palm gives off a gentle idle stream.
- Works with up to two hands at once.
- The Fire / Water / Air segmented control re-themes the whole UI and
  switches the effect: fire rises and glows, water falls and arcs with a
  fraction of ice shards mixed into throws, air is a light fast-dissipating
  gust.
- Opening this on a small/narrow screen (or a touch device) shows a "use a
  laptop or desktop" gate instead of a broken experience.
- "Show tracking" checkbox toggles the faint landmark dots.
- No-hand hint appears when tracking loses your hand.

## Not yet built

Record & share (capture a clip and download it) is left as a TODO comment
in main.js — a nice-to-have that wasn't built since the brief said to only
add it once the core is solid.
