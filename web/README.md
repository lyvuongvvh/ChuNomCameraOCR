# Phase 5: Frontend (mobile web app)

Plain HTML/CSS/JS, no build step - a single page: take a photo, review recognized text, opt in to
translate specific lines. Served by the same FastAPI process as the backend (`api/main.py` mounts
this directory as static files), so there's no separate frontend server and no CORS to configure.

## Why a native-camera file input, not a live in-browser camera preview

`<input type="file" accept="image/*" capture="environment">` opens the phone's own camera app
directly (take/retake/confirm there, then return the photo to the page) instead of embedding a
live `getUserMedia` video feed. Confirmed before choosing this: it does **not** require a secure
context (HTTPS/localhost) the way `getUserMedia` does, and it isn't affected by a real, still-open
WebKit bug that breaks `getUserMedia` specifically when a PWA is launched in installed/
"standalone" mode on iOS. Trade-off: no live viewfinder in the page itself - acceptable for a
document-photo use case, and it means the phone can reach the laptop over plain HTTP on the local
network, no tunnel or certificate needed.

## Running it

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
`--host 0.0.0.0` makes it reachable from other devices on the same network, not just the laptop
itself. Find the laptop's LAN IP (`ipconfig` on Windows, `ifconfig`/`ip addr` on macOS/Linux), then
on the phone (same Wi-Fi), open `http://<that-ip>:8000`.

Requires the real data files `api/config.py` points at (recognizer weights, `char_lm.json`,
`best_lambda.json`, `reading_dict.json`, the Rung 1 detection checkpoint) to already exist on
disk - see `api/config.py`'s `*_PATH` env vars to point at them if they're not in the default
`experiments/*/data` locations. `ANTHROPIC_API_KEY` is optional - without it, `translate=true`
requests fail fast with a 503 (nothing else in the app requires it).

## What each file is

- `index.html` - the whole page: capture button, image preview, results list, translate controls.
- `app.js` - keeps the captured `File` in memory (so it can be POSTed twice - once for
  recognition, again for translation - without recapturing), calls `POST /v1/ocr` matching
  `api/routers/ocr.py`'s exact request shape, and renders `api/schemas.py`'s `OcrResponse`.
- `style.css` - note: any element that's toggled via the HTML `hidden` attribute must not have its
  own `display` CSS property set on a plain `#id` rule, or that rule's specificity beats the
  browser's `[hidden] { display: none }` default - use `#id:not([hidden]) { display: ...; }`
  instead (a real bug hit and fixed while building this: the image preview showed a broken-image
  placeholder even while "hidden").
- `manifest.webmanifest` + `icons/` - installable-PWA metadata (a bonus; capture doesn't depend on
  it, so there's no iOS-specific caveat to work around here).

## Verified so far (this session, against the real pipeline - not mocked)

Ran a real page image (`DVSKTT_thu_III_1a`) through the actual browser UI end-to-end: detection
found 13 lines and recognized readable Literary Chinese text for each. Also verified the
`translate=true` fail-fast path for real (503 with no `ANTHROPIC_API_KEY` configured) - the error
surfaces cleanly in the UI without losing the already-rendered recognition results.

**Not yet done:** a real run on an actual phone (needs a real device on the same Wi-Fi + a real
Nôm source photo - the Browser tool used for the above has no camera hardware, so the file-input
capture step itself couldn't be exercised, only simulated by injecting a file programmatically),
and a real Anthropic-backed translate run (small and deliberate, like Phase 3's own $0.33-$0.42
samples - not run automatically here to avoid spending on the user's API key without asking).
