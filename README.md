# Holo desktop trajectory capture

This repository contains a deterministic localhost-only booking-results fixture and a capture pipeline for replayable Holo trajectories. Generated captures remain under `data/` and are git-ignored. The fixture never links to a real booking, checkout, login, mail, payment, or deletion surface.

## Setup

```bash
uv sync --dev
cp .env.example .env
make doctor BACKEND=scripted
make test
```

The installed Holo CLI is detected from `PATH`; it does not need to be installed into this project's Python environment. On macOS, grant Screen Recording and Accessibility only to Holo's runtime and the terminal used for the demo. Use a disposable browser profile, close unrelated apps, and use the double-Esc kill switch if the desktop run leaves the fixture.

## Backends

- `scripted`: deterministic, offline reference policy. It exercises fixture reset, exact model-input images, actions, capture, validation, and report generation without controlling the desktop.
- `local`: calls an OpenAI-compatible `HOLO_BASE_URL` using `Hcompany/Holo-3.1-4B`. Requests contain the exact screenshot bytes and constrained action schema. Only this mode is eligible for the local-4B research split.
- `hosted`: reserved for Holo CLI plumbing. Captures are labelled `hosted` and never accepted into the local research split.

Run preflight without taking an action:

```bash
make doctor BACKEND=local
make smoke BACKEND=local
```

`doctor` records CLI/runtime and endpoint metadata in `data/manifests/`. `smoke` checks the endpoint with a tiny text request; it does not actuate the desktop.

## Safe deterministic demo

The following one command starts the fixture, runs a bounded replayable policy, writes a bundle, validates it, replays its actions against a fresh reset, and creates `report.html`:

```bash
make demo BACKEND=scripted SEED=0 VARIANT=0
```

For the native local model:

```bash
cd instrumented_server
HOLO_DEVICE=mps HOLO_DTYPE=auto uv run instrumented-holo-server
# In another terminal, from the repository root:
HOLO_BASE_URL=http://127.0.0.1:8000/v1 make demo BACKEND=local SEED=0
```

The local model must emit exactly one schema-valid action per step, either as a native tool call or JSON fallback. Invalid output stops cleanly and is retained. Each local response carries an `instrumented_trace_id`; the trajectory event and annotations retain that ID so the exact request/frame can be joined to its attention and hidden-state bundle under `data/traces/`. A successful run opens only a local details route.

Generate the v0 benchmark slice (four layout seeds × five task variants, with bounded failures included):

```bash
make benchmark BACKEND=scripted
```

## Holo CLI plumbing run

`holo-capture holo` launches/verifies the fixture, invokes the installed `holo run` with bounded steps, and imports its run JSONL as raw events. Set `HOLO_BROWSER_COMMAND` to a command that opens `{url}` with a disposable `{profile}` directory. Example:

```bash
export HOLO_BROWSER_COMMAND="'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' --user-data-dir={profile} --new-window {url}"
export HOLO_CAPTURE_CONSENT=I_HAVE_ISOLATED_THE_DESKTOP
uv run holo-capture holo --backend local --seed 0
```

The command refuses to run without the exact capture-consent acknowledgement above, a browser command containing the `{profile}` placeholder, and an attached browser process (not macOS `open`) that it can terminate afterward. Set the acknowledgement only after closing unrelated apps and verifying that no permission prompt or other desktop is visible. Runtime-only captures are labelled `replay_ready=false` unless exact model requests/images can be correlated from the local endpoint. They are useful for CLI plumbing, not mechanistic replay.

## Retention and privacy

Raw events, exact request/response JSON, screenshots, and reports are local-only. No artifact is uploaded automatically. The writer rejects secrets, cookies, authorization headers, and non-local URLs before finalization. Configured redaction rectangles are applied before an image is written; the manifest records each redaction. Delete `data/trajectories/` manually when retention is no longer needed.

See [the trajectory contract](docs/trajectory-format.md) for the stable format and replay guarantees.
