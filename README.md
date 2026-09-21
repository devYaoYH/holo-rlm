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

## Visual attention attribution

See [the attention attribution pipeline](docs/attention-attribution.md) for the capture artifacts, value-norm correction, cross-layer rollout, parameter-span aggregation, multi-frame linkage, and interpretation limits.

Turn any traced local request—or a trajectory bundle that references one—into an interactive patch-level viewer:

```bash
uv run holo-capture attribution data/trajectories/v0/<trajectory-id>
# or select a trace directly
uv run holo-capture attribution data/traces/<trace-id>
```

To compare every completed pre-action frame in one filmstrip, while also generating the detailed token viewer for each trace:

```bash
uv run holo-capture attribution data/trajectories/v0/<trajectory-id> --all-frames
```

The command writes a self-contained `viewer.html`, an `attribution.json` summary, and aggregate PNG previews under `data/attributions/<trace-id>/`. The image is shown first with the controls below it. Each request retains all earlier model-input screenshots, and the viewer's **Input frame** selector lets you project a later output token or parameter value back onto any historical or current frame. You can also isolate layers or heads for direct maps and switch among direct attention, value-norm correction, and cross-layer rollout. Parameter maps are arithmetic means over the value's generated tokens, so token length cannot inflate a parameter's attribution. **Subtract previous-token baseline** removes the mean map of only the tokens generated before the selected token or parameter began; future output tokens are never used.

The corrected view computes `A' = A * ||V||₂ / sum(A * ||V||₂)` over all keys for every captured layer and query head before selecting and reshaping the contiguous image-token span. Grouped-query heads reuse the norm of their corresponding KV head. Rollout row-normalizes the head-mean matrix, mixes it equally with the identity residual path, and composes all eight conventional full-attention blocks in model order. Qwen's interleaved linear-attention blocks do not expose square softmax matrices and therefore are not part of this matrix rollout. These remain routing diagnostics rather than causal explanations.

The [ScreenSpot-Pro case study](docs/screenspot-case-study.md) adds a same-image diverse-instruction baseline, strict versus visually repaired grounding scores, leave-one-control-out stability, and per-layer/per-head diagnostics for a fixed PowerPoint success/failure pair.

The [matched causal coordinate intervention](docs/causal-intervention.md) swaps one target tile with an equally sized distractor, scores a teacher-forced coordinate sequence log-likelihood ratio in natural-log units, and tests clean-to-corrupted activation patching plus clean-run ablation for visual residuals, layer 19/head 10, selected MLPs, and matched target/distractor patches. The [activation patching library](docs/activation-patching-library.md) exposes the same mechanics for other screenshot and multi-frame trajectory pairs.

For an RTX 5090/A100 host, follow the [remote GPU experiment workflow](docs/remote-gpu-experiments.md). It covers locked setup and CUDA preflight, resumable/sharded ScreenSpot-Pro inference with all generated-token log probabilities, multi-frame attribution batches, manifest-driven activation patching, and checksum-verified result retrieval.

To retain attribution for every generated token in the bounded 128-token action response:

```bash
uv run holo-capture run --backend local --task cheapest --max-steps 5 --stop-on-click --trace-generation-steps 128
```

`--stop-on-click` makes the capture boundary explicit: the bundle is finalized immediately after the first click is applied, even if the model selected the wrong result.

Local Holo pointer coordinates are projected from its native 0–1000 space into the captured viewport, and its native wheel sign is converted into fixture scroll direction before replay.

The old four-seed fixture is retained for replay compatibility. New cheapest-hotel evaluation uses the frozen 120-item `booking-synth-v2` manifest, with independently seeded ordering, prices, content, visuals, and geometry. See [the synthetic evaluation dataset contract](docs/synthetic-eval.md).

Run the first 20 frozen items, including bounded failures:

```bash
make benchmark BACKEND=scripted
```

Run or inspect a particular frozen item:

```bash
uv run holo-capture run --backend scripted --eval-item test-0000
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
