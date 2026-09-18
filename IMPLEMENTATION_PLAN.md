# Holo desktop demo → trajectory capture: implementation handoff

## Objective and boundary

Build a reproducible, side-effect-free desktop-agent demo and the first version of a trajectory-capture pipeline. The milestone ends when a local, deterministic booking-results task produces validated, replayable trajectory bundles.

This milestone deliberately does **not** implement attention analysis, LoRA training, RLM recursion, the audio product, real bookings, payments, mail, or deletion workflows. It creates the data substrate for all of them.

## Non-negotiable choices

- Use a local fixture app, not Booking.com, personal accounts, or any checkout endpoint.
- Run desktop automation only in a disposable browser profile/test desktop. Do not capture a normal working desktop.
- Store the exact screenshot bytes, messages, tool schema, action result, model revision, and preprocessing configuration for every model step.
- Keep raw captures local; do not commit them. Never store secrets, cookies, tokens, or credentials in a trajectory.
- Treat hosted Holo only as a plumbing fallback. Interpretability/fine-tuning data must ultimately be captured with the local `Hcompany/Holo-3.1-4B` inference target (or explicitly labelled as a different backend).

## Deliverables

1. `apps/booking_fixture/`: deterministic local web fixture that emulates a booking-search results page.
2. `src/demo/`: one command that launches or verifies the fixture and drives the desktop-agent demo.
3. `src/capture/`: event-stream adapter, normalizer, validator, and exporter.
4. `data/trajectories/v0/<trajectory-id>/`: ignored replayable trajectory bundles.
5. `docs/trajectory-format.md`: stable schema and capture/replay contract.
6. `tests/`: fixture, schema, hash-linkage, and no-secret/redaction tests.
7. `README.md`: setup, permissions, backend selection, safe-run instructions, and demo command.

## Phase 0 — establish the execution contract

### Tasks

1. Create a Python project with a locked dependency file and a `.env.example`; do not put keys in source control.
2. Install HoloDesktop CLI in the project environment and run its diagnostic command. Record the CLI/runtime version in a generated run manifest.
3. Grant macOS Screen Recording and Accessibility permissions only to the terminal/runtime used for the demo. Use a dedicated browser profile and close all unrelated apps.
4. Define two backend configurations:

   - `hosted`: HoloDesktop’s default hosted Holo route, used only to validate CLI plumbing if necessary.
   - `local`: HoloDesktop pointed at an OpenAI-compatible local endpoint serving `Hcompany/Holo-3.1-4B`.

5. Implement a backend preflight that records model ID, model revision, endpoint type, CLI/runtime version, and whether function calling or structured output is available.
6. Prefer HoloDesktop’s Python session/event-stream API for the final wrapper. Before coding against it, run a short spike against the installed version and document the actual event types and artifact locations. Do not hard-code undocumented SDK method names from examples.

### Acceptance criteria

- `make doctor` (or equivalent) checks the CLI/runtime, permissions, and selected backend without executing an agent action.
- `make smoke BACKEND=hosted` succeeds if credentials are available.
- `make smoke BACKEND=local` succeeds before research captures are accepted; it uses the native checkpoint through an OpenAI-compatible endpoint.
- A manifest clearly says which backend generated every run. Hosted runs cannot be mixed into the local-4B research split.

### Risk and fallback

The desktop CLI supports a local OpenAI-compatible endpoint, but exact local server support depends on hardware and the installed model server. If local mode is blocked, finish the fixture and capture plumbing with hosted mode, label those trajectories `backend=hosted`, and keep the local-research capture milestone blocked until the native checkpoint passes the preflight.

## Phase 1 — deterministic desktop demo fixture

### Scenario

Build a local, static booking-results site with a fixed viewport and seeded data. It must contain a uniquely named hotel, e.g. `Harbor Lantern Hotel`, plus distractor hotels with similar names.

The agent receives an initial task to inspect results and scroll down. The target result appears early and is then scrolled out of view. A later user turn asks it to locate the target’s booking link. The expected safe action is to navigate back to the target result and open a **local details page**—never a purchase or confirmation screen.

### Fixture requirements

- Serve only on `localhost`; no third-party resources, analytics, authentication, or network calls.
- Deterministic IDs, text, image assets, ordering, page dimensions, and scroll positions.
- A visible state panel for test assertions, but do not expose it in the agent prompt.
- A local “details” destination with a success marker such as `data-test=hotel-details-open`.
- Variants for later generalization: target position, hotel name, distractor density, scroll distance, and layout seed.
- A reset endpoint or CLI command that restores the exact initial state.

### Agent interaction contract

Use a constrained action format supported by the selected backend (native function calls if available; otherwise one schema-validated structured action per step). At minimum support:

- `click(x, y)`
- `scroll(delta_y)`
- `wait(milliseconds)`
- `finish(summary)`

Log the complete tool schema and the raw model response at every decision point. The fixture test runner, not the model, determines whether the task succeeded.

### Acceptance criteria

- The scenario resets bit-for-bit from a seed.
- At least one scripted action trace reaches the target-details success marker.
- A HoloDesktop run can complete the benign task or cleanly stop at a bounded step limit.
- No run can reach a real external URL or action with a side effect.

## Phase 2 — capture adapter and canonical trajectory format

### Capture design

Build a small adapter around the desktop client’s event stream. Preserve every raw SDK/runtime event in `raw/events.jsonl`, then convert it to a vendor-neutral event stream. Do not discard unknown fields.

Capture two image sources when available:

1. the screenshot or image payload actually passed to the model; this is mandatory for future mechanistic replay;
2. an optional post-action desktop screenshot for human debugging.

Do not treat a convenience OS screenshot as equivalent to the model input. If the installed event stream does not expose the model-input image and prompt, move model invocation into an outer local harness that constructs and logs those requests, while using the desktop runtime solely for actuation.

### Bundle layout

```text
data/trajectories/v0/<trajectory-id>/
  manifest.json
  events.jsonl
  annotations.json
  frames/
    0000-model-input.png
    0000-post-action.png
  requests/
    0000.json
  responses/
    0000.json
  actions/
    0000.json
  raw/
    events.jsonl
```

`manifest.json` must include schema version, fixture version/seed, task ID, timestamps, software and model revisions, backend, viewport, capture consent, and SHA-256 hashes of all referenced files.

Each normalized event must have a monotonic sequence number, monotonic timestamp, wall-clock timestamp, event type, artifact references, and parent-step reference. Required event types are:

- `user_turn`
- `observation`
- `model_request`
- `model_response`
- `proposed_action`
- `action_result`
- `termination`

Each `model_request` stores the exact message list or lossless reference to it, image references/hashes, tool schema hash, sampling and generation settings, model ID/revision, processor revision/configuration, and any conversation/session IDs. Do not store hidden reasoning if the backend returns it; retain only the externally usable response and action fields.

Each `proposed_action` stores the raw output plus normalized action name/arguments. `action_result` stores success/failure, before/after state reference, destination URL, and fixture assertion result.

Reserve the following nullable fields now so later datasets have one compatible format:

```json
{
  "safety": {
    "risk_class": null,
    "side_effect_boundary": null,
    "confirmation_stage": null,
    "recoverability": null
  },
  "interpretability": {
    "input_token_ids_ref": null,
    "image_grid_ref": null,
    "replay_ready": null
  },
  "rlm": {
    "external_object_ref": null,
    "retrieval_calls": []
  }
}
```

### Privacy and retention

- Add `data/`, `.env`, local run logs, browser profiles, and any artifact cache to `.gitignore`.
- Reject captures containing known secret patterns, authorization headers, cookies, or non-localhost URLs.
- Support configured rectangular redactions before an artifact is persisted, while retaining the fact that redaction occurred in the manifest.
- Keep capture retention local and documented. Do not upload artifacts automatically.

### Acceptance criteria

- A recorded run produces a self-contained bundle with no broken artifact reference.
- Validator verifies file hashes, step ordering, required fields, image dimensions, and the pre-action/request/action/result linkage.
- A raw event can be traced to every normalized event derived from it.
- The validator rejects a deliberately injected key, cookie, non-local URL, missing screenshot, or mismatched hash.

## Phase 3 — capture the v0 benchmark slice

### Dataset slice

Capture a small but clean initial set, for example 20 trajectories across four fixture seeds and five task variants. Include successful and unsuccessful/bounded runs, because both are useful for later causal comparisons.

Annotate only observable labels:

- target hotel and expected destination;
- task success/failure and terminal reason;
- target frame interval and target-visible region;
- whether the target was seen before it left the viewport;
- step where the second “find it again” turn occurs.

Do not manufacture mechanistic labels such as “the model attended to frame 2.” Those belong to the later analysis.

### Reproducibility checks

1. Reset the fixture, replay the same action trace, and confirm the same terminal fixture state.
2. Re-run the validator over every bundle.
3. Generate a static HTML or Markdown run report that displays the task, pre-action screenshot, proposed action, post-action screenshot, and terminal result per step.
4. Manually review all captures for private content and broken task labels.

### Definition of done

The milestone is complete only when an implementer can run one documented command that:

1. starts/resets the fixture;
2. launches a bounded Holo desktop session;
3. records a trajectory bundle;
4. validates it; and
5. produces a human-readable replay report.

The result must include the exact visual/model inputs required to replay a step through the local Holo checkpoint later. At that point, it is safe to begin the separate mech-interp, safety-adapter, and RLM branches.

## Recommended implementation order

1. Project skeleton, `.gitignore`, `.env.example`, and `README`.
2. Local fixture plus deterministic reset and scripted end-to-end test.
3. HoloDesktop preflight and a manually observed one-step smoke run.
4. Bounded agent loop against the fixture.
5. Raw event recorder.
6. Normalizer, bundle writer, and validator.
7. Capture one hand-inspected trajectory, then automate the v0 slice.

Do not start the next item until the previous acceptance criteria pass.

## References

- [HoloDesktop CLI overview](https://hub.hcompany.ai/holo-desktop-cli/introduction): CLI, MCP, A2A, and Python surfaces; hosted or local backend routing.
- [Holo model card](https://huggingface.co/Hcompany/Holo-3.1-4B): Transformers-native multimodal checkpoint and supported inference servers.
- [RLM paper](https://arxiv.org/pdf/2512.24601): the later RLM branch should treat trajectories as external objects, rather than continually compacting them into chat history.
