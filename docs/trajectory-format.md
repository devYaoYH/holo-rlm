# Trajectory format v0

Schema identifier: `holo-trajectory-v0`.

Each directory under `data/trajectories/v0/<trajectory-id>/` is a self-contained, local-only capture. Moving the directory does not break references. Paths are bundle-relative and SHA-256 hashes cover their exact persisted bytes.

## Layout

```text
manifest.json              run identity, provenance, revisions, hashes
events.jsonl               vendor-neutral ordered event stream
annotations.json           observable task labels only
report.html                static human replay report
frames/0000-model-input.png
frames/0000-post-action.png
requests/0000.json         exact request, including messages/tool schema/image bytes
responses/0000.json        externally usable response (no hidden reasoning)
actions/0000.json          raw response plus normalized action
raw/events.jsonl           lossless source/harness events
```

`manifest.json` is not self-hashed. Every other file is listed under `files` with `sha256` and `bytes`. Research-ready local captures use `backend=local`, `research_split=local-4b`, and `research_eligible=true`. Scripted captures are deterministic pipeline tests. Hosted and runtime-only captures are never mixed into the local-4B split.

The manifest records:

- schema and trajectory IDs;
- fixture version, seed, localhost base URL, and viewport;
- task ID and start/completion timestamps;
- capture/software, model, model revision, processor revision/configuration;
- backend and research eligibility;
- explicit capture consent and any rectangles redacted before persistence;
- hashes and byte lengths for all bundle files.

## Event contract

Each JSONL record has:

- a zero-based contiguous `sequence`;
- elapsed `monotonic_ns` and ISO-8601 `wall_time`;
- one of `user_turn`, `observation`, `model_request`, `model_response`, `proposed_action`, `action_result`, or `termination`;
- nullable `parent_step_ref`;
- event-specific `data` and typed `artifacts`;
- one or more `raw_event_refs` back to `raw/events.jsonl`;
- reserved `safety`, `interpretability`, and `rlm` objects.

For each replay-ready step, ordering is exactly:

```text
observation → model_request → model_response → proposed_action → action_result
```

The observation's `model_input` PNG is the exact current image encoded into the request. The model request links that image plus every earlier model-input image retained in its complete message history, including each file path and hash. It also records the action schema hash, sampling/generation settings, model and processor revisions, and conversation/session IDs. The action result records before/after fixture state, local destination URL, success, fixture assertion, and a post-action debugging image.

The constrained action union is:

```json
{"action":"click","x":0,"y":0}
{"action":"scroll","delta_y":600}
{"action":"wait","milliseconds":250}
{"action":"finish","summary":"done"}
```

No additional action fields are accepted. Coordinates are bounded to the fixed 1280×800 viewport, scroll deltas to ±1600, and waits to three seconds.

## Exact input and redaction semantics

The outer local harness owns request construction, so each `frames/*-model-input.png` and its corresponding base64 image in the current or later `requests/*.json` are byte-for-byte the same PNG. Requests retain prior screenshot-bearing user turns in chronological order and append the current observation last. A convenience OS screenshot is never substituted for model input. Optional post-action screenshots are debugging artifacts only.

Configured redaction rectangles are painted solid black before PNG serialization or request construction. The manifest records the coordinates and method. A capture cannot claim to retain pre-redaction pixels.

## Validation

`holo-capture validate <bundle>` rejects:

- missing or unhashed files, path traversal, SHA/size mismatch;
- empty streams, non-contiguous sequence/step IDs, decreasing monotonic timestamps, or unknown event types;
- a normalized event with no valid raw-event lineage;
- missing required event types or broken observation/request/response/action/result linkage;
- an absent model-input image, wrong dimensions, or mismatched artifact hash;
- malformed or out-of-bounds actions;
- authorization/cookie/private-key/API-key patterns;
- any HTTP(S) URL whose host is not `localhost`, `127.0.0.1`, or `::1`.

Use `--allow-runtime-only` only for Holo CLI plumbing captures whose public event stream did not expose exact model inputs. Such a bundle explicitly has `replay_ready=false` and is ineligible for mechanistic replay.

## Replay contract

Replay resets the fixture to the manifest seed, applies `actions/*.json` in lexical order through the localhost fixture API, and compares both initial and terminal fixture state to `annotations.json`. This proves deterministic action/state replay. Replaying model decisions later additionally requires the exact request, screenshot, model revision, and processor configuration retained here.

Only observable annotations are permitted: target/destination, success and terminal reason, target-visible frame interval/region, whether the target was seen before leaving the viewport, and the second-turn step. Attention or other mechanistic interpretations are not annotations in v0.
