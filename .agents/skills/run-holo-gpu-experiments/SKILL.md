---
name: run-holo-gpu-experiments
description: Operate this repository's Holo-3.1-4B experiments on a local or SSH-accessible NVIDIA GPU host. Use for CUDA preflight, server startup, ScreenSpot-Pro inference with token logprobs, multi-frame attention attribution, activation patching/ablation manifests, resumable shards, and safe result retrieval.
---

# Run Holo GPU Experiments

Use the repository's checked-in CLIs and manifests to run reproducible experiments. Read `references/remote-workflow.md` before executing commands on an SSH host.

## Establish the run contract

1. Resolve the repository root, checkpoint path, trace path, benchmark paths, output path, and requested experiment.
2. Keep the model API on `127.0.0.1`. Never print or persist SSH private keys, provider tokens, or access URLs.
3. Inspect the current branch/commit and dirty state. Do not overwrite remote results from a different commit.
4. Set `HOLO_MODEL_PATH`, `HOLO_TRACE_DIR`, `HOLO_DEVICE=cuda`, and `HOLO_DTYPE=bfloat16` explicitly.

## Preflight before GPU inference

Run `bash scripts/bootstrap_remote_gpu.sh` for a fresh host. On an already prepared host, run:

```bash
cd instrumented_server
uv sync --dev --frozen
uv run holo-gpu-doctor --require-cuda --output ../data/manifests/remote-gpu-doctor.json
uv run pytest
```

Stop before inference when CUDA, BF16, checkpoint shards, required metadata, tests, or output storage fail. Report the exact failed check. Do not silently switch to CPU.

## Start and smoke-test the server

Start `uv run instrumented-holo-server` from `instrumented_server/` in a persistent remote shell. Keep it bound to localhost. Use the checkpoint-native single-frame benchmark profile (`HOLO_EAGER_ATTENTION=0`, `HOLO_IMAGE_MAX_PIXELS=16777216`) for ScreenSpot-Pro. Restart with eager attention and an explicit, measured frame budget for attention capture; `HOLO_IMAGE_MAX_PIXELS=262144` is the conservative multi-frame starting point. From the repository root, set `HOLO_BASE_URL=http://127.0.0.1:8000/v1` and run `uv run holo-capture smoke --backend local`.

Expect the first request to load model weights. Capture the server log and doctor report with the run metadata.

## Choose the experiment

- Full or sharded ScreenSpot-Pro: use `holo-capture screenspot-benchmark`; require H Company's official localization protocol and the checkpoint-native image limit, default to `--trace-profile logprobs`, run `--count 1` first, then rely on resume.
- Multi-frame attribution: use `holo-capture trajectory-prompt-case` to capture a target plus controls, then `trajectory-prompt-contrast` or `attribution-batch` to render outputs.
- Activation intervention: use `holo-activation-patch MANIFEST --output DIR`; use `--save-activations` only when downstream analysis needs raw selected tensors.

Use the exact command shapes in `docs/remote-gpu-experiments.md`. Prefer existing example manifests under `benchmarks/` and copy them to a run-specific path before changing scientific choices.

## Preserve scientific invariants

For activation patching, keep clean/corrupted message text, tool schema, image count/order, tokenized shape, visual grids, and scored candidates aligned. Permit a pixel corruption only when it is declared in the manifest. The built-in `swap_equal_tiles` operation crops both equal-size, non-overlapping boxes before either paste and does no resampling.

Report the metric as a teacher-forced log-probability margin in nats:

```text
log p(correct scored tokens) - log p(distractor scored tokens)
```

Interpret `exp(margin)` as the sequence-probability ratio for the declared scoring spans. Treat a component as supported only when clean activation interchange restores the corrupted margin and clean-run ablation lowers it across the relevant test set.

## Package and retrieve

Run `holo-capture package-results` so the archive contains the run and only referenced traces. Download the `.tar.gz` and `.sha256` sidecar, verify locally, and inspect `bundle-manifest.json`. Do not delete remote traces or outputs until the local archive has been verified and extracted.

Summarize the commit, GPU, model/checkpoint identity, command, shard, completed/failed counts, output path, archive checksum, and any deviations from the manifest.
