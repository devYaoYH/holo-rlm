# Base-Qwen versus Holo delta lens

This experiment compares Qwen3.5-4B and Holo3.1-4B at identical teacher-forced action tokens. It is designed to support the fine-tuning-delta research direction on slide 10 without conflating Holo's output-format reliability with its visual/action preference.

## Frozen comparison contract

The checked-in manifest is `benchmarks/delta_lens/qwen35_4b_vs_holo31_4b.json`. Regenerate it only through the official request builders:

```bash
uv run python scripts/build_delta_lens_manifest.py
```

The manifest contains:

- five locally available ScreenSpot-Pro cases using `hcompany_element_localization_v1`: image first, H Company's localization prompt and `VisualLocalizerOutput` schema, thinking disabled, and no desktop tool;
- three decisions from `test-0035-large-ui`, retaining one, two, and three full 1024×720 screenshots respectively, using the checked-in Holo desktop-action system prompt and normalized tool schema;
- one oracle and one declared diagnostic distractor per decision;
- a fixed Holo processor and chat template for both checkpoints, so token IDs, visual grids, image order, and action suffixes must align exactly.

The ScreenSpot distractor for `powerpoint_windows_59` is the adjacent matched theme tile from the existing causal intervention. The other four are prior Holo clicks from the same screenshots and therefore support exploratory diagnostics only. Do not describe their margins as pre-registered causal effects.

The hotel history uses native Holo scroll direction. Its common oracle sequence is:

```text
scroll(delta_y=-500)
scroll(delta_y=-500)
click(x=642, y=616)
```

The first two probes score `scroll` against a premature `click`. The final probe scores the cheapest visible button against a visible non-cheapest button. Only declared semantic value spans enter each score; shared XML/JSON syntax does not. Candidate scores are the mean log probability per scored token, which avoids length bias when coordinates or actions tokenize to different lengths.

## Estimator

For every transformer layer, the scorer applies that checkpoint's final RMSNorm and tied language-model head to the residual stream at each next-token prediction position. For a forced token sequence `a`, the primary delta is:

```text
delta(layer, a) = log p_Holo(a | prompt, prior forced tokens)
                - log p_Qwen(a | prompt, prior forced tokens)
```

For an oracle/distractor pair, the margin delta is:

```text
[log p_Holo(oracle) - log p_Holo(distractor)]
- [log p_Qwen(oracle) - log p_Qwen(distractor)]
```

Positive values mean fine-tuning increased oracle preference relative to the base model. This is a layerwise logit-lens diagnostic, not a causal intervention. It uses each checkpoint's own final norm and unembedding, so the delta includes changes in both the residual representation and readout weights.

## Validation and run

Validation resolves every image and candidate without loading either checkpoint:

```bash
cd instrumented_server
./.venv/bin/python -m instrumented_holo.delta_lens \
  ../benchmarks/delta_lens/qwen35_4b_vs_holo31_4b.json \
  --validate-only
```

For a local Metal diagnostic, place the official base checkpoint at `models/Qwen3.5-4B`, keep the Holo checkpoint at `models/Holo-3.1-4B`, and run sequential loading:

```bash
cd instrumented_server
HOLO_DEVICE=mps \
HOLO_DTYPE=auto \
HOLO_EAGER_ATTENTION=0 \
./.venv/bin/python -m instrumented_holo.delta_lens \
  ../benchmarks/delta_lens/qwen35_4b_vs_holo31_4b.json \
  --output ../data/delta-lens/qwen35-4b-vs-holo31-4b
```

The manifest sets the checkpoint-native 16,777,216-pixel ceiling for every case. ScreenSpot images therefore retain their source resolution, and each hotel frame remains 1024×720 rather than being reduced to the former 262,144-pixel safety profile. The three-frame prompt uses non-eager attention and is intended for a CUDA host with preflighted capacity. The local Metal path remains available for a one-case smoke test but is not the planned full run.

The output directory contains `base.json`, `tuned.json`, `delta-lens.json`, and a manifest snapshot. The scorer rejects any checkpoint pair whose input-token hash, image grids, candidate token IDs, scored offsets, or layer indices fail to align.

## Completed remote run

The verified RTX 5090 run is stored at `data/remote-results/delta-lens-qwen35-vs-holo31-20260921-dcd8c5e5/`. Its archive SHA-256 is `d5e0c1d1c48725b200a7776a7b2fb4b16f5894fb19ce1c1fb633171252dc8745`.

- Official Qwen snapshot: `Qwen/Qwen3.5-4B` at Hugging Face revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Holo checkpoint: `Hcompany/Holo-3.1-4B`, model revision `624347ebe6ab4df6f9701bcfc3c1bdfcfbc20b56`.
- Runtime: CUDA 13.0, BF16, non-eager attention, Transformers 5.17.0, and the shared Holo processor at the native 16,777,216-pixel ceiling.
- ScreenSpot: all five final oracle-minus-distractor margins shift toward the oracle, with a mean Holo-minus-base change of `+1.14` nats per scored token and a `+0.31` to `+2.42` range. Three probes cross from a negative base margin to a positive Holo margin.
- Hotel trajectory: final margin changes are `−1.13`, `+6.37`, and `+0.37` nats per scored token for the first scroll, second scroll, and final click decisions.

These are eight diagnostic probes, not a population estimate. Four ScreenSpot distractors are exploratory prior Holo clicks. The layerwise projection also uses each checkpoint's own final RMSNorm and unembedding, so a follow-up should separate residual-stream changes from readout-weight changes.

## Input audit

Run the processor-only audit before inference:

```bash
PYTHONPATH=instrumented_server/src \
instrumented_server/.venv/bin/python scripts/audit_delta_lens_inputs.py \
  benchmarks/delta_lens/qwen35_4b_vs_holo31_4b.json \
  --output data/manifests/delta-lens-input-audit.json
```

The audit fails unless every request and candidate uses a 0–1000 coordinate schema and every source image stays below its declared pixel ceiling without material processor downsampling. The current audited grids retain 99.6–100% of ScreenSpot source area and 97.8% of each 1024×720 hotel frame; the small differences are patch-grid alignment, not the former low-resolution cap. The final hotel oracle is stored as `(642, 616)` in normalized space and projects through the official harness to pixel `(657, 443)` in the 1024×720 viewport.

## Vast.ai deployment bundle

Create the minimal package locally:

```bash
uv run python scripts/package_delta_lens_deployment.py
```

This writes `data/deployment/holo-delta-lens-vastai.tar.gz` and its `.sha256` sidecar. The archive includes code, lockfiles, tests, the frozen manifest, the five ScreenSpot images and their three official application annotation files, the three hotel frames, and the processor audit. It excludes checkpoints, credentials, Git metadata, traces, and prior results.

After transferring and verifying the archive on an authorized CUDA host, set explicit unpacked checkpoint and output paths:

```bash
export HOLO_MODEL_PATH=/workspace/models/Holo-3.1-4B
export QWEN_MODEL_PATH=/workspace/models/Qwen3.5-4B
export DELTA_OUTPUT_DIR=/workspace/holo-results/qwen35-4b-vs-holo31-4b
bash scripts/run_delta_lens_remote.sh
```

The runner refuses to overwrite an existing output. It runs both locked test suites, CUDA/BF16 doctor checks for both checkpoints, the processor input audit, a two-checkpoint one-case smoke at `${DELTA_OUTPUT_DIR}-smoke`, sequential base/tuned scoring over the complete manifest, result packaging, and checksum verification. It does not create, stop, or delete a Vast.ai instance.
