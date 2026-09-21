# Remote GPU experiments

This workflow packages the Holo instrumentation into two installable Python projects: the root trajectory/evaluation harness and `instrumented_server`, which owns native model loading, token log probabilities, and interventions. It is designed for a private RTX 5090 or A100 host reached over SSH. No command uploads artifacts or credentials automatically.

## 1. Transfer and bootstrap

Clone or pull the repository on the host, place the unpacked Holo checkpoint and ScreenSpot-Pro data on attached storage, then set explicit paths:

```bash
export HOLO_MODEL_PATH=/workspace/models/Holo-3.1-4B
export HOLO_TRACE_DIR=/workspace/holo-results/traces
export HOLO_DEVICE=cuda
export HOLO_DTYPE=bfloat16
bash scripts/bootstrap_remote_gpu.sh
```

The bootstrap uses both lockfiles, runs both test suites, and writes `data/manifests/remote-gpu-doctor.json`. It does not install `uv`, download the model, or fetch benchmark data. Inspect the doctor report before spending GPU time; it verifies CUDA/BF16 support, checkpoint metadata and shards, package versions, and free trace storage.

For a Vast.ai image, choose a recent PyTorch/CUDA base image, expose only SSH, and keep the API bound to `127.0.0.1`. Do not commit SSH keys, access tokens, checkpoint files, ScreenSpot images, traces, or result bundles.

## 2. Start and verify the server

Use a persistent shell such as `tmux`:

```bash
cd instrumented_server
HOLO_SERVER_HOST=127.0.0.1 uv run instrumented-holo-server
```

In a second shell:

```bash
export HOLO_BASE_URL=http://127.0.0.1:8000/v1
uv run holo-capture smoke --backend local
```

The first request loads the checkpoint. Keep the server process and all experiment commands in the same environment so `HOLO_TRACE_DIR` resolves consistently.

## 3. ScreenSpot-Pro with generated-token log probabilities

Start with one item:

```bash
uv run holo-capture screenspot-benchmark \
  --annotations data/screenspot-pro/annotations \
  --images data/screenspot-pro/images \
  --trace-root "$HOLO_TRACE_DIR" \
  --output /workspace/holo-results/screenspot-smoke \
  --trace-profile logprobs \
  --count 1
```

Then run the full benchmark. The runner writes one atomic result per sample and resumes completed items by default:

```bash
uv run holo-capture screenspot-benchmark \
  --annotations data/screenspot-pro/annotations \
  --images data/screenspot-pro/images \
  --trace-root "$HOLO_TRACE_DIR" \
  --output /workspace/holo-results/screenspot-full \
  --trace-profile logprobs
```

`token_logprobs.json` stores every generated token's natural-log probability, probability, cumulative log probability, character span, and action/x/y parameter assignment. The benchmark summary retains action/coordinate aggregates while the referenced trace keeps the full token sequence. To distribute work, give each process a distinct output and `--shard-index K --num-shards N`; avoid serving concurrent jobs from one model process until memory behavior has been measured.

## 4. Multi-frame attention attribution

Capture a target plus controls over an existing trace containing several historical frames:

```bash
uv run holo-capture trajectory-prompt-case data/traces/trace-example \
  --target-instruction "Click the cheapest available hotel" \
  --control-prompt "Click the hotel with the highest rating" \
  --target-bbox 120,240,420,330 \
  --frame-bbox none \
  --frame-bbox 120,240,420,330 \
  --trace-generation-steps 64 \
  --output /workspace/holo-results/trajectory-case
```

Provide exactly one `--frame-bbox` per retained input frame. The request captures direct attention, value-norm-corrected attention, rollout, and every generated token's log probability. Render many traces or trajectories with the portable manifest in `benchmarks/attribution_batch.example.json`:

```bash
uv run holo-capture attribution-batch benchmarks/attribution_batch.example.json \
  --trace-root "$HOLO_TRACE_DIR" \
  --output /workspace/holo-results/attribution-viewers
```

## 5. Activation patching and image swaps

The CLI consumes a serializable experiment manifest:

```bash
cd instrumented_server
uv run holo-activation-patch \
  ../benchmarks/activation_patching/powerpoint_tile_swap.json \
  --validate-only
uv run holo-activation-patch \
  ../benchmarks/activation_patching/powerpoint_tile_swap.json \
  --output /workspace/holo-results/powerpoint-patching \
  --save-activations
```

The example's corruption is an exact equal-size tile swap: both rectangles are cropped from the clean PNG first and pasted into the opposite locations with no resizing or resampling. Everything outside the two rectangles remains byte-for-byte unchanged after decoding.

Each representation uses a pyvene-inspired declarative shape:

```json
{
  "name": "layer_19_target_residuals",
  "representation": {
    "layer": 19,
    "component": "residual_output",
    "unit": "image_region",
    "region": "target"
  },
  "intervention": "interchange",
  "ablation": "mean"
}
```

Supported components are `residual_output`, `mlp_output`, and `attention_head_output`. Residuals can select all image tokens, one image, or a named pixel region; MLP and attention-head interventions currently select the prediction positions for the scored candidate tokens. `interchange` patches the captured clean activation into the corrupted run. Its paired ablation replaces selected clean residuals with the mean of non-selected image tokens, or zeros selected MLP/head outputs.

The primary metric is `margin_nats = log p(correct coordinate tokens) - log p(distractor coordinate tokens)`. A nat is a natural-log unit: a +1 nat margin means the correct coordinate sequence is `e^1 ≈ 2.72` times as probable as the distractor under the declared teacher-forced scoring spans. Positive restoration and positive clean-run ablation drop jointly support a causal role.

See [activation-patching-library.md](activation-patching-library.md) for the Python API and alignment requirements.

## 6. Package and retrieve results

From the repository root, create an archive containing the run plus only its referenced traces:

```bash
uv run holo-capture package-results /workspace/holo-results/screenspot-full \
  --trace-root "$HOLO_TRACE_DIR" \
  --output /workspace/holo-results/screenspot-full.tar.gz
sha256sum -c /workspace/holo-results/screenspot-full.tar.gz.sha256
```

Download both the archive and checksum with `rsync` or `scp`, verify the checksum locally, and inspect `bundle-manifest.json`. Do not delete the remote copy until extraction and validation succeed.
