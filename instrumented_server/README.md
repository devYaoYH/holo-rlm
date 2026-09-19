# Instrumented Holo server

This is a local OpenAI-compatible endpoint for the native Holo-3.1-4B checkpoint. It is intentionally not an Ollama/vLLM replacement: when a request opts into tracing, it loads the native Transformers model with eager attention and writes focused attention rows, token IDs, vision-grid metadata, and optional KV-cache tensors.

## Start

```bash
cd instrumented_server
uv sync
HOLO_DEVICE=mps HOLO_DTYPE=auto uv run instrumented-holo-server
```

The server listens on `http://127.0.0.1:8000/v1` and does not load model weights until the first completion.

`HOLO_DEVICE=auto` selects Metal on Apple Silicon. The loader uses Accelerate's meta-device skeleton and reads safetensors directly to the target one tensor at a time: it does **not** create a full CPU model and then call `.to("mps")`, nor does it materialize a whole 6.4 GB checkpoint shard. On Metal, `HOLO_DTYPE=auto` keeps the checkpoint's native BF16 precision. The loader opens each shard as a memory map, materializes one tensor directly on Metal, installs it into the meta-device skeleton with an explicit BF16 target dtype, drops the temporary reference, and empties unused MPS cache at each shard boundary. The explicit setter dtype is essential because an Accelerate meta skeleton otherwise defaults to FP32 and silently doubles the checkpoint's final residency.

The server refuses dtype-converting Metal loads by default. Converting this BF16 checkpoint with `HOLO_DTYPE=float16` can cause the MPS allocator to retain both source and converted allocations and approach 2× weight memory. `HOLO_ALLOW_DTYPE_CONVERSION=1` is an explicit escape hatch, not a recommended setting.

On this approximately 18 GiB usable-memory Mac, that still leaves a tight practical budget for macOS, the vision encoder, activations, and KV cache. Start with a one-token smoke test and short inputs. `HOLO_LOAD_STRATEGY=stream` is the only supported loader mode.

The validated native-BF16 load used 6.02 GiB after shard one and 9.64 GiB after both shards. The server logs these MPS figures at every shard boundary and asserts that no parameter remains on `meta` or silently changes dtype.

Historical-frame requests bound each image to 262,144 pixels before visual tokenization (`HOLO_IMAGE_MAX_PIXELS`, with a 65,536-pixel minimum). This keeps exact source PNGs in the trace while preventing full prompt-attention matrices from growing beyond Metal memory as frames accumulate. The effective runtime limits and resulting `image_grid_thw` are stored with every trace.

## Trace request

Send screenshots as base64 data URLs. The server deliberately refuses remote URLs so a trace is self-contained and inference does not fetch browser-provided addresses.

```json
{
  "model": "Hcompany/Holo-3.1-4B",
  "messages": [{"role": "user", "content": "Describe the screen."}],
  "max_tokens": 32,
  "trace": {
    "capture_attentions": true,
    "capture_hidden_states": true,
    "capture_kv": false,
    "capture_value_norms": true,
    "capture_rollout": true,
    "max_generation_steps": 4
  }
}
```

Trace bundles are written to `data/traces/` and must remain uncommitted. They contain the exact request/response and model-input image bytes, input/generated token IDs, image-grid metadata, checkpoint and processor provenance, focused attention rows, and per-layer last-query hidden states. `capture_value_norms` stores one L2 norm per KV head and sequence key. `capture_rollout` stores float16 head-mean square prompt matrices, both direct and value-weighted, for cross-layer rollout; it is on by default and materially increases trace size. `capture_kv: true` writes the much larger full tensors and remains off by default.

## HoloDesktop route

After the server passes a direct API smoke test, point HoloDesktop at it with:

```bash
holo run --base-url http://127.0.0.1:8000/v1 "<bounded test-fixture task>"
```

The endpoint accepts OpenAI `tools`, renders them through the checkpoint's native tool template, parses native Holo/Qwen XML calls back into OpenAI `tool_calls`, strips hidden thinking, and stops generation after `</tool_call>`. Keep HoloDesktop runs bounded and restricted to the local fixture.
