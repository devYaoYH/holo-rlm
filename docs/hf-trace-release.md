# Holo trace release audit and viewer

This audit covers the trace bundles currently under `data/`, including retrieved GPU runs. It was run on 2026-09-24 with `python3 scripts/audit_trace_release.py --verify-hashes`. The machine-readable report is `data/release-audit/audit.json` (git-ignored). It checks every completed trace manifest's file set, byte length, and SHA-256, token-count agreement, capture-step layout, and simple text patterns for secrets and email addresses. It does **not** OCR screenshots or establish redistribution rights.

## Inventory

| Source | Complete traces | Bytes | Residual traces | Token log-probability traces |
| --- | ---: | ---: | ---: | ---: |
| ScreenSpot resolution pilot and expansion, shared trace directory | 360 | 36.26 GB | 0 | 360 |
| Local `data/traces/` | 62 | 3.20 GB | 30 | 0 |
| Remote hotel and paired free generation | 11 | 3.50 GB | 0 | 0 |
| **Total** | **433** | **42.97 GB** | **30** | **360** |

The 360 ScreenSpot traces consist of 120 requests from a six-case pilot and 240 from a twelve-case expansion, with four image scales and target/control prompts. Every trace appears in one of the two run summaries. All 360 capture attention for every generated token (11–14 tokens each). The local residual traces contain 1,590 captured generation-step states across the 30 bundles. Ten additional directories have a request but no completed manifest and are excluded. Eight early local bundles omit the explicit full-attention layer map; seven omit model revision. One remote paired trace is from Qwen rather than Holo and must remain a separate comparison.

The full traces use about 37.1 GB for quadratic prompt attention matrices and about 5.0 GB for last-query attention rows. These matrices are useful for rollout but are not needed for stepwise direct attention inspection. Archive files and split transfers under `data/remote-results/` are additional copies or containers; this inventory counts extracted, completed trace directories, not archive bytes. The SHA-256 check found no mismatch. The text-pattern scan found no obvious secret or email in `request.json`, `response.json`, or `completion.json`; this is not a privacy clearance.

## Local release candidates

Run:

```bash
python3 scripts/audit_trace_release.py --verify-hashes
python3 scripts/stage_hf_traces.py --profile screenspot
python3 scripts/stage_hf_traces.py --profile residual-review
python3 scripts/stage_hf_traces.py --profile hotel-trajectory-review
python3 scripts/stage_hf_traces.py --profile hotel-action-review
python3 scripts/verify_hf_stage.py data/hf-stage/screenspot
python3 scripts/verify_hf_stage.py data/hf-stage/residual-review
python3 scripts/verify_hf_stage.py data/hf-stage/hotel-trajectory-review
python3 scripts/verify_hf_stage.py data/hf-stage/hotel-action-review
```

- `data/hf-stage/screenspot/`: 360 Holo ScreenSpot traces, about 2.71 GB of staged files. The index links each trace to its sample, resolution, target/control role, target box, predicted click, correctness label, and run-manifest hash. It includes images, generated tokens, chosen-token log probabilities, last-query attention rows, and small value-norm arrays.
- `data/hf-stage/residual-review/`: 24 local traces with explicit model revision and residual states, about 0.95 GB of staged files. This is an internal review set. Input screenshots and raw requests are omitted. Six earlier residual traces without a model revision and other local traces are excluded pending provenance review.
- `data/hf-stage/hotel-trajectory-review/`: one replay-ready, successful synthetic `test-0010` Holo replicate with three actions and 3 trace bundles, about 0.21 GB. Its attention rows cover only steps 0-63 of 302, 282, and 363 generated tokens. These rows do **not** reach the generated scroll/click parameters. It is useful for multi-frame early-token inspection and behavioral replay, but not a coordinate-token attention study.
- `data/hf-stage/hotel-action-review/`: one replay-ready, successful synthetic `test-0035-large-ui` Holo diagnostic with four actions and 4 trace bundles, about 1.00 GB. The final decision captures the freely generated click x and y value-token attention at steps 246-248 and 252-254. It is a different fixture and run from the Holo/Qwen paired `test-0010` free-generation pilot. Its third decision saves only the first 256 of 355 generated-token attention steps.

The matched Holo/Qwen coordinate-token attention experiment is a separate **teacher-forced probe**, recorded under `data/local-results/attention-delta-hotel-step2-native-v1/`. It reconstructs the attention rows predicting six declared oracle coordinate tokens for both checkpoints, then stores three-frame direct-attention and value-norm maps, layer mass, and comparison summaries. It is not a free-generation trajectory and is not included in the 433 raw trace count or the hotel stages above. The paired free-generation `test-0010` pilot has Holo and Qwen behavioral trajectories, but only a separately replicated Holo run and one failed-first-action Qwen run have raw activation traces; those captures stop attention at step 63. Do not describe the early-token maps as action-token attention.

Staging uses local hard links by default, so those apparent bytes do not occupy another full copy on this filesystem. Use `--copy` for an independent folder. Each bundle gets a new manifest for its actual staged files. The staging code excludes raw requests and responses, machine-specific model paths, processor file lists, and prompt attention matrices. The dataset card intentionally leaves its license unset while image and derived-data rights are reviewed. Do not upload the `residual-review` profile before checking local input provenance; hidden vectors can retain information about their inputs even without screenshots.

The [Holo model card](https://huggingface.co/Hcompany/Holo-3.1-4B) currently lists Apache-2.0 and the [ScreenSpot-Pro dataset](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) lists MIT. These labels do not by themselves settle the rights to redistribute every screenshot or derivative. [Hugging Face dataset cards](https://huggingface.co/docs/hub/en/datasets-cards) support provenance and limitations metadata. After review, the current [Hub upload command](https://huggingface.co/docs/huggingface_hub/main/guides/upload) is `hf upload <namespace>/<dataset> <folder> --repo-type=dataset`.

## Trace Atlas

```bash
python3 -m apps.trace_viewer.server --root data
# Open http://127.0.0.1:8765
```

For a smaller selection, point `--root` at one trace directory or a staged `traces/` directory. The server binds to `127.0.0.1` and serves a read-only API. It loads NPZ members on demand. Use the generation slider or Play button to advance through tokens, click one of the 32 language blocks, choose a head on a conventional full-attention block, and select an input frame for its patch heatmap. Where captured, the viewer shows all 2,560 post-block residual dimensions or the change added by that block, arranged in a hoverable index grid. It shows generated token pieces, IDs, and log probabilities when present.

The query at generation step `t` predicts output token `t`; at step zero it is the last prompt position, and afterward it is the preceding generated token position. Holo's eight conventional full-attention blocks expose attention rows. Its interleaved linear-attention blocks do not expose equivalent softmax rows in this capture, though their residual output vectors can be shown when saved. Attention maps are routing observations, and vector intensity is not a causal explanation. The existing activation-patching experiments are the appropriate basis for causal claims.
