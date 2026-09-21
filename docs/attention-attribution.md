# Attention attribution to image patches

This document describes the verified happy path used by this project: value-norm-corrected cross-layer attention rollout, projected independently onto every image in the prompt, followed by either a strictly previous-token baseline or a matched same-image instruction-ensemble baseline. The output is a routing diagnostic for a generated token or structured action parameter.

The word *causal* below refers only to information timing: a token baseline may use earlier generated tokens, never future ones. Attention rollout is not proof that a highlighted pixel causally changed the model output.

The separate [matched causal coordinate intervention](causal-intervention.md) performs activation patching and ablation against a teacher-forced coordinate-margin metric. It should not be confused with the observational rollout described here.

## 1. Captured tensors

For one batch-size-one completion, let:

- `P` be the prompt length after the processor expands every image into visual tokens;
- `t` be a zero-based generated-token step;
- `L` be the ordered conventional full-attention blocks;
- `Hq` and `Hkv` be the query- and KV-head counts.

The instrumented server stores:

- `input_ids.npy`, including the exact image-token positions;
- each exact `model-input-NNN.png` and its captured `image_grid_thw`;
- last-query attention rows `A[l,t]` with shape `[Hq, P+t]`;
- prompt attention matrices with shape `[P,P]`, averaged over query heads;
- value-vector L2 norms with shape `[Hkv, keys]` for each full-attention block;
- generated token IDs and tokenizer pieces.

Full KV tensors and generated-token hidden states are not needed for this path. The local loader streams checkpoint shards directly to Metal, while the trace retains only the smaller attribution artifacts above.

Holo 3.1 is a hybrid model. Its linear-attention blocks do not expose square softmax-attention matrices, so matrix rollout includes all conventional full-attention blocks in model order and explicitly omits the interleaved linear-attention blocks.

## 2. Recover each image patch grid

For every input image, the loader finds one contiguous image-token span in `input_ids`. Images and spans are paired in prompt order.

For a captured vision grid `[T, Hg, Wg]` and processor merge size `m`, the displayed visual-token grid is:

```text
rows    = Hg / m
columns = Wg / m
tokens  = T * rows * columns
```

The division must be exact, `T` must currently equal one, and `tokens` must equal the length of that image's prompt span. These checks prevent silently reshaping attention onto the wrong geometry.

Attention values at the image-token positions are selected without renormalizing the image slice, then reshaped in processor token order to `[rows, columns]`. Not renormalizing is deliberate: the map sum continues to express how much flow reached this frame instead of forcing every frame to total one.

## 3. Correct attention by value magnitude

An attention coefficient alone says how strongly a key is routed, but not how much signal its value vector carries. Query head `h` is mapped to its grouped-query KV head `g(h)`. Before rollout, each row is weighted by the corresponding value norm:

```text
W[l,t,h,k] = A[l,t,h,k] * ||V[l,g(h),k]||2
```

For rollout, weighted rows are averaged across query heads and then row-normalized over **all** available keys, not merely image keys:

```text
M[l,t,k] = mean_h W[l,t,h,k]
Mhat[l,t,:] = M[l,t,:] / sum_j M[l,t,j]
```

Normalizing before selecting image positions preserves competition from text, earlier images, and generated tokens. A zero denominator produces a zero row rather than NaNs.

## 4. Compose attention through layers

For generation step `t`, the prediction query position is:

```text
q(t) = P + t - 1
```

At `t=0`, this is the final prompt position that predicts the first output token. At later steps it is the preceding generated-token position.

For each full-attention layer, the loader constructs the causally available transition rows for the current sequence prefix:

- prompt rows come from the captured `[P,P]` prompt matrix;
- generated position `u` uses the captured last-query row from generation step `u`;
- no row can point to a future position.

Every row is normalized, then mixed equally with the residual path:

```text
T[l,t] = 0.5 * row_normalize(M[l,t]) + 0.5 * I
```

Starting with one unit of influence at the prediction query, rollout proceeds from the highest captured full-attention block to the lowest:

```text
r[L] = one_hot(q(t))
r[l] = r[l+1] @ T[l,t]       for l = L-1 ... 0
```

The attribution for input frame `f` is the final influence selected at that frame's image-token positions and reshaped to its verified patch grid:

```text
rollout[t,f] = reshape(r[0][image_positions[f]], rows[f], columns[f])
```

This is full cross-layer rollout over every captured conventional attention block. It is not a single-layer heatmap.

## 5. Aggregate a structured action parameter

Generated output is parsed into parameter-value token spans: native function calls provide values such as `action = scroll` or `delta_y = 400`, while the official element-localization JSON provides `x` and `y`. If a value occupies generated steps `S`, its target map is the arithmetic mean:

```text
target[f] = mean_t_in_S rollout[t,f]
```

The mean, rather than a sum, prevents tokenizer length from giving a multi-token value more attribution merely because it contains more pieces.

## 6. Subtract a strictly causal token baseline

The viewer's **Subtract previous-token baseline** control asks what is elevated for the selected token or parameter relative to the completion so far.

Let `s` be the first generated step in the target span. The reference set contains only captured steps strictly before `s`:

```text
previous = {t | t < s}
baseline[f] = mean_t_in_previous rollout[t,f]
difference[f] = target[f] - baseline[f]
```

For a multi-token value, baseline membership stops before the first value token; later pieces of the value never leak into its own reference. For a single token, no later generated token can affect the baseline. A target beginning at step zero has no causal token baseline, so the reference implementation raises an explicit error and the viewer displays an empty differential overlay.

The signed difference remains available computationally. The heatmap colors only:

```text
positive_difference = max(difference, 0)
```

Negative patches are transparent rather than being misrepresented as positive saliency. The viewer matches the baseline by input frame, attribution method, selected layer, and selected head.

This is analogous to condition-minus-baseline analysis, but it is not a separately measured resting state. A true null-task control would require a second matched inference request and additional assumptions about prompt comparability.

## 7. Subtract a same-image diverse-instruction baseline

The prompt-comparison viewer adds that separately measured control. For a static case, it sends the exact same image bytes through the same model, processor, tool schema, decoding settings, and click-only task format while changing only the instruction. For a trajectory decision, every probe also retains the same ordered frame history and assistant action history. Controls name visible targets distributed across the interface rather than paraphrasing the target instruction.

For the target and each control request, coordinate attribution averages the captured `x` and `y` value-token maps. Each request is then L1-normalized over every image patch before requests are combined:

```text
request_map[r,f] = mean_t_in_(x,y) rollout[r,t,f]
normalized[r,f] = request_map[r,f] / sum_over_all_frames_and_patches(request_map[r])
prompt_baseline[f] = mean_r_in_controls normalized[r,f]
prompt_difference[f] = normalized[target,f] - prompt_baseline[f]
```

Normalizing each request first prevents a completion with more total image allocation from dominating the control mean. The result is signed: positive patches carry more routing mass for the target instruction than for the matched instruction ensemble, while negative patches carry less. The viewer uses a diverging color map and preserves both signs.

Before subtraction, the implementation requires identical image SHA-256 values for every retained frame, frame order, pixel dimensions, patch grids, model metadata, processor metadata, full-attention layer identities, and head counts. A mismatch fails closed rather than resampling incompatible maps. Multi-frame requests are normalized jointly across every patch in every frame, so a historical frame's mass remains comparable with the current frame instead of being forced to sum to one independently.

Control-set sensitivity is reported with leave-one-control-out cosine similarity. A high minimum similarity means the differential direction does not depend on one convenient control prompt. This does not make the attribution causal: all requests remain observational forward passes, and prompt wording may change generation dynamics beyond visual grounding.

## 8. Spatial, layer, and head diagnostics

When a benchmark target box is available, each patch receives the fraction of its area covered by the box. For the positive part of a map, the viewer reports:

- target mass: normalized positive attribution falling inside the box, including fractional boundary patches;
- target lift: target mass divided by the box's image-area fraction, where `1.0` is uniform allocation;
- peak distance: distance between the peak patch center and box center, normalized by the image diagonal;
- normalized entropy: spatial dispersion from zero (concentrated) to one (uniform).

The final rollout averages query heads at each layer before matrix composition, so it has no post hoc head axis. Head and layer diagnostics therefore inspect the value-norm-corrected direct maps immediately before rollout. For every conventional full-attention layer and query head, the analysis computes raw, previous-token differential, and prompt-ensemble differential target metrics. Layer summaries average across heads and identify the best aligned head. These rankings are exploratory diagnostics from a small number of cases, not evidence of a globally specialized GUI-grounding head.

## 9. Display transform

Unsigned single-trace maps and signed prompt-difference maps use separate portable encodings. A signed comparison map stores its maximum absolute scale and int8 codes:

```text
code = round(clamp(value / scale, -1, 1) * 127)
decoded = code / 127 * scale
```

The maximum absolute round-trip error is one half of a signed quantization step, `scale / (2*127)`, apart from floating-point tolerance. The comparison viewer renders positive values in warm colors and negative values in blue, using a power of `0.62` to improve low-signal contrast. This display transform changes appearance, not the raw allocation or differential metric.

## 10. Computational verification

`tests/test_attribution.py` exercises the transformations with tiny tensors whose results are calculated by hand:

- exact image-token span selection and row-major patch reshaping;
- separate projection for multiple prompt images;
- grouped-query value-norm expansion and normalization over all keys;
- two-layer top-to-bottom rollout with the `0.5 attention + 0.5 residual` transition;
- exact expected rollout maps for prompt and generated query positions;
- multi-token parameter averaging;
- previous-only baseline subtraction, including a check that a future token cannot change a step-one baseline;
- an explicit failure when no previous token exists;
- exact-image validation and rejection of mismatched prompt-control traces;
- per-request L1 normalization, control averaging, and signed target-minus-control subtraction;
- joint L1 normalization across two input frames, per-frame boxes, metrics, and previews;
- multi-frame request cloning that keeps image bytes and assistant actions fixed while changing only instruction semantics;
- fractional target-box overlap and target-lift calculations;
- viewer quantization round-trip error bounded by `scale / (2*255)` for unsigned maps and `scale / (2*127)` for signed comparisons.

Run the focused proof suite with:

```bash
uv run pytest -q tests/test_attribution.py tests/test_contrast.py tests/test_trajectory_contrast.py
```

## 11. Generate viewers

For one traced completion:

```bash
uv run holo-capture attribution data/traces/<trace-id>
```

For every completed action referenced by a trajectory bundle:

```bash
uv run holo-capture attribution data/trajectories/v0/<trajectory-id> --all-frames
```

The detailed viewer provides input-frame, generated-target, method, layer, head, opacity, patch-boundary, and previous-token-baseline controls. A trajectory viewer links every action to its detailed completion viewer and reports whether a click was captured.

For one ScreenSpot-Pro target plus matched same-image controls:

```bash
uv run holo-capture screenspot-case <sample-id> \
  --trace-generation-steps 64 \
  --control-prompt "<matched click instruction 1>" \
  --control-prompt "<matched click instruction 2>" \
  --control-prompt "<matched click instruction 3>" \
  --control-prompt "<matched click instruction 4>"

uv run holo-capture screenspot-contrast data/screenspot-pro/runs/<sample-id>/case.json
```

The contrast viewer compares raw value-norm rollout, the strictly previous-token differential, the diverse-instruction mean, and the signed target-minus-instruction differential on the same screen. It also displays the benchmark box, predicted click, control prompts, stability score, and ranked layer/head statistics.

For a captured final trajectory decision, replay the exact frame and action history under one target instruction and several visible control targets, then build the same comparison viewer:

```bash
uv run holo-capture trajectory-prompt-case data/traces/<final-trace-id> \
  --target-instruction "<target click instruction>" \
  --control-prompt "<matched visible control 1>" \
  --control-prompt "<matched visible control 2>" \
  --target-bbox X1,Y1,X2,Y2 \
  --frame-bbox none \
  --frame-bbox X1,Y1,X2,Y2 \
  --trace-generation-steps 64

uv run holo-capture trajectory-prompt-contrast data/trajectory-prompt-cases/<case>/case.json
```

Supply exactly one `--frame-bbox` per retained frame. Use `none` when the target is not visible. `--max-frame-width` is available for a uniformly resampled, lower-memory probe; the manifest records the scale and all target boxes are scaled with the frames.

To collect a bounded local trajectory with every normal tool-call token retained:

```bash
uv run holo-capture run --backend local --task cheapest --max-steps 5 --stop-on-click --trace-generation-steps 128
```

## 12. Interpretation limits

These maps describe attention routing under a residual-rollout approximation. They do not establish causal necessity or sufficiency of a pixel region. The fixed residual weight is a modeling choice, hybrid linear-attention blocks are omitted, and both token-history and prompt-ensemble baselines depend on reference choices. Strong causal claims require interventions such as patch ablation, activation patching, or controlled counterfactual inputs.
