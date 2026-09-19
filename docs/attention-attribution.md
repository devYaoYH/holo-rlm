# Attention attribution to image patches

This document describes the verified happy path used by this project: value-norm-corrected cross-layer attention rollout, projected independently onto every image in the prompt, optionally followed by a strictly previous-token baseline. The output is a routing diagnostic for a generated token or structured action parameter.

The word *causal* below refers only to information timing: a token baseline may use earlier generated tokens, never future ones. Attention rollout is not proof that a highlighted pixel causally changed the model output.

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

The generated native tool call is parsed into parameter-value token spans such as `action = scroll`, `delta_y = 400`, or click coordinates. If a value occupies generated steps `S`, its target map is the arithmetic mean:

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

## 7. Display transform

Float32 maps are independently quantized for the self-contained viewer. Each map stores its maximum scale and uint8 codes:

```text
code = round(clamp(value / scale, 0, 1) * 255)
decoded = code / 255 * scale
```

The maximum absolute round-trip error is one half of a quantization step, `scale / (2*255)`, apart from floating-point tolerance. For coloring, negative and non-finite values become zero, the 99th percentile is the visual ceiling, and a power of `0.62` improves low-signal contrast. This per-view color scaling changes appearance, not the raw allocation or differential metric.

## 8. Computational verification

`tests/test_attribution.py` exercises the transformations with tiny tensors whose results are calculated by hand:

- exact image-token span selection and row-major patch reshaping;
- separate projection for multiple prompt images;
- grouped-query value-norm expansion and normalization over all keys;
- two-layer top-to-bottom rollout with the `0.5 attention + 0.5 residual` transition;
- exact expected rollout maps for prompt and generated query positions;
- multi-token parameter averaging;
- previous-only baseline subtraction, including a check that a future token cannot change a step-one baseline;
- an explicit failure when no previous token exists;
- viewer quantization round-trip error bounded by `scale / (2*255)`.

Run the focused proof suite with:

```bash
uv run pytest -q tests/test_attribution.py
```

## 9. Generate viewers

For one traced completion:

```bash
uv run holo-capture attribution data/traces/<trace-id>
```

For every completed action referenced by a trajectory bundle:

```bash
uv run holo-capture attribution data/trajectories/v0/<trajectory-id> --all-frames
```

The detailed viewer provides input-frame, generated-target, method, layer, head, opacity, patch-boundary, and previous-token-baseline controls. A trajectory viewer links every action to its detailed completion viewer and reports whether a click was captured.

To collect a bounded local trajectory with every normal tool-call token retained:

```bash
uv run holo-capture run --backend local --task cheapest --max-steps 5 --stop-on-click --trace-generation-steps 128
```

## Interpretation limits

These maps describe attention routing under a residual-rollout approximation. They do not establish causal necessity or sufficiency of a pixel region. The fixed residual weight is a modeling choice, hybrid linear-attention blocks are omitted, and the baseline is generated-token history rather than an independent resting condition. Strong causal claims require interventions such as patch ablation, activation patching, or controlled counterfactual inputs.
