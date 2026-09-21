# ScreenSpot-Pro attribution case study

This case study transfers the attribution pipeline from the deterministic hotel fixture to two official ScreenSpot-Pro examples. The synthetic fixture remains the computational test bed; ScreenSpot-Pro supplies the presentation evidence.

New captures use H Company's official single-turn element-localization protocol: image first, the published localization prompt and JSON schema, `structured_outputs`, thinking disabled, and temperature zero. The trace records `inference_protocol = hcompany_element_localization_v1`. Earlier captures made with the repository's custom `desktop_action` tool prompt remain readable, but must not be compared with H's reported ScreenSpot-Pro accuracy.

## Fixed cases

The selected pair holds the application and operating system constant:

- `powerpoint_windows_59`: visual grounding success for **Create a “Psychedelic vibrant” presentation**;
- `powerpoint_windows_48`: grounding failure for **Create new slide**.

The exact prompts and control ensemble are frozen in `benchmarks/screenspot_presentation_cases.json`. Images and activation traces stay under ignored `data/` paths because the official screenshots are large and the traces are derived artifacts.

## Score separation

Malformed legacy captures sometimes contain punctuation inside an integer coordinate field, for example `"484>"`. The harness therefore reports three separate facts for both old and new artifacts:

1. `format_valid`: whether the response strictly satisfies the official two-integer JSON schema;
2. `grounding_correct`: whether the point is inside the official bounding box after the narrow, logged repair `first_unsigned_integer_per_coordinate_field`;
3. `correct`: the strict end-to-end result, which remains false when tool syntax is invalid.

The repair is for diagnosis only. It is never counted as an official ScreenSpot-Pro success.

## Same-image diverse-instruction baseline

Each target screenshot is rerun with four unrelated, visually grounded click instructions whose targets cover different regions of the same frame. Model, processor, image bytes, localization schema, decoding settings, layer set, and head count must match the target trace exactly.

For target instruction \(t\), control instructions \(c_1,\ldots,c_K\), image patches \(p\), and the value-norm cross-layer rollout \(R\), the coordinate map for request \(q\) is

\[
M_q(p)=\frac{\operatorname{mean}_{s\in S_{x,y}}R_{q,s}(p)}
{\sum_{p'}\operatorname{mean}_{s\in S_{x,y}}R_{q,s}(p')}.
\]

Normalize each request before averaging controls:

\[
B_{\text{prompt}}(p)=\frac{1}{K}\sum_{k=1}^{K}M_{c_k}(p),\qquad
D_{\text{prompt}}(p)=M_t(p)-B_{\text{prompt}}(p).
\]

Positive differential attribution means the target instruction routes more coordinate-token influence through that patch than the diverse control ensemble; negative attribution means less. This is an instruction contrast, not a causal intervention on pixels.

Baseline robustness is the cosine similarity between the full differential map and every leave-one-control-out differential map. Both the mean and worst case are shown in the live viewer.

## Reproduce

With the instrumented Metal server running:

```bash
uv run holo-capture screenspot-case powerpoint_windows_59 \
  --annotations data/screenspot-pro/annotations \
  --images data/screenspot-pro/images \
  --trace-generation-steps 64 \
  --control-prompt 'Create a blank presentation' \
  --control-prompt 'Select the Scientific discovery theme' \
  --control-prompt 'Select the Animal magnetism theme' \
  --control-prompt 'Open the File menu' \
  --output data/screenspot-pro/presentation/powerpoint_windows_59

uv run holo-capture screenspot-contrast \
  data/screenspot-pro/presentation/powerpoint_windows_59/case.json \
  --output data/attributions/screenspot-powerpoint_windows_59
```

Repeat with the failure case and its controls from the frozen manifest.

## Observed legacy-protocol results

The values below were captured before adoption of H's official localization protocol. They remain useful for checking the attribution implementation, but should be regenerated before they are used as evidence about official Holo3.1 benchmark behavior.

| Case | Raw lift | Causal-token lift | Prompt-differential lift | Prompt-differential peak | LOO cosine, mean / min |
|---|---:|---:|---:|---|---:|
| `powerpoint_windows_59` visual hit | 3.85× | 4.82× | 16.08× | inside target | 0.925 / 0.828 |
| `powerpoint_windows_48` grounding miss | 13.85× | 14.68× | 13.50× | outside target; 9.9% diagonal away | 0.970 / 0.953 |

For the visual hit, the raw value-norm rollout peaks on a generic top-left patch. Same-image prompt subtraction moves the positive peak inside the “Psychedelic vibrant” target. Direct per-head diagnostics peak at transformer layer 19, head 10, with 98.05× target lift; the layer-19 mean is 38.99×.

For the miss, the prompt differential concentrates in the correct top-left task neighborhood but peaks on the slide thumbnail rather than the tiny New Slide toolbar control. This is consistent with coarse semantic selection followed by fine-grained target-binding failure. It is not evidence that attention caused the miss.

## Interpretation guardrails

- ScreenSpot-Pro uses point-in-bounding-box scoring; attribution alignment is a separate diagnostic.
- Cross-layer rollout composes the model's captured conventional full-attention layers. Hybrid linear-attention blocks do not provide square softmax matrices and are not silently approximated.
- Per-head tables use direct value-norm-corrected attention because a multiplied cross-layer rollout no longer has a unique head identity.
- Results from two presentation cases are illustrative, not population-level evidence about all ScreenSpot-Pro tasks.

Sources: [H Company element-localization guide](https://hub.hcompany.ai/models-api/element-localization), [ScreenSpot-Pro paper](https://arxiv.org/abs/2504.07981), [official repository](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding), and [official dataset](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro).
