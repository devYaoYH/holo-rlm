# Matched causal coordinate intervention

This experiment tests whether components suggested by the attention analysis causally support one GUI coordinate choice. It uses the local native-BF16 Holo 3.1 4B checkpoint and the same processor and tool prompt as the ScreenSpot-Pro case study.

## Matched screenshot pair

The clean case is `powerpoint_windows_59`, with the instruction **Create a “Psychedelic vibrant” presentation**. The corrupted screenshot losslessly swaps the complete `Psychedelic vibrant` tile with the adjacent, equally sized `Woven fibers` tile. The code crops both 200 × 172 rectangles from the original image before either paste, then pastes each crop into the other rectangle. It performs no resize, interpolation, or re-encoding between the two crops. No pixels outside those rectangles change, and the swap changes 0.905% of the 2880 × 1800 canvas.

The fixed coordinate candidates are the centers of the original target and distractor boxes:

| Candidate | Normalized coordinate | Pixel box |
|---|---:|---:|
| Original target slot | `(482, 351)` | `(1289, 563, 1489, 702)` |
| Adjacent distractor slot | `(406, 351)` | `(1069, 563, 1269, 702)` |

Both native tool calls contain six coordinate tokens and identical syntax. The metric excludes tool-syntax tokens:

```text
coordinate margin = sum ln p(original-target-slot x/y tokens) - sum ln p(distractor-slot x/y tokens)
```

A **nat** is one natural-log unit. Exponentiating a margin converts it to a sequence-probability ratio: `exp(margin) = P(target coordinate tokens) / P(distractor coordinate tokens)`. The clean margin of `+1.700` nats therefore means the target coordinate sequence is `5.47×` as likely as the distractor sequence under teacher forcing. The corrupted margin of `-2.644` nats means the target sequence is `0.071×` as likely, or the distractor is `14.07×` as likely. The resulting clean-minus-corrupted gap is `4.344` nats.

This is a sequence log-likelihood ratio, not a raw attention score and not a single-token logit difference. Each candidate conditions later coordinate tokens on its own earlier teacher-forced tokens. The two candidates contain the same six scored coordinate tokens in aggregate length, so summed log probabilities are directly comparable. For candidates with unequal scored-token counts, use the library's mean reduction or define a length-matched contrast.

## Interventions

The model uses a 12 × 20 merged visual-token grid for this image. Target and distractor regional tests each retain the four patches with the largest box overlap so patch count cannot explain the comparison.

Clean-to-corrupted patching replaces one corrupted activation with the activation from the clean run under the same teacher-forced candidate sequence. Clean-run ablation uses:

- image-mean replacement for all visual-token residuals;
- mean replacement from the remaining image tokens for target or distractor regions;
- zero ablation for one attention-head slice or one MLP output at coordinate-prediction queries.

Layer 19/head 10 modifies that head’s gated output immediately before the layer’s `o_proj`. MLP tests modify the module output at the query positions that predict the x/y value tokens. Each row uses a separate forward pass.

## Result

Positive restoration means clean activations recover the original-target-slot margin in the corrupted run. Positive ablation drop means removing that component reduces the margin in the clean run. “Gap recovered” is the additive restoration in nats divided by the `4.344`-nat clean-minus-corrupted gap. It is not a percentage of probability mass.

| Component | Restoration (nats) | Gap recovered | Clean ablation drop (nats) |
|---|---:|---:|---:|
| Layer 19, all visual-token residuals | +1.853 | 42.7% | +0.846 |
| Layer 19, four target-patch residuals | +1.444 | 33.2% | +1.481 |
| Layer 19, four distractor-patch residuals | -0.130 | -3.0% | -0.515 |
| Layer 19, head 10 attention output | +0.024 | 0.5% | +0.090 |
| Layer 15 MLP output | +0.082 | 1.9% | +0.202 |
| Layer 19 MLP output | -0.429 | -9.9% | +0.252 |
| Layer 23 MLP output | -0.479 | -11.0% | -1.055 |

The regional residual result passes both directions of the test: clean target-patch states partially restore the corrupted margin, and mean-replacing those same states in the clean run removes nearly the entire original `+1.700`-nat preference. The matched distractor region moves both tests in the opposite direction.

Layer 19/head 10 had the strongest direct prompt-differential target lift in the observational analysis, but its isolated intervention recovers only `0.024` nats. The attention map therefore supplied a useful layer and spatial hypothesis, not a sufficient head-level mechanism. The tested MLP outputs also show no consistent bidirectional support.

This is one matched case. It establishes a reproducible causal case study, not a population-level claim about Holo. The next rigorous step is a frozen set of target-distractor swaps with pre-registered coordinates, repeated runs where nondeterminism applies, and uncertainty intervals over intervention effects.

## Reusable library

The case script now uses `instrumented_holo.activation_patching.ActivationPatchingRunner`. The library accepts arbitrary clean and corrupted message histories, including multiple retained screenshots, two teacher-forced candidate completions with explicit scored character spans, named regions on any input frame, and either summed or mean token log probability. It validates token and image-grid alignment before a clean activation is injected into a corrupted run.

See [Activation patching library](activation-patching-library.md) for the API and the invariants required when extending the diagnostic to another screenshot or trajectory pair.

## Reproduce

```bash
cd instrumented_server
HOLO_DEVICE=mps HOLO_DTYPE=auto uv run holo-causal-experiment
```

The command writes the exact clean/corrupted images and machine-readable measurements to `artifacts/causal-intervention/powerpoint_windows_59_swap/`. The recorded checkpoint revision is `624347ebe6ab4df6f9701bcfc3c1bdfcfbc20b56`.
