# Holo 3.1 attention attribution: presenter notes

## Eleven-minute run of show

| Time | Slide | Talk track |
|---|---|---|
| 0:00-0:35 | 1 | Ask what is specific to the current instruction after removing generic visual awareness. |
| 0:35-1:15 | 2 | Separate static ScreenSpot grounding from replayable multi-turn visual memory. |
| 1:15-2:10 | 3 | Establish the full-benchmark replication, then locate the error concentration in icon targets across action families. |
| 2:10-3:20 | 4 | Explain value-norm, eight-layer residual rollout, parameter-token aggregation, and both baselines. |
| 3:20-4:20 | 5 | Establish the headline attribution metric: same-image instruction differential. |
| 4:20-5:15 | 6 | Static miss: coarse task region is salient, but the tiny affordance loses. |
| 5:15-6:25 | 7 | Multi-turn probe: target evidence is measurable in an earlier frame and stronger in the current frame, yet the click misses the button. |
| 6:25-7:30 | 8 | Present the balanced four-item aggregate: Layer 19 ranks first narrowly, with a strong multi-head cluster rather than a unique H10 feature. |
| 7:30-8:20 | 9 | Define the lossless tile swap and teacher-forced coordinate sequence log-likelihood ratio. |
| 8:20-9:25 | 10 | Show activation restoration and ablation together. Contrast target residuals with the head-level null. |
| 9:25-10:20 | 11 | Scale the protocol into a frozen causal evaluation set. |
| 10:20-11:30 | 12 + demo | Recap the routing-versus-mechanism distinction, then show the matched pair and intervention chart. |

## Exact claims to make

- **Full ScreenSpot-Pro replication:** the official element-localization harness completes all **1,581 items** with **1,042 strict hits**, for **65.9% accuracy**, **100% valid JSON**, and zero failed requests. This is **0.6 percentage point** below the project owner's reported 4B reference of about **66.5%**.
- **UI × action breakdown:** text targets reach **79.4%**, while icon targets reach **44.0%**. The weakest labeled cell is **icon + file/transfer at 36.1% (13/36)**, and the larger **icon + navigation/reveal** cell is also weak at **40.3% (60/149)**. Windows icons are the largest weak platform cell at **39.6% (150/379)**. Action families come from a deterministic first-verb keyword mapping for diagnosis and are not an official ScreenSpot-Pro taxonomy.
- **ScreenSpot visual hit:** prompt-differential target lift is **16.08x**, versus **3.85x** for raw rollout. Its peak falls inside the annotated target. Leave-one-control-out cosine is **0.925 mean / 0.828 minimum**.
- **ScreenSpot miss:** the prompt-differential peak is outside the target and **9.9% of the frame diagonal** from its center. The repaired click is `(331, 270)`, on the slide thumbnail rather than the New Slide button. Stability is **0.970 mean / 0.953 minimum**.
- **Multi-turn hotel probe:** after replaying the same four screenshots and three-scroll history, the cheapest hotel's button has **2.78x** differential lift when it first appears in frame 2 and **8.59x** in the final frame. The probe clicks `(960,120)` on the correct hotel card, outside the button. Three controls are included; one is excluded for missing a `y` span. Stability is **0.898 mean / 0.769 minimum**. The 1280×800 source becomes a 640×384 processor raster and a 20×12 merged-token map, so the 105×65-pixel button spans only about **1.6×1.0 visual tokens**.
- **Shifted-layout diagnostic:** frozen item `test-0035` moves the price column to about **58%** of viewport width and enlarges the UI. The £122 target receives **4.12x** causal previous-token lift for x and **4.01x** for y; the £135 runner-up receives **0.81x / 0.91x**. Holo still emits malformed, off-target coordinates. This supports position robustness but is not a same-image prompt-baseline result.
- **Balanced Layer 19 result:** under the official harness, the frozen pilot contains **four items: two hits and two misses**, each with four visible same-image controls at the 2,097,152-pixel cap. Layer 19 ranks first at **158.4x mean target lift**, narrowly ahead of Layer 15 at **151.4x**. Layer 19 is top on two of four items and has median item rank 2.
- **Head cluster:** the top four aggregate heads are **L19/H11 312.1x, L19/H14 293.8x, L19/H10 264.7x, and L19/H2 245.6x**. H10 ranks third within Layer 19 and is never the top Layer 19 head on an individual pilot item.
- **Matched corruption:** the clean coordinate margin is **+1.700 nats**, equivalent to **5.47×** target preference. The tile swap reverses it to **-2.644 nats**, equivalent to **14.07×** distractor preference.
- **Causal residual result:** all layer-19 visual residuals recover **1.853 nats / 42.7%** of the gap. Four target patches recover **1.444 nats / 33.2%**, and their clean-run ablation removes **1.481 nats**.
- **Head-level null:** layer 19/head 10 restores only **0.024 nats / 0.5%** and its ablation removes **0.090 nats**. Its observational saliency does not establish a sufficient head-level mechanism.
- **Fine-tuning delta lens:** teacher-forcing identical candidates through official Qwen3.5-4B and Holo3.1-4B shifts the final oracle margin toward the oracle on **5/5 ScreenSpot probes**, by **+1.14 nats per scored token on average** with a **+0.31 to +2.42** range. Three probes cross from a negative base margin to a positive Holo margin. The three hotel decisions shift by **−1.13, +6.37, and +0.37 nats per token**, so the multi-turn effect is selective rather than uniformly positive.
- **Scope:** the two ScreenSpot cases use different images and prompts. They are not a matched success/failure causal pair. The hotel target is a matched re-probe over captured history, not the exact original final model call.
- **Causal scope:** the tile swap is one matched case. It supports a regional residual-stream claim for this coordinate contrast, not a population claim about Holo.

## Ninety-second live demo

1. Open the [ScreenSpot hit viewer](../../data/attributions/screenspot-powerpoint_windows_59/viewer.html).
2. Start on **Raw value-norm rollout**, then select **Target minus diverse-instruction baseline**. Show the peak entering the template and the 16.08x lift.
3. Sort heads by **Prompt difference** and point out layer 19 / head 10 as a strong direct value-norm head in this selected case, then contrast that with the balanced aggregate where H11 and H14 rank higher.
4. Open the [ScreenSpot miss viewer](../../data/attributions/screenspot-powerpoint_windows_48/viewer.html). Show the task-region signal and the click on the slide thumbnail.
5. Open the [multi-frame hotel viewer](live/hotel-cheapest-multiframe/viewer.html).
6. Select **Target minus diverse-instruction baseline**. Step from frame 2 to frame 3. The green box marks the target button; the high-contrast white ring/cross on frame 3 marks the failed click.
7. End on the balanced layer/head table as a source of intervention hypotheses, not a causal conclusion.
8. Show the clean/corrupted tile pair, then the intervention chart. Contrast the target-patch result with the layer-19/head-10 null.

## Five-minute Q&A crib sheet

### Is attention a faithful explanation?

Not by itself. The matched experiment shows why: the strongest direct attention head has high observational target lift but recovers only 0.024 nats when patched alone. Target-region residuals pass both restoration and ablation tests in this case.

### What exactly is the causal metric?

We teacher-force two equal-length native tool calls and sum log probabilities over only their six x/y value tokens. The reported margin is target-coordinate log probability minus distractor-coordinate log probability. Syntax tokens do not enter the score.

A nat is one natural-log unit. Exponentiating the margin gives the target-to-distractor sequence-probability ratio. This is a sequence log-likelihood ratio, not a raw attention score or a single-token logit difference. “Gap recovered” divides the additive restoration in nats by the 4.344-nat clean-minus-corrupted gap; it is not a percentage of probability mass.

### Is this activation patching or pixel swapping?

Both operations appear, at different stages. The pixel swap creates the minimally corrupted input by cropping both equal-size tiles from the original image and pasting them into each other’s boxes without resize or resampling. Activation patching then replaces one internal tensor in the corrupted forward pass with the corresponding tensor captured from the clean pass. The ablation control removes the same component from the clean pass.

### Why use both patching and ablation?

Patching asks whether clean state can restore the corrupted decision. Ablation asks whether the same state supports the clean decision. The target-patch residual result is stronger because both tests point in the predicted direction, while distractor patches move both in the opposite direction.

### Why multiply by the value-vector norm?

High attention can carry little signal when the corresponding value vector is near zero. We use `normalize(A * ||V||2)` before rollout, which suppresses strong-but-empty paths. This still omits value direction and downstream nonlinear effects.

### What is cross-layer rollout here?

At each captured full-attention block, row-normalize the value-weighted attention, mix it 50/50 with identity for residual flow, and multiply the eight matrices in model order. Holo is hybrid: its interleaved linear-attention blocks do not expose an equivalent square matrix, so they are outside this rollout.

### What exactly is the prompt baseline?

For each control instruction, run the same model, processor, decoding settings, image bytes, and, for the hotel case, the same action and frame history. L1-normalize each coordinate-token map across all retained image patches, average controls, then subtract from the target map. Leave-one-control-out cosine checks sensitivity to any one control.

### Why include a malformed hotel control?

One logo control emitted invalid tool JSON but retained complete `x` and `y` parameter spans, so it is usable for a semantic coordinate-token baseline even though it is not executable. A different control omitted `y` entirely and is excluded. The viewer and `analysis.json` disclose the exclusion.

### Does the hotel trace prove long-term memory?

No. It shows that the final action's differential attribution allocates positive mass to the target button in an earlier retained frame and more mass in the current frame. That is consistent with cross-frame retrieval. A causal memory claim needs frame removal, shuffling, or patch intervention.

### Could downsampling explain the hotel miss?

It is a plausible contributor, not an isolated cause. After patch merging, the button is about one token high, so the model can preserve coarse hotel identity while losing affordance-level localization. But the four-frame history and longer prompt change at the same time. Test a factorial sweep: fixed task and layout across vision-token budgets, plus a UI-scale sweep that enlarges text and controls while keeping price order and position fixed. Uniformly shrinking the screenshot is not enough because it shrinks the UI too. A practical multi-scale input is a low-resolution full frame plus a high-resolution content or price-strip crop; it adds image context without introducing an environment-specific memory tool.

### Why is the uncertainty statement on slide 8 so cautious?

The new slide aggregates four independent items rather than treating heads as replicates. The item-level bootstrap range for Layer 19 is 23.2x to 293.7x, which is intentionally described as a range for this frozen pilot rather than a population confidence interval. A larger preregistered cohort should bootstrap by item or trajectory, not by token, layer, or head.

### Which head looks most promising?

The balanced aggregate points to a Layer 19 cluster rather than a single winner. H11 ranks first, followed by H14, H10, and H2. H10 remains relevant because it was the original candidate and is third overall, but the matched activation-patching experiment finds that H10 alone restores just 0.024 nats. The next intervention should patch the cluster and low-rank Layer 19 subspaces rather than declaring one specialized head.

### Why did full-resolution multi-frame tracing take so long?

The model weights remain streamed to Metal at about 9.64 GiB. The expensive part is the first full-attention prompt matrix over four high-resolution frames, whose memory grows quadratically with prompt length and causes swap pressure. This motivates frame retrieval or patch compression for research instrumentation.

### Can layer statistics become confidence?

Potentially, but not from a four-item pilot. The new balanced set is enough to audit the earlier selected-case claim, not to train or validate a confidence model. Next freeze a larger task-distributed cohort, train any outcome or ambiguity readout on a disjoint split, and evaluate held-out Brier score, log loss, expected calibration error, and selective risk. Safety refusal and clarification thresholds should be chosen from calibrated risk, not raw saliency.

### Is base Qwen versus Holo interesting?

Yes. The completed delta lens teacher-forces the same image, prompt, and candidate action tokens through both checkpoints. Holo increases the final oracle margin on all five ScreenSpot probes, but the hotel trajectory is mixed. The mean ScreenSpot delta becomes most negative at layer 22 and flips positive at layers 30–31, which points to a late transformation or readout effect. Because each checkpoint uses its own final RMSNorm and unembedding, the next analysis should separate residual changes from readout changes before assigning the effect to a specific internal representation.

### What should we train next?

Keep a frozen diverse test split. Generate and inspect oracle SFT trajectories, measure mid-training lift, then run GRPO with LoRA and a KL penalty using verifiable environment rewards. Use attribution for audit and diagnosis, not as the reward target.

## Sources and reproducibility

- ScreenSpot-Pro paper: <https://arxiv.org/abs/2504.07981>
- Official repository: <https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding>
- Full-benchmark measurements: `data/remote-results/run/screenspot-full-official/summary.json`
- Benchmark breakdown script: `scripts/summarize_screenspot_benchmark.py`
- Slide-ready benchmark summary: `artifacts/screenspot-presentation/screenspot-benchmark-breakdown.json`
- Frozen case/control manifest: `benchmarks/screenspot_presentation_cases.json`
- Method: `docs/attention-attribution.md`
- Case study: `docs/screenspot-case-study.md`
- Balanced Layer 19 pilot: `docs/screenspot-balanced-layer19.md`
- Frozen balanced manifest: `benchmarks/attention_attribution/screenspot_balanced_layer19_v1.json`
- Aggregate measurements: `data/remote-results/attention-layer19-balanced-v1/aggregate/aggregate.json`
- Causal method and results: `docs/causal-intervention.md`
- Causal measurements: `artifacts/causal-intervention/powerpoint_windows_59_swap/results.json`
- Static measurements: `data/attributions/screenspot-powerpoint_windows_{59,48}/analysis.json`
- Tracked hotel viewer and measurement: `artifacts/screenspot-presentation/live/hotel-cheapest-multiframe/`
- Hotel probe manifest: `data/trajectory-prompt-cases/hotel-cheapest-final-640/case.json` (directory name is historical; manifest records full-resolution `1280x800` frames and `frame_scale=1.0`)
- Revised deck: `artifacts/screenspot-presentation/holo-attribution-research-v16.pptx`
