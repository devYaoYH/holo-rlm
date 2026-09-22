# Holo 3.1 attention attribution: presenter notes

## Thirteen-minute run of show

| Time | Slide | Talk track |
|---|---|---|
| 0:00-0:35 | 1 | Ask what is specific to the current instruction after removing generic visual awareness. |
| 0:35-1:15 | 2 | Separate static ScreenSpot grounding from replayable multi-turn visual memory. |
| 1:15-2:10 | 3 | Establish the full-benchmark replication and show the raw UI-by-action breakdown. |
| 2:10-2:55 | 4 | Show the observed target-area distributions before interpreting the raw icon gap. |
| 2:55-4:05 | 5 | Explain value-norm, eight-layer residual rollout, parameter-token aggregation, and both baselines. |
| 4:05-5:05 | 6 | Use one native-resolution free-generation hit to compare causal prior-token subtraction with a same-image diverse-instruction baseline. |
| 5:05-6:00 | 7 | Show a native-resolution free-generation failure whose diverse-instruction-subtracted peak is inside the target even though the click misses by 7.4 pixels. |
| 6:00-7:10 | 8 | Follow Holo's free multi-turn hotel rollout across retained frames and actions. |
| 7:10-8:15 | 9 | Present the balanced four-item aggregate: Layer 19 ranks first narrowly, with a strong multi-head cluster rather than a unique H10 feature. |
| 8:15-9:05 | 10 | Define the lossless tile swap and teacher-forced coordinate margin. |
| 9:05-10:00 | 11 | Show activation restoration and ablation together. Contrast target residuals with the head-level null. |
| 10:00-11:05 | 12 | Compare Qwen and Holo action margins, ScreenSpot attention, and three-frame hotel allocation. |
| 11:05-11:55 | 13 | Scale the protocol into a frozen causal evaluation set. |
| 11:55-12:45 | 14 + demo | Recap the routing-versus-mechanism distinction, then show the matched pair and intervention chart. |
| Optional | 15 | Use one real instruction per UI × action cell to make the diagnostic taxonomy concrete. |

## Exact claims to make

- **Full ScreenSpot-Pro replication:** the official element-localization harness completes all **1,581 items** with **1,042 strict hits**, for **65.9% accuracy**, **100% valid JSON**, and zero failed requests. This is **0.6 percentage point** below the project owner's reported 4B reference of about **66.5%**.
- **UI label and target size:** ScreenSpot-Pro labels the clicked target, not the instruction semantics. An icon target has no text hint; a target with a text label is classified as text even when an icon is also present. Raw text accuracy is **79.4%**, versus **44.0%** for icons. The median normalized text-target area is **4.77×** the icon median, driven mostly by median width (**111 px versus 26 px**) rather than height (**25 px versus 24 px**). The distribution slide reports the observed bucketed geometry rather than an adjusted accuracy estimate. Target size is an important confound, but the histogram does not estimate how much of the accuracy gap it causes.
- **UI × action breakdown:** the weakest labeled raw cell is **icon + file/transfer at 36.1% (13/36)**, and the larger **icon + navigation/reveal** cell reaches **40.3% (60/149)**. Treat the matrix as a diagnostic breakdown rather than a causal effect of UI type. Action families come from a deterministic first-verb keyword mapping and are not an official ScreenSpot-Pro taxonomy.
- **Baseline comparison on a native free-generation hit:** on `powerpoint_windows_63`, Holo freely emits normalized `(209,243)`, projecting to `(601.9,437.4)` pixels inside the annotated Fill control. With value-norm rollout fixed, causal prior-token subtraction places **6.75%** of positive residual mass in the target (**168.8× lift**), while subtraction of four same-image diverse instructions places **18.58%** there (**464.4× lift**). The control ensemble has a **0.991** minimum leave-one-control-out cosine. The baseline is an analysis choice: the causal map asks what exceeds prior completion state, while the diverse ensemble asks what is specific to this instruction after removing shared image saliency.
- **Native free-generation action-precision miss:** on `powerpoint_windows_54`, Holo freely emits `(124,73)`, projecting to `(357.1,131.4)`. The x coordinate is inside the target width, while y is **7.4 pixels below** the annotated bottom edge. After subtracting four same-image diverse instructions, the peak patch is inside the oracle and **17.15%** of positive residual mass falls there (**278.6× lift**), with a **0.951** leave-one-control-out floor. This is consistent with successful target-specific routing followed by an imprecise action readout.
- **Axis-stripe audit:** the old `powerpoint_windows_48` prompt-difference map averages x- and y-token queries. When split, **51.4% of positive x-token differential mass** lies in columns intersecting the tiny target, versus **32.4%** in its rows. The conspicuous vertical stripe therefore comes primarily from x-coordinate routing. It is not a tensor-indexing bug or evidence that an entire image column shares one residual feature; it is an axis-factorized pattern amplified by averaging coordinate-token queries and by controls with different forced coordinates.
- **Shifted-layout diagnostic:** frozen item `test-0035` moves the price column to about **58%** of viewport width and enlarges the UI. The £122 target receives **4.12x** causal previous-token lift for x and **4.01x** for y; the £135 runner-up receives **0.81x / 0.91x**. Holo still emits malformed, off-target coordinates. This supports position robustness but is not a same-image prompt-baseline result.
- **Balanced Layer 19 result:** under the official harness, the frozen pilot contains **four items: two hits and two misses**, each with four visible same-image controls at the 2,097,152-pixel cap. Layer 19 ranks first at **158.4x mean target lift**, narrowly ahead of Layer 15 at **151.4x**. Layer 19 is top on two of four items and has median item rank 2.
- **Layer architecture:** Layers 15 and 19 are both full-attention layers in Holo3.1-4B. The official configuration uses full attention at zero-based layers **3, 7, 11, 15, 19, 23, 27, and 31**, with Gated DeltaNet linear attention in the intervening layers. Our attention rollout measures only those eight full-attention blocks.
- **Head cluster:** the top four aggregate heads are **L19/H11 312.1x, L19/H14 293.8x, L19/H10 264.7x, and L19/H2 245.6x**. H10 ranks third within Layer 19 and is never the top Layer 19 head on an individual pilot item.
- **Matched corruption:** the clean coordinate margin is **+1.700 nats**, equivalent to **5.47×** target preference. The tile swap reverses it to **-2.644 nats**, equivalent to **14.07×** distractor preference.
- **Causal residual result:** all layer-19 visual residuals recover **1.853 nats / 42.7%** of the gap. Four target patches recover **1.444 nats / 33.2%**, and their clean-run ablation removes **1.481 nats**.
- **Head-level null:** layer 19/head 10 restores only **0.024 nats / 0.5%** and its ablation removes **0.090 nats**. Its observational saliency does not establish a sufficient head-level mechanism.
- **Fine-tuning delta lens:** teacher-forcing identical candidates through official Qwen3.5-4B and Holo3.1-4B shifts the final oracle margin toward the oracle on **5/5 ScreenSpot probes**, by **+1.14 nats per scored token on average** with a **+0.31 to +2.42** range. Three probes cross from a negative base margin to a positive Holo margin. The three hotel decisions shift by **−1.13, +6.37, and +0.37 nats per token**, so the multi-turn effect is selective rather than uniformly positive.
- **Paired instruction-control delta:** on `powerpoint_windows_59`, every checkpoint sees the target plus four visible alternate instructions on the same native-resolution image. Each task uses an independently curated oracle coordinate. The value-norm target-minus-control mass is **13.15% for Qwen** and **12.37% for Holo**, a **−0.78 percentage-point Holo-minus-Qwen change**. Minimum leave-one-control-out cosine is **0.949 / 0.950**. Raw attention moves the opposite way at **+1.36 points**. The primary estimator therefore does not support stronger target grounding in this case.
- **Hotel frame redistribution:** at the final three-frame hotel step, Holo value-norm image attention shifts from **36.8 / 15.0 / 48.2%** across earliest, middle, and current frames to **20.2 / 23.0 / 56.8%**. This moves **16.6 percentage points** away from the earliest frame and **8.0 / 8.6 points** toward the middle/current frames.
- **Resolution and coordinate audit:** the paired attention run keeps the native **16,777,216-pixel ceiling**. The 2880×1800 ScreenSpot image becomes a 56×90 merged-token grid and retains **99.6%** of source area; each 1024×720 hotel frame becomes 22×32 and retains **97.8%** after patch alignment. All click coordinates remain normalized to **0–1000**.
- **Scope:** the two ScreenSpot cases use different images and prompts. They are not a matched success/failure causal pair. The hotel target is a matched re-probe over captured history, not the exact original final model call.
- **Causal scope:** the tile swap is one matched case. It supports a regional residual-stream claim for this coordinate contrast, not a population claim about Holo.

## Ninety-second live demo

1. Open the [ScreenSpot hit viewer](../../data/attributions/screenspot-powerpoint_windows_59/viewer.html).
2. Start on **Raw value-norm rollout**, then select **Target minus diverse-instruction baseline**. Show the peak entering the template and the 16.08x lift.
3. Sort heads by **Prompt difference** and point out layer 19 / head 10 as a strong direct value-norm head in this selected case, then contrast that with the balanced aggregate where H11 and H14 rank higher.
4. Show the native `powerpoint_windows_48` value-norm overlay. The green box marks New Slide; the official `(66,61)` output is a strict hit.
5. Show the three native hotel value-norm overlays in `native-saliency-ppt48-hotel35-v1/`.
6. Step from frame 0 to frame 2. The final map resolves onto the cheapest visible View details button; emphasize that the coordinates are teacher-forced.
7. End on the balanced layer/head table as a source of intervention hypotheses, not a causal conclusion.
8. Show the clean/corrupted tile pair, then the intervention chart. Contrast the target-patch result with the layer-19/head-10 null.

## Five-minute Q&A crib sheet

### Is attention a faithful explanation?

Not by itself. The matched experiment shows why: the strongest direct attention head has high observational target lift but recovers only 0.024 nats when patched alone. Target-region residuals pass both restoration and ablation tests in this case.

### What does icon versus text mean, and is size controlled?

It describes the annotated click target. ScreenSpot-Pro defines an icon target as one with no text hint. If a text label appears in the target, the benchmark calls it text even when an icon is also present. The released annotations contain some apparent edge cases or label noise, so the appendix uses unambiguous examples.

The raw accuracy table does not control target geometry. That matters because strict accuracy is point-in-box and larger boxes have a wider hit tolerance. In our audit, the median normalized text box is 4.77 times the icon median. The dedicated histogram shows the full observed size distributions instead of reporting a model-adjusted gap. It establishes substantial imbalance in click-box geometry, but it does not determine how much of the performance difference size causes.

### What exactly is the causal metric?

We teacher-force two equal-length native tool calls and sum log probabilities over only their six x/y value tokens. The reported margin is target-coordinate log probability minus distractor-coordinate log probability. Syntax tokens do not enter the score.

A nat is one natural-log unit. Exponentiating the margin gives the target-to-distractor sequence-probability ratio. This is a sequence log-likelihood ratio, not a raw attention score or a single-token logit difference. “Gap recovered” divides the additive restoration in nats by the 4.344-nat clean-minus-corrupted gap; it is not a percentage of probability mass.

### Is this activation patching or pixel swapping?

Both operations appear, at different stages. The pixel swap creates the minimally corrupted input by cropping both equal-size tiles from the original image and pasting them into each other’s boxes without resize or resampling. Activation patching then replaces one internal tensor in the corrupted forward pass with the corresponding tensor captured from the clean pass. The ablation control removes the same component from the clean pass.

### Why use both patching and ablation?

Patching asks whether clean state can restore the corrupted decision. Ablation asks whether the same state supports the clean decision. The target-patch residual result is stronger because both tests point in the predicted direction, while distractor patches move both in the opposite direction.

### Why multiply by the value-vector norm?

High attention can carry little signal when the corresponding value vector is near zero. We use `normalize(A * ||V||2)` before rollout, which suppresses strong-but-empty paths. This still omits value direction and downstream nonlinear effects.

### Was the vertical stripe under the New Slide box a code bug?

Not a tensor-indexing bug. The old map averaged attention from the scored x and y coordinate-token queries. In the exact native-resolution decomposition, 51.4% of the positive x-token target-minus-control mass lies in the target's columns. That x-coordinate routing creates the vertical band. The y-token map does not reproduce the same column.

The visualization nevertheless overstates object-level saliency if the stripe is read as one coherent feature. The alternate instructions also use different forced coordinates, so subtraction mixes instruction conditioning with output-coordinate identity. The revised pre-delta-lens slides therefore use Holo's own freely generated action, compare the causal prior-token and diverse-instruction baselines explicitly, and treat the maps as routing diagnostics rather than causal estimators.

### What is cross-layer rollout here?

At each captured full-attention block, row-normalize the value-weighted attention, mix it 50/50 with identity for residual flow, and multiply the eight matrices in model order. Holo is hybrid: its interleaved linear-attention blocks do not expose an equivalent square matrix, so they are outside this rollout.

### What exactly is the prompt baseline?

For each control instruction, run the same model, processor, decoding settings, image bytes, and, for the hotel case, the same action and frame history. L1-normalize each coordinate-token map across all retained image patches, average controls, then subtract from the target map. Leave-one-control-out cosine checks sensitivity to any one control.

When each control is forced through a different coordinate, this estimates the combined instruction-plus-action route; it does not isolate instruction conditioning alone. A strict instruction-only control must hold the output coordinate string fixed across prompts.

### How are the new hotel controls constructed?

Every condition uses the same generic official hotel system prompt, neutral task setup, three image bytes, and two teacher-forced scroll actions. Only the final visible-target instruction changes. The four controls request Juniper's button, Ember's button, the Lumen name, or the StayLocal logo. Each uses an independently curated oracle coordinate, so malformed free generation cannot enter the control mean.

### Does the hotel trace prove long-term memory?

No. It shows that the final action's differential attribution allocates positive mass to the target button in an earlier retained frame and more mass in the current frame. That is consistent with cross-frame retrieval. A causal memory claim needs frame removal, shuffling, or patch intervention.

### Did the rerun remove the downsampling confound?

Yes for this descriptive probe. PowerPoint remains 2880×1800 and becomes a 56×90 merged-token grid; each hotel frame remains 1024×720 and becomes 22×32. Both use the checkpoint-native 16,777,216-pixel ceiling. This does not prove that resolution caused the older miss, because the prompt and trajectory protocol were corrected at the same time. A causal resolution claim still needs a factorial sweep that holds the prompt, task, and layout fixed while varying only the vision-token budget.

### Why is the uncertainty statement on slide 8 so cautious?

The new slide aggregates four independent items rather than treating heads as replicates. The item-level bootstrap range for Layer 19 is 23.2x to 293.7x, which is intentionally described as a range for this frozen pilot rather than a population confidence interval. A larger preregistered cohort should bootstrap by item or trajectory, not by token, layer, or head.

### Are Layer 15 and Layer 19 full-attention layers?

Yes. Both are full softmax-attention layers. Holo3.1-4B inherits the Qwen3.5-4B hybrid schedule: three Gated DeltaNet layers followed by one full-attention layer. With zero-based indexing, the full-attention layers are 3, 7, 11, 15, 19, 23, 27, and 31. The earlier suggestion that Layer 19 was linear attention was incorrect.

### Why do layer-lens studies often peak in the middle?

Several mechanisms can produce that shape. Early layers still build local visual and lexical features. Middle layers have enough depth to integrate the instruction with image tokens while retaining spatial detail. Later layers increasingly transform the residual stream toward the output distribution, action syntax, and next-token decision, so a spatial or semantic probe can decline even when the information still affects the answer. Residual scaling, normalization, and the probe definition can also move the apparent peak.

This explanation is a useful hypothesis, not a universal law. Our result is a prompt-differential target-concentration metric over only eight full-attention blocks, not raw activation magnitude. Layer 19 beats Layer 15 by only 4.7%, and the ordering varies by item. Prior probing work also warns that clean early/middle/late stories can be probe-dependent. The causal experiment supports target-region residuals in one case, but it does not establish a general Layer 19 mechanism.

### Which head looks most promising?

The balanced aggregate points to a Layer 19 cluster rather than a single winner. H11 ranks first, followed by H14, H10, and H2. H10 remains relevant because it was the original candidate and is third overall, but the matched activation-patching experiment finds that H10 alone restores just 0.024 nats. The next intervention should patch the cluster and low-rank Layer 19 subspaces rather than declaring one specialized head.

### Why did full-resolution multi-frame tracing take so long?

The model weights remain streamed to Metal at about 9.64 GiB. The expensive part is the first full-attention prompt matrix over four high-resolution frames, whose memory grows quadratically with prompt length and causes swap pressure. This motivates frame retrieval or patch compression for research instrumentation.

### Can layer statistics become confidence?

Potentially, but not from a four-item pilot. The new balanced set is enough to audit the earlier selected-case claim, not to train or validate a confidence model. Next freeze a larger task-distributed cohort, train any outcome or ambiguity readout on a disjoint split, and evaluate held-out Brier score, log loss, expected calibration error, and selective risk. Safety refusal and clarification thresholds should be chosen from calibrated risk, not raw saliency.

### Is base Qwen versus Holo interesting?

Yes. The completed delta lens teacher-forces the same image, prompt, and candidate action tokens through both checkpoints. Holo increases the final oracle margin on all five ScreenSpot probes, but the hotel trajectory is mixed. The mean ScreenSpot delta becomes most negative at layer 22 and flips positive at layers 30–31, which points to a late transformation or readout effect. Because each checkpoint uses its own final RMSNorm and unembedding, the next analysis should separate residual changes from readout changes before assigning the effect to a specific internal representation.

The paired attention diagnostic adds a different view. After subtracting four same-image alternate-instruction maps, value-norm target mass is 0.78 percentage point lower for Holo than Qwen, despite stable leave-one-control-out maps. Raw attention moves 1.36 points in the opposite direction. In the hotel trace, value-norm attention moves away from the earliest frame and toward the two more recent frames. These are descriptive routing changes, not evidence that attention itself caused the improved action margin.

### What should we train next?

Keep a frozen diverse test split. Generate and inspect oracle SFT trajectories, measure mid-training lift, then run GRPO with LoRA and a KL penalty using verifiable environment rewards. Use attribution for audit and diagnosis, not as the reward target.

## Sources and reproducibility

- ScreenSpot-Pro paper: <https://arxiv.org/abs/2504.07981>
- Official repository: <https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding>
- Full-benchmark measurements: `data/remote-results/run/screenspot-full-official/summary.json`
- Benchmark breakdown script: `scripts/summarize_screenspot_benchmark.py`
- Slide-ready benchmark summary: `artifacts/screenspot-presentation/screenspot-benchmark-breakdown.json`
- Appendix examples: `artifacts/screenspot-presentation/screenspot-category-examples.json`
- Target-size audit: `artifacts/screenspot-presentation/screenspot-target-size-analysis.json`
- Target-size audit script: `scripts/analyze_screenspot_target_size.py`
- Official Holo3.1-4B layer schedule: <https://huggingface.co/Hcompany/Holo-3.1-4B/blob/main/config.json>
- Middle-layer visual-information study: <https://arxiv.org/abs/2411.16724>
- Probe-dependence caution: <https://aclanthology.org/2022.coling-1.278/>
- Frozen case/control manifest: `benchmarks/screenspot_presentation_cases.json`
- Method: `docs/attention-attribution.md`
- Case study: `docs/screenspot-case-study.md`
- Balanced Layer 19 pilot: `docs/screenspot-balanced-layer19.md`
- Frozen balanced manifest: `benchmarks/attention_attribution/screenspot_balanced_layer19_v1.json`
- Aggregate measurements: `data/remote-results/attention-layer19-balanced-v1/aggregate/aggregate.json`
- Causal method and results: `docs/causal-intervention.md`
- Causal measurements: `artifacts/causal-intervention/powerpoint_windows_59_swap/results.json`
- Qwen-to-Holo layerwise delta: `data/remote-results/delta-lens-qwen35-vs-holo31-20260921-dcd8c5e5/`
- Paired ScreenSpot instruction-control delta: `data/local-results/attention-delta-ppt59-controls-v1/`
- Paired hotel attention delta: `data/local-results/attention-delta-hotel-step2-native-v1/`
- Slide-ready paired attention assets: `artifacts/screenspot-presentation/delta-lens-paired-controls-v1/`
- Native slide-7/8 control manifest: `benchmarks/attention_attribution/powerpoint48_hotel35_native_controls_v1.json`
- Native slide-7/8 paired results: `data/local-results/attention-delta-ppt48-hotel35-native-controls-v1/`
- Native slide-7/8 overlays: `artifacts/screenspot-presentation/native-saliency-ppt48-hotel35-v1/`
- Static measurements: `data/attributions/screenspot-powerpoint_windows_{59,48}/analysis.json`
- Tracked hotel viewer and measurement: `artifacts/screenspot-presentation/live/hotel-cheapest-multiframe/`
- Hotel probe manifest: `data/trajectory-prompt-cases/hotel-cheapest-final-640/case.json` (directory name is historical; manifest records full-resolution `1280x800` frames and `frame_scale=1.0`)
- Revised deck: `artifacts/screenspot-presentation/holo-attribution-research-v31.pptx`
