# Holo 3.1 attention attribution: presenter notes

## Eleven-minute main talk

| Time | Slide | Talk track |
|---|---|---|
| 0:00-0:35 | 1 | Ask what is specific to the current instruction after removing generic visual awareness. |
| 0:35-1:15 | 2 | Introduce static UI localization and multi-turn context retrieval as complementary capabilities. |
| 1:15-2:10 | 3 | Establish the full-benchmark replication and show the raw UI-by-action breakdown. |
| 2:10-2:55 | 4 | Show the observed target-area distributions before interpreting the raw icon gap. |
| 2:55-3:50 | 5 | Show paired success retention as screenshot resolution falls, then connect the larger icon loss to target geometry. |
| 3:50-4:30 | 6 | Show the completed six-case attribution pilot: target signal usually weakens with resolution, with one readout-failure counterexample. |
| 4:30-5:30 | 7 | Compare raw value-norm rollout with the same map after diverse-instruction subtraction. |
| 5:30-6:20 | 8 | Show a strict miss whose diverse-instruction-subtracted peak remains inside the target. |
| 6:20-7:20 | 9 | Follow the successful free hotel rollout and show that all three retained frames contribute to the final click. |
| 7:20-8:20 | 10 | Define the eight-prompt tile-swap intervention and the teacher-forced x-coordinate margin. |
| 8:20-9:20 | 11 | Compare clean-state restoration, zero-ablation, and the all-head sweep for Qwen and Holo. |
| 9:20-10:10 | 12 | Present the next causal and training experiments. |
| 10:10-11:10 | 13 | Close on resolution sensitivity, surviving target attention, multi-frame retrieval, and the Layer 15 causal result. |
| Backup | 14 | Explain value-norm rollout, coordinate-token aggregation, and the diverse-instruction baseline. |
| Backup | 15 | Use one real instruction per UI × action cell to make the diagnostic taxonomy concrete. |

## Exact claims to make

- **Full ScreenSpot-Pro replication:** the official element-localization harness completes all **1,581 items** with **1,042 strict hits**, for **65.9% accuracy**, **100% valid JSON**, and zero failed requests. This is **0.6 percentage point** below the project owner's reported 4B reference of about **66.5%**.
- **UI label and target size:** ScreenSpot-Pro labels the clicked target, not the instruction semantics. An icon target has no text hint; a target with a text label is classified as text even when an icon is also present. Raw text accuracy is **79.4%**, versus **44.0%** for icons. The median normalized text-target area is **4.77×** the icon median, driven mostly by median width (**111 px versus 26 px**) rather than height (**25 px versus 24 px**). The distribution slide reports the observed bucketed geometry rather than an adjusted accuracy estimate. Target size is an important confound, but the histogram does not estimate how much of the accuracy gap it causes.
- **Resolution success retention:** the paired cohort contains **36 ScreenSpot-Pro items that are strict hits at native resolution**, balanced across three applications and icon/text targets with size-spread sampling. The same-run native rerun remains **36/36**. Retention falls to **25/36 (69.4%)** at 75% linear resolution, **22/36 (61.1%)** at 50%, and **4/36 (11.1%)** at 25%. Wilson 95% intervals are **53.1–82.0%**, **44.9–75.2%**, and **4.4–25.3%**. At half resolution, text targets retain **15/18 (83.3%)** clicks versus **7/18 (38.9%)** for icons. This estimates paired retention among selected native successes, not unconditional benchmark accuracy.
- **UI × action breakdown:** the weakest labeled raw cell is **icon + file/transfer at 36.1% (13/36)**, and the larger **icon + navigation/reveal** cell reaches **40.3% (60/149)**. Treat the matrix as a diagnostic breakdown rather than a causal effect of UI type. Action families come from a deterministic first-verb keyword mapping and are not an official ScreenSpot-Pro taxonomy.
- **Baseline comparison on a native free-generation hit:** on `powerpoint_windows_63`, Holo freely emits normalized `(209,243)`, projecting to `(601.9,437.4)` pixels inside the annotated Fill control. Raw value-norm rollout places **4.43%** of positive mass in the target (**110.7× lift**). Subtracting four same-image diverse instructions raises the task-specific concentration to **18.58%** (**464.4× lift**), with a **0.991** minimum leave-one-control-out cosine. The subtraction removes image saliency shared across visible alternate tasks.
- **Native free-generation action-precision miss:** on `powerpoint_windows_54`, Holo freely emits `(124,73)`, projecting to `(357.1,131.4)`. The x coordinate is inside the target width, while y is **7.4 pixels below** the annotated bottom edge. After subtracting four same-image diverse instructions, the peak patch is inside the oracle and **17.15%** of positive residual mass falls there (**278.6× lift**), with a **0.951** leave-one-control-out floor. This is consistent with successful target-specific routing followed by an imprecise action readout.
- **Free multi-turn hotel success:** with our hotel prompt and the exact HoloDesktop runtime 0.1.10 tool set, Holo freely emits three scrolls and then `click_desktop(x=650, y=450)`. The projected pixel click `(665,324)` lands inside the button for **Lumen Harbor Rooms at £122**. Final y-token value-norm rollout allocates **38.6% / 29.2% / 32.2%** across the three retained frames. All three frames contribute; this run does not establish a recency preference or a causal memory mechanism.
- **Resolution-attribution pilot:** six native-success cases span Photoshop, PowerPoint, and VS Code with icon and text targets represented. Each runs at 100%, 75%, 50%, and 25% linear resolution. Every condition uses value-norm rollout minus the mean of four same-image alternate-instruction maps, for 120 deterministic requests. Median target-specific mass falls from **5.22% to 2.13% to 0.67% to 0.04%**, while median target lift falls from **241.5× to 113.5× to 45.9× to 10.4×**. Strict hits fall from **6/6 to 3/6 to 3/6 to 1/6**. One failed 50% VS Code click still has its top attribution patch inside the target, so weakened localization dominates this pilot without eliminating coordinate readout as a secondary failure mode.
- **Causal panel:** four equal-size tile pairs and two mirrored instructions create eight within-image tests. The primary metric is teacher-forced target-x log probability minus distractor-x log probability. Patching the clean Layer 15 coordinate residual into the corrupted Holo run restores a median **+1.01 nats**, versus **+0.10 nats** for Qwen. Holo restoration is positive in **7/8** prompts; the paired Holo-minus-Qwen median is **+0.72 nats**.
- **Zero-ablation:** set the same Layer 15 coordinate residual to zero in the clean run and measure the target-margin drop. The median drop is **4.51 nats for Holo** and **6.27 nats for Qwen**. Both checkpoints rely on this state, so the result supports increased Holo recoverability rather than a Holo-exclusive representation.
- **All-head sweep:** Holo head 1 restores **+0.54 nats**, while Qwen uses the same head at **+0.36 nats**. No single head explains the Layer 15 restoration effect.
- **Scope:** attention maps remain descriptive routing diagnostics. The causal panel uses one PowerPoint image with eight mirrored prompt/swap conditions; it does not estimate independent-image or benchmark-level generalization.

## Five-minute Q&A crib sheet

### Is attention a faithful explanation?

Not by itself. The maps identify candidate routes and sites. The causal panel separately changes internal states and measures coordinate-margin effects. Its Layer 15 residual result is distributed across heads, which is exactly why the deck avoids treating a salient head as a mechanism.

### What does icon versus text mean, and is size controlled?

It describes the annotated click target. ScreenSpot-Pro defines an icon target as one with no text hint. If a text label appears in the target, the benchmark calls it text even when an icon is also present. The released annotations contain some apparent edge cases or label noise, so the appendix uses unambiguous examples.

The raw accuracy table does not control target geometry. That matters because strict accuracy is point-in-box and larger boxes have a wider hit tolerance. In our audit, the median normalized text box is 4.77 times the icon median. The dedicated histogram shows the full observed size distributions instead of reporting a model-adjusted gap. It establishes substantial imbalance in click-box geometry, but it does not determine how much of the performance difference size causes.

### What exactly is the causal metric?

We teacher-force two equal-length native coordinate strings and score only the x-coordinate digit tokens because the paired y coordinate is held fixed in the horizontal tile-swap panel. The margin is target-x log probability minus distractor-x log probability. Syntax tokens do not enter the score.

A nat is one natural-log unit. Exponentiating the margin gives the target-to-distractor sequence-probability ratio. This is a sequence log-likelihood ratio, not an attention score.

### Is this activation patching or pixel swapping?

Both operations appear, at different stages. The pixel swap creates the minimally corrupted input by cropping both equal-size tiles from the original image and pasting them into each other’s boxes without resize or resampling. Activation patching then replaces one internal tensor in the corrupted forward pass with the corresponding tensor captured from the clean pass. The ablation control removes the same component from the clean pass.

### Why use both patching and ablation?

Patching asks whether the clean Layer 15 coordinate state restores the corrupted decision. Zero-ablation sets the same state to zero in the clean run and asks how much target margin disappears. Both checkpoints show an ablation cost, while Holo shows much larger clean-state restoration.

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

### Why did full-resolution multi-frame tracing take so long?

The model weights remain streamed to Metal at about 9.64 GiB. The expensive part is the first full-attention prompt matrix over four high-resolution frames, whose memory grows quadratically with prompt length and causes swap pressure. This motivates frame retrieval or patch compression for research instrumentation.

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
- Causal method and results: `docs/causal-intervention.md`
- Causal measurements: `artifacts/causal-intervention/powerpoint_windows_59_swap/results.json`
- Eight-prompt Qwen/Holo causal panel: `data/remote-results/qwen-holo-action-panel-v1-20260922/`
- Layer 15 ablation and all-head sweep: `data/remote-results/qwen-holo-action-panel-v1-20260922-phase-b-layer15/`
- Official-tool native free hotel rollout and attribution packages: `data/remote-results/hotel-official-tools-native-20260922-rerun/`
- Official-tool native free hotel slide assets: `artifacts/screenspot-presentation/hotel-freegen-official-tools-v1/`
- Static measurements: `data/attributions/screenspot-powerpoint_windows_{59,48}/analysis.json`
- Tracked hotel viewer and measurement: `artifacts/screenspot-presentation/live/hotel-cheapest-multiframe/`
- Hotel probe manifest: `data/trajectory-prompt-cases/hotel-cheapest-final-640/case.json` (directory name is historical; manifest records full-resolution `1280x800` frames and `frame_scale=1.0`)
- Resolution-ablation cohort and analysis: `benchmarks/resolution_ablation/screenspot_success_retention_v1.json` and `artifacts/screenspot-presentation/screenspot-resolution-ablation-v1.json`
- Resolution-saliency pilot: `artifacts/screenspot-presentation/resolution-saliency-pilot-v1/summary.json`
- Revised deck: `artifacts/screenspot-presentation/holo-attribution-research-v43.pptx`
