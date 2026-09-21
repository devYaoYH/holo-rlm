# Holo 3.1 attention attribution: presenter notes

## Ten-minute run of show

| Time | Slide | Talk track |
|---|---|---|
| 0:00-0:35 | 1 | Ask what is specific to the current instruction after removing generic visual awareness. |
| 0:35-1:25 | 2 | Separate static ScreenSpot grounding from replayable multi-turn visual memory. |
| 1:25-2:35 | 3 | Explain value-norm, eight-layer residual rollout, parameter-token aggregation, and both baselines. |
| 2:35-3:45 | 4 | Establish the headline metric: same-image instruction differential. |
| 3:45-4:45 | 5 | Static miss: coarse task region is salient, but the tiny affordance loses. |
| 4:45-6:05 | 6 | Multi-turn probe: target evidence is measurable in an earlier frame and stronger in the current frame, yet the click misses the button. |
| 6:05-7:20 | 7 | Layer patterns are candidate features; explain why the current trace inventory cannot validate correctness. |
| 7:20-8:35 | 8 | Propose calibrated heads, causal tests, visual-memory research, base-model comparison, and the SFT-to-RLVR loop. |
| 8:35-10:00 | 9 + demo | Recap three defensible claims, then show the static hit, static miss, and four-frame hotel viewer. |

## Exact claims to make

- **ScreenSpot visual hit:** prompt-differential target lift is **16.08x**, versus **3.85x** for raw rollout. Its peak falls inside the annotated target. Leave-one-control-out cosine is **0.925 mean / 0.828 minimum**.
- **ScreenSpot miss:** the prompt-differential peak is outside the target and **9.9% of the frame diagonal** from its center. The repaired click is `(331, 270)`, on the slide thumbnail rather than the New Slide button. Stability is **0.970 mean / 0.953 minimum**.
- **Multi-turn hotel probe:** after replaying the same four screenshots and three-scroll history, the cheapest hotel's button has **2.78x** differential lift when it first appears in frame 2 and **8.59x** in the final frame. The probe clicks `(960,120)` on the correct hotel card, outside the button. Three controls are included; one is excluded for missing a `y` span. Stability is **0.898 mean / 0.769 minimum**.
- **Layer result:** layer 19 mean-head lift is **38.99x** for the static visual hit, **10.47x** for the hotel miss, and **2.88x** for the static miss. This is an exploratory three-case pattern.
- **Evidence audit:** the hotel corpus has **39 outcome-labeled trajectories**, including **20 successes**. Only **11 trajectories have activation trace IDs**, and all 11 are failures. We therefore do not have a balanced activation set for testing an early correctness feature.
- **Scope:** the two ScreenSpot cases use different images and prompts. They are not a matched success/failure causal pair. The hotel target is a matched re-probe over captured history, not the exact original final model call.

## Ninety-second live demo

1. Open the [ScreenSpot hit viewer](../../data/attributions/screenspot-powerpoint_windows_59/viewer.html).
2. Start on **Raw value-norm rollout**, then select **Target minus diverse-instruction baseline**. Show the peak entering the template and the 16.08x lift.
3. Sort heads by **Prompt difference** and point out layer 19 / head 10 as the strongest direct value-norm head in this case.
4. Open the [ScreenSpot miss viewer](../../data/attributions/screenspot-powerpoint_windows_48/viewer.html). Show the task-region signal and the click on the slide thumbnail.
5. Open the [multi-frame hotel viewer](live/hotel-cheapest-multiframe/viewer.html).
6. Select **Target minus diverse-instruction baseline**. Step from frame 2 to frame 3. The green box marks the target button; the high-contrast white ring/cross on frame 3 marks the failed click.
7. End on the layer/head table as a source of intervention hypotheses, not a causal conclusion.

## Five-minute Q&A crib sheet

### Is attention a faithful explanation?

No. It is a routing diagnostic for one observed forward pass. Faithfulness requires interventions: patch masking or swapping, activation patching, and targeted layer/head ablation followed by action measurement.

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

### Why did full-resolution multi-frame tracing take so long?

The model weights remain streamed to Metal at about 9.64 GiB. The expensive part is the first full-attention prompt matrix over four high-resolution frames, whose memory grows quadratically with prompt length and causes swap pressure. This motivates frame retrieval or patch compression for research instrumentation.

### Can layer statistics become confidence?

Potentially, but not from three prompt-baselined cases, and not from the current trajectory archive. Although 20 of 39 hotel trajectories succeeded, none of the 11 instrumented trajectories succeeded. First capture a balanced set of successful and failed activation traces, then freeze the backbone, train a small outcome or ambiguity readout, and evaluate held-out Brier score, log loss, expected calibration error, and selective risk. Safety refusal and clarification thresholds should be chosen from calibrated risk, not raw saliency.

### Is base Qwen versus Holo interesting?

Yes if action tokens are teacher-forced. The base model may not emit the tool syntax reliably, so free-running heatmaps confound format and vision. Feed the same image, prompt, and action-token sequence to base Qwen and Holo, then compare where instruction-specific routing changes by layer and head.

### What should we train next?

Keep a frozen diverse test split. Generate and inspect oracle SFT trajectories, measure mid-training lift, then run GRPO with LoRA and a KL penalty using verifiable environment rewards. Use attribution for audit and diagnosis, not as the reward target.

## Sources and reproducibility

- ScreenSpot-Pro paper: <https://arxiv.org/abs/2504.07981>
- Official repository: <https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding>
- Frozen case/control manifest: `benchmarks/screenspot_presentation_cases.json`
- Method: `docs/attention-attribution.md`
- Case study: `docs/screenspot-case-study.md`
- Static measurements: `data/attributions/screenspot-powerpoint_windows_{59,48}/analysis.json`
- Tracked hotel viewer and measurement: `artifacts/screenspot-presentation/live/hotel-cheapest-multiframe/`
- Hotel probe manifest: `data/trajectory-prompt-cases/hotel-cheapest-final-640/case.json` (directory name is historical; manifest records full-resolution `1280x800` frames and `frame_scale=1.0`)
- Revised deck: `artifacts/screenspot-presentation/holo-attribution-research-v5.pptx`
