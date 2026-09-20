# ScreenSpot-Pro attention attribution — presenter notes

## Ten-minute run of show

| Time | Slide | Talk track |
|---|---|---|
| 0:00–0:45 | 1 | The question is not “where is attention high?” but “what is specific to this instruction after removing generic image saliency?” |
| 0:45–1:45 | 2 | Introduce ScreenSpot-Pro, the controlled PowerPoint pair, and the separation of format validity, repaired visual grounding, and official end-to-end correctness. |
| 1:45–3:10 | 3 | Explain value-norm correction, residual cross-layer rollout, x/y token aggregation, and the two baselines. |
| 3:10–4:25 | 4 | Success: raw rollout peaks on a generic corner; the prompt differential peaks inside the selected template. |
| 4:25–5:35 | 5 | Miss: the correct semantic neighborhood is selected, but the slide thumbnail wins over the tiny New Slide affordance. |
| 5:35–6:45 | 6 | Baselines answer different questions. Target lift must be read with peak position and the actual overlay. |
| 6:45–8:05 | 7 | Success-specific specialization peaks at layer 19; head 10 is the strongest direct value-norm head in this one case. |
| 8:05–9:00 | 8 | State the three conclusions and the limits. |
| 9:00–10:00 | live viewer | Raw → target-minus-prompt baseline → head sort → failure viewer. |

## Exact claims to make

- **Success case:** prompt-differential target lift is **16.08×**, versus **3.85×** for raw rollout; its peak falls inside the annotated target. Leave-one-control-out cosine is **0.925 mean / 0.828 minimum**.
- **Failure case:** the prompt-differential peak is outside the target and **9.9% of the frame diagonal** from its center. The repaired click is `(331, 270)`, on the slide thumbnail rather than the New Slide toolbar button. Baseline stability is **0.970 mean / 0.953 minimum**.
- **Layer result:** the success reaches **38.99× mean head target lift at layer 19**. Layer 19 / head 10 reaches **98.05×** on direct value-norm prompt-differential attribution.
- **Scope:** these are two case studies. The layer/head pattern is a hypothesis to replicate and intervene on, not a population-level conclusion.

## Ninety-second live demo

1. Open the [success viewer](../../data/attributions/screenspot-powerpoint_windows_59/viewer.html).
2. Start on **Raw value-norm rollout**. Point out the generic top-left peak.
3. Select **Target minus diverse-instruction baseline**. Point to the red patch inside the green target box and the displayed 16.08× lift.
4. Under **Per-head ranking**, keep **Prompt difference** selected and show layer 19 / head 10 at the top.
5. Open the [failure viewer](../../data/attributions/screenspot-powerpoint_windows_48/viewer.html).
6. Select the same prompt-differential mode. Show the red top-left neighborhood, the green toolbar target, and the white cross on the slide thumbnail.

## Five-minute Q&A crib sheet

### “Is attention a faithful explanation?”

No. This is a routing diagnostic. It identifies where information can flow under the observed forward pass. Faithfulness requires interventions: mask or swap candidate patches, perturb the instruction, or ablate the implicated layer/head and measure click changes.

### “Why multiply by the value-vector norm?”

An attention probability can be large while the value carried along that edge is nearly zero. We replace each row with `normalize(A ⊙ ||V||₂)` before rollout, so strong-but-empty paths do not dominate the map. This still does not recover value direction or downstream nonlinear effects.

### “Why does rollout cover only eight layers?”

Holo 3.1 4B is hybrid. Its conventional full-attention blocks expose square causal attention matrices; its interleaved linear-attention blocks do not expose an equivalent matrix that can be multiplied without inventing an approximation. The viewer and docs state this boundary explicitly.

### “How were the diverse prompt controls chosen?”

Four visible click instructions target spatially distinct regions of the exact same screenshot. Every request shares image bytes, model, processor, tool schema, and decoding settings. Each coordinate-token map is L1-normalized before averaging. Leave-one-control-out cosine quantifies whether one control dominates. A larger study should use a stratified control bank and report confidence intervals.

### “Is the success actually a benchmark success?”

Not end to end. Holo produced the visually correct coordinates but inserted `>` inside integer fields. The harness records `format_valid=false`, `grounding_correct=true` after a narrow first-integer repair, and `correct=false` for the official result. The repair diagnoses vision; it never upgrades the score.

### “Why is lift high in the failure?”

The New Slide target is only about 0.074% of the frame and lies inside a generally salient top-left region. Dividing even modest target mass by such a tiny area yields high lift. That is why the presentation pairs lift with peak-inside, peak distance, and the overlay.

### “What is interesting about layer 19 / head 10?”

In the success case it is the strongest direct value-norm head after prompt subtraction, and the layer-mean curve peaks at 19. It is not yet a universal “GUI grounding head.” The next step is replication across the frozen set followed by causal ablation or activation patching.

### “What should we train next?”

Separate failures into formatting, coarse semantic localization, and fine target binding. SFT can fix tool syntax and reinforce exact point targets; RLVR can optimize point-in-box success. Keep the frozen ScreenSpot-Pro subset untouched and use the synthetic benchmark for cheap engineering/debugging.

## Sources and reproducibility

- ScreenSpot-Pro paper: https://arxiv.org/abs/2504.07981
- Official repository: https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding
- Official dataset: https://huggingface.co/datasets/likaixin/ScreenSpot-Pro
- Frozen case/control manifest: `benchmarks/screenspot_presentation_cases.json`
- Method: `docs/attention-attribution.md`
- Case study: `docs/screenspot-case-study.md`
- Local measurements: `data/attributions/screenspot-powerpoint_windows_{59,48}/analysis.json`
