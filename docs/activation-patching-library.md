# Activation patching library

`instrumented_holo.activation_patching` provides the tensor-level mechanics and `instrumented_holo.interventions` provides a serializable, pyvene-inspired public API. Together they turn the matched PowerPoint intervention into a reusable diagnostic for other screenshot or trajectory pairs. They handle teacher-forced candidate scoring, multi-image prompt preparation, pixel-region to visual-token mapping, clean-to-corrupted activation replacement, and clean-run ablation.

## Core objects

- `CandidateSequence` contains a full native assistant completion and the exact character spans to score. `coordinate_tool_candidate()` builds the common click-coordinate candidate.
- `ImageRegion(image_index, box, limit)` maps a pixel box on any image in message order to visual-token positions.
- `ActivationPatchingRunner.prepare()` accepts an OpenAI-style message history, tools, two candidates, and named regions. Histories may contain one screenshot or several retained trajectory frames.
- `ActivationPatchingRunner.score()` returns each candidate's teacher-forced log likelihood and the candidate-0 minus candidate-1 margin in nats.
- `residual_hook()`, `attention_hook()`, and `prediction_hook()` construct the patch or ablation hooks used by the current case.
- `RepresentationConfig(layer, component, unit, ...)` declares where to intervene. `InterventionConfig` adds the interchange and paired ablation operations.
- `IntervenableHolo.compare(clean, corrupted)` captures each clean source once, returns original clean/corrupted scores, and runs each patch/ablation pair.

## Minimal use

```python
from instrumented_holo.activation_patching import (
    ActivationPatchingRunner,
    ImageRegion,
    coordinate_tool_candidate,
    require_patch_alignment,
)

runner = ActivationPatchingRunner(engine, reduction="sum")
candidates = (
    coordinate_tool_candidate("target", (482, 351)),
    coordinate_tool_candidate("distractor", (406, 351)),
)
regions = {
    "target": ImageRegion(image_index=0, box=(1289, 563, 1489, 702), limit=4),
    "distractor": ImageRegion(image_index=0, box=(1069, 563, 1269, 702), limit=4),
}

clean = runner.prepare(
    name="clean",
    messages=clean_messages,
    tools=tools,
    candidates=candidates,
    regions=regions,
)
corrupted = runner.prepare(
    name="corrupted",
    messages=corrupted_messages,
    tools=tools,
    candidates=candidates,
    regions=regions,
)
require_patch_alignment(clean, corrupted)

from instrumented_holo.interventions import (
    IntervenableHolo,
    InterventionConfig,
    RepresentationConfig,
)

config = (
    InterventionConfig(
        name="target visual residuals",
        representation=RepresentationConfig(
            layer=19,
            component="residual_output",
            unit="image_region",
            region="target",
        ),
    ),
    InterventionConfig(
        name="layer 19 head 10",
        representation=RepresentationConfig(
            layer=19,
            component="attention_head_output",
            unit="scored_token_predictions",
            head=10,
        ),
        ablation="zero",
    ),
)
output = IntervenableHolo(runner, config, regions=regions).compare(clean, corrupted)
```

This deliberately borrows pyvene's representation/configuration separation rather than depending on pyvene's model registry. Holo's Qwen3.5 hybrid VLM path needs processor-derived visual grids, semantic pixel-region units, multimodal alignment checks, and a domain-specific teacher-forced margin. Manifests use the same configuration objects, so a remote CLI run and an in-process experiment share one API contract. The initial `kind`/`scope` manifest spelling remains accepted as a compatibility layer.

For a trajectory pair, place the retained screenshots in the message history in the same order for both conditions. Set `image_index` to the historical or current frame whose patches you want to intervene on. The model processor derives a separate visual grid and contiguous image-token span for every frame.

## Required invariants

Activation replacement assumes that source and destination tensors refer to aligned token positions. `require_patch_alignment()` checks:

- identical clean and corrupted token-sequence shapes;
- identical coordinate-prediction positions;
- the same number of visual tokens for each image;
- identical per-image visual grids.

Keep message text, tool schema, image count, image order, and candidate completions fixed across the pair. Change only the intervention variable, such as selected pixels, one retained frame, or one action-history element whose token length remains matched. If the histories tokenize differently, define an explicit alignment method instead of directly copying tensor positions.

## Score semantics

With `reduction="sum"`, both candidates must have the same number of scored tokens. The margin is:

```text
sum ln p(candidate 0 scored tokens) - sum ln p(candidate 1 scored tokens)
```

One nat is one natural-log unit. `exp(margin)` is the candidate-0 to candidate-1 sequence-probability ratio. Use `reduction="mean"` only when unequal candidate lengths are scientifically unavoidable, and report that the estimand changed to mean token log probability.

The runner scores only the declared spans, but each token remains conditioned on the candidate's own earlier teacher-forced prefix. For the current coordinate-pair metric this includes downstream consequences of the x tokens when scoring y. If the research question is strictly x-choice discrimination, declare spans that include x only.

## Corruption construction

`swap_equal_tiles()` validates that both boxes are in bounds, equal in size, and non-overlapping. It crops both boxes from the original image before either paste. This avoids cascading edits and performs no resize or resampling. The case CLI also accepts `--corrupted-image` for a caller-supplied matched image pair.

## Current limits

The library provides the intervention mechanics, not an automatic causal design. A new use still needs pre-registered candidate coordinates, a corruption that preserves nuisance variables, component choices made independently of the result, and repeated cases for uncertainty estimates. The current MPS result is one deterministic teacher-forced pass per intervention and does not estimate numerical or sampling variance.
