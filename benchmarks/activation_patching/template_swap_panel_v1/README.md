# Template-swap causal panel, v1

This frozen Phase-A panel uses four target/distractor equal-tile swaps within
the public `powerpoint_windows_59` screenshot.  Each manifest is constructed
with `demo.screenspot.build_screenspot_request`, i.e. the official H Company
element-localization JSON request and `VisualLocalizerOutput` schema.

Run `uv run python scripts/build_causal_template_panel.py` after changing the
generator.  The phase-A analysis sweeps target and matched distractor visual
residuals over full-attention layers 3, 7, 11, 15, 19, 23, 27, and 31, at the
native 2880 x 1800 image resolution (the configured ceiling is 16,777,216
pixels).  It is a within-image spatial replication panel, not a benchmark
accuracy estimate or independent-image causal result.

Phase B is allowed only for layers meeting the manifest's predeclared rule:
positive target restoration and clean ablation drop in at least three of the
four cases, with median target restoration above the matched distractor-region
median.  For every selected layer, it must sweep *all* attention heads across
all four cases.

The associated command can generate a separate, exhaustive 16-head manifest
set for a Phase-A-selected layer, without changing prompts, images, or cases:

```bash
uv run python scripts/build_causal_template_panel.py \
  --output benchmarks/activation_patching/template_swap_panel_v1/layer15_head_sweep \
  --head-sweep-layer 15
```

The sign-only gate is intentionally permissive; a selected layer still needs a
material, paired, and replicated head effect before it can support a localized
circuit claim.
