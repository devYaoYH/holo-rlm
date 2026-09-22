# ScreenSpot-Pro resolution-success retention

This experiment quantifies how Holo 3.1 4B grounding changes when only screenshot resolution is reduced. It is a paired diagnostic over 36 items that were strictly correct in the 1,581-item native-resolution reference run. The cohort contains six size-spread targets from each application × UI-type stratum across Photoshop, PowerPoint, and VS Code.

Every item is rerun in one model process at 100%, 75%, 50%, and 25% of its original width and height. Lanczos downsampling preserves aspect ratio. The server remains at the checkpoint-native `image_max_pixels=16777216`, so it does not impose an additional hidden resize. The official ScreenSpot-Pro localization prompt, `VisualLocalizerOutput` schema, temperature zero, thinking-disabled decoding, and normalized 0–1000 coordinate contract are held fixed.

## Result

| Linear resolution | Retained source area | Strict hits | Success retention | Median click drift (0–1000) |
|---:|---:|---:|---:|---:|
| 100% | 100% | 36/36 | 100.0% | 0.0 |
| 75% | 56.25% | 25/36 | 69.4% | 2.9 |
| 50% | 25% | 22/36 | 61.1% | 6.7 |
| 25% | 6.25% | 4/36 | 11.1% | 84.8 |

All 144 outputs were valid JSON. The same-run native rerun remained correct on all 36 items, so the retention denominator is fully paired. Wilson 95% intervals for retention are **53.1–82.0%** at 75%, **44.9–75.2%** at 50%, and **4.4–25.3%** at 25% linear resolution. The 25%-linear condition loses 88.9 percentage points of strict click success and produces a large median coordinate shift. This supports a strong resolution-sensitivity claim for UI localization on this selected-success cohort.

The degradation is concentrated in small/icon targets. At 50% linear resolution, text-target retention is **15/18 (83.3%)**, while icon-target retention is **7/18 (38.9%)**. The median normalized target area among retained items is about **4.1×** the median among failures. These subgroup estimates are small and descriptive, but they align with the independent target-geometry analysis.

This is not an unconditional ScreenSpot-Pro accuracy estimate: the cohort was selected for native success and covers three applications. A full causal accuracy curve would rerun an application-balanced random sample, including original failures, at every resolution. The item-level pattern is also not perfectly monotonic because strict point-in-box scoring and tiny targets make boundary flips possible; aggregate retention is the intended statistic.

Tracked inputs and analysis:

- frozen cohort: `benchmarks/resolution_ablation/screenspot_success_retention_v1.json`
- runner: `src/demo/screenspot_resolution.py`
- analysis: `scripts/analyze_screenspot_resolution_ablation.py`
- compact result: `artifacts/screenspot-presentation/screenspot-resolution-ablation-v1.json`

The full response package is retained locally under `data/remote-results/screenspot-resolution-success-retention-v1/` and is intentionally ignored by Git.
