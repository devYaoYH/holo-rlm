# Balanced ScreenSpot-Pro Layer 19 pilot

This experiment re-tests the earlier single-case Layer 19 observation on a frozen, balanced set of four official ScreenSpot-Pro items. The cohort contains two benchmark successes and two failures, paired by application, platform, and UI type. Every target is contrasted with four visible non-target instructions over the exact same screenshot.

## Frozen protocol

- Model: `Hcompany/Holo-3.1-4B`, revision `624347ebe6ab4df6f9701bcfc3c1bdfcfbc20b56`
- Repository commit used for capture: `14ec1719b7958b044522886df1fdd8de363c055b`
- Benchmark protocol: `hcompany_element_localization_v1`
- Instrumentation: eager attention, bfloat16, 2,097,152-pixel cap, 16 generated steps
- Cohort: 4 items, balanced 2 successes and 2 failures
- Controls: 4 same-image visible-target instructions per item
- Captures: 20 activation traces total

The frozen item and control manifest is [screenspot_balanced_layer19_v1.json](../benchmarks/attention_attribution/screenspot_balanced_layer19_v1.json).

## Metric

The reported value is **same-image prompt-differential target lift**. For each coordinate-token attribution map, the mean normalized map from the four control instructions is subtracted from the target-instruction map. Negative residuals are removed for this localization score. Target lift is then:

```text
positive residual mass inside the target box / target-box area fraction
```

A value of `1x` means the positive residual mass is spatially uniform. A value above `1x` means instruction-specific residual attribution is enriched inside the benchmark target box. This is an attention-routing diagnostic, not a probability, percentage, or causal effect. It is also distinct from the log-probability margin measured in **nats** in activation-patching experiments.

## Aggregate result

| Layer | Mean target lift | Median | Success mean | Failure mean | Aggregate rank |
|---:|---:|---:|---:|---:|---:|
| 19 | 158.4x | 129.3x | 293.7x | 23.2x | 1 |
| 15 | 151.4x | 89.6x | 284.7x | 18.1x | 2 |
| 23 | 125.0x | 58.4x | 242.1x | 7.9x | 3 |
| 27 | 94.6x | 43.1x | 183.7x | 5.6x | 4 |
| 11 | 85.1x | 67.4x | 154.5x | 15.6x | 5 |
| 31 | 67.0x | 29.6x | 131.2x | 2.8x | 6 |
| 7 | 6.8x | 4.8x | 11.4x | 2.1x | 7 |
| 3 | 3.3x | 2.7x | 4.0x | 2.7x | 8 |

Layer 19 ranks first in the frozen four-item aggregate, but only by `1.05x` relative to Layer 15. It is the top layer on two of four items and has a median per-item rank of 2. This supports treating Layer 19 as a high-value routing region, not as a uniquely isolated layer.

The strongest aggregate heads are a Layer 19 cluster:

| Layer/head | Mean target lift | Aggregate head rank |
|---|---:|---:|
| L19/H11 | 312.1x | 1 |
| L19/H14 | 293.8x | 2 |
| L19/H10 | 264.7x | 3 |
| L19/H2 | 245.6x | 4 |

L19/H10 remains a strong candidate, but it is third within Layer 19 by aggregate mean and is never the top Layer 19 head on an individual pilot item. The evidence therefore supports a Layer 19 head cluster rather than unique dominance by H10.

## Per-item Layer 19 result

| Item | Official outcome | L19 mean lift | L19 layer rank | L19/H10 lift | H10 rank within L19 |
|---|---|---:|---:|---:|---:|
| PowerPoint text | success | 373.0x | 3 | 661.1x | 3 |
| PowerPoint text | failure | 2.2x | 5 | 1.4x | 7 |
| Windows icon | success | 214.3x | 1 | 353.2x | 3 |
| Windows icon | failure | 44.3x | 1 | 43.2x | 6 |

Leave-one-control-out cosine stability ranges from `0.957` to `0.989`, indicating that the four-control baseline is internally stable for these items. The large success/failure gap is descriptive because this pilot is intentionally small and matched, not sampled for population inference.

## Reproduce the aggregate

After producing one `analysis.json` per item with `holo-capture screenspot-contrast`, run:

```bash
uv run python scripts/aggregate_screenspot_layer_heads.py \
  benchmarks/attention_attribution/screenspot_balanced_layer19_v1.json \
  --contrasts-root data/remote-results/attention-layer19-balanced-v1/contrasts \
  --output data/remote-results/attention-layer19-balanced-v1/aggregate
```

The command validates the cohort balance, outcomes, control prompts, and layer/head layout before writing `aggregate.json`, `items.csv`, `layers.csv`, and `heads.csv`.

The complete 20-trace archive is `data/remote-results/attention-layer19-balanced-v1.tar.gz`, with SHA-256 `b6ad83121a809cbd11349ef2f1e6fcc248165d36f07ddf6ba192bed051f2e462`.

## Limitations

This is a four-item pilot. The deterministic item bootstrap interval for Layer 19 is wide (`23.2x` to `293.7x`) and describes only this frozen cohort. The result motivates targeted intervention experiments at Layer 19, but it does not establish population-level generality or causal necessity.
