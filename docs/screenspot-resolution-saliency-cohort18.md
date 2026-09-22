# Resolution-saliency cohort

18 native-success cases, 72 conditions. Attribution is value-norm rollout minus the mean of four same-image instruction controls.

| Linear resolution | Strict hits | Median target mass | Median target lift | Median best-patch rank | Lost clicks with peak inside |
|---:|---:|---:|---:|---:|---:|
| 100% | 18/18 | 5.22% | 168.7x | 1.0 | 0/0 |
| 75% | 12/18 | 2.18% | 79.2x | 1.0 | 1/6 |
| 50% | 10/18 | 1.16% | 29.2x | 1.0 | 3/8 |
| 25% | 3/18 | 0.21% | 5.4x | 6.5 | 3/15 |

## Key comparisons

At quarter resolution, the median within-case target-mass ratio is 6.23% of native (bootstrap 95% interval 0.18% to 14.47%).

Across 29 non-native missed clicks, 7 (24.1%) still rank an oracle-overlapping patch first after diverse-instruction subtraction. At half resolution this occurs in 3/8 misses; at quarter resolution it occurs in 3/15.

Most resolution-induced misses coincide with weaker target localization, while a repeatable minority retain a top-ranked target patch and fail at coordinate readout.

The cohort is selected on native success, so these are paired resolution-sensitivity diagnostics rather than unconditional benchmark estimates.
