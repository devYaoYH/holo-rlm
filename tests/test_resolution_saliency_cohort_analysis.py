from __future__ import annotations

import importlib.util
import math
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_resolution_saliency_cohort.py"
SPEC = importlib.util.spec_from_file_location("resolution_saliency_analysis", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = MODULE.wilson_interval(3, 6)
    assert lower < 0.5 < upper


def test_summarize_group_counts_click_map_disagreement() -> None:
    rows = [
        {
            "strict_correct": False,
            "metrics": {
                "prompt_difference": {
                    "target_mass": 0.2,
                    "target_lift": 20.0,
                    "best_target_patch_rank": 1,
                    "peak_distance_diagonal": 0.01,
                    "peak_inside_target": True,
                }
            },
        },
        {
            "strict_correct": True,
            "metrics": {
                "prompt_difference": {
                    "target_mass": 0.1,
                    "target_lift": 10.0,
                    "best_target_patch_rank": 3,
                    "peak_distance_diagonal": 0.1,
                    "peak_inside_target": False,
                }
            },
        },
    ]
    summary = MODULE.summarize_group(rows, resamples=100, rng=MODULE.random.Random(7))
    assert summary["strict_correct_count"] == 1
    assert summary["peak_inside_count"] == 1
    assert summary["rank_one_count"] == 1
    assert math.isclose(summary["median_target_mass"], 0.15)
