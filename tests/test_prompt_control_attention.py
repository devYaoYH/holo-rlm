from __future__ import annotations

import numpy as np

from scripts.analyze_prompt_control_attention import _role_summary


def _case(case_id: str, values: list[float]) -> dict:
    return {
        "id": case_id,
        "maps": {
            "value_norm_attention": [values],
        },
    }


def test_prompt_control_summary_uses_independently_normalized_maps() -> None:
    cases = {
        "target": _case("target", [0.1, 0.7, 0.2]),
        "control_a": _case("control_a", [0.4, 0.3, 0.3]),
        "control_b": _case("control_b", [0.5, 0.2, 0.3]),
    }

    summary, difference = _role_summary(
        cases,
        "target",
        ["control_a", "control_b"],
        "value_norm_attention",
        np.asarray([1]),
    )

    assert np.allclose(difference, [-0.35, 0.45, -0.1])
    assert np.isclose(summary["control_mean_target_region_mass"], 0.25)
    assert np.isclose(summary["prompt_difference_target_region_mass"], 0.45)
    assert summary["minimum_leave_one_out_cosine"] > 0.9
