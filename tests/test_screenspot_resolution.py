from __future__ import annotations

from demo.screenspot_resolution import (
    build_native_success_cohort,
    point_to_bbox_distance_normalized,
    scaled_size,
    summarize_resolution_results,
)


def test_build_native_success_cohort_balances_and_spreads_target_sizes() -> None:
    annotations = []
    items = []
    for application in ("photoshop", "powerpoint", "vscode"):
        for ui_type in ("icon", "text"):
            for index in range(7):
                sample_id = f"{application}-{ui_type}-{index}"
                annotations.append(
                    {
                        "id": sample_id,
                        "instruction": "click",
                        "img_filename": f"{sample_id}.png",
                        "bbox": [0, 0, index + 1, 10],
                        "img_size": [100, 100],
                        "application": application,
                        "platform": "test",
                        "ui_type": ui_type,
                    }
                )
                items.append({"sample_id": sample_id, "strict_correct": index != 3})
    cohort = build_native_success_cohort(
        {"items": items}, annotations, per_stratum=3, applications=("photoshop", "powerpoint", "vscode")
    )
    assert len(cohort) == 18
    assert all(case["selection_native_strict_correct"] for case in cohort)
    counts = {(case["application"], case["ui_type"]) for case in cohort}
    assert len(counts) == 6


def test_resolution_geometry_and_paired_summary() -> None:
    assert scaled_size((2880, 1800), 0.5) == (1440, 900)
    assert point_to_bbox_distance_normalized({"x": 50, "y": 50}, (40, 40, 60, 60), (100, 100)) == 0
    assert point_to_bbox_distance_normalized({"x": 70, "y": 50}, (40, 40, 60, 60), (100, 100)) == 100
    results = [
        {"sample_id": "a", "linear_scale": 1.0, "format_valid": True, "strict_correct": True, "point_to_box_distance_normalized": 0, "drift_from_native_normalized": 0},
        {"sample_id": "b", "linear_scale": 1.0, "format_valid": True, "strict_correct": False, "point_to_box_distance_normalized": 20, "drift_from_native_normalized": 0},
        {"sample_id": "a", "linear_scale": 0.5, "format_valid": True, "strict_correct": False, "point_to_box_distance_normalized": 10, "drift_from_native_normalized": 15},
        {"sample_id": "b", "linear_scale": 0.5, "format_valid": True, "strict_correct": True, "point_to_box_distance_normalized": 0, "drift_from_native_normalized": 12},
    ]
    summary = summarize_resolution_results(results)
    assert summary["native_rerun_correct_count"] == 1
    half = next(row for row in summary["scales"] if row["linear_scale"] == 0.5)
    assert half["selection_retention"] == 0.5
    assert half["paired_native_rerun_retention"] == 0.0
