from __future__ import annotations

from pathlib import Path

from demo.screenspot import ScreenSpotSample
from demo.screenspot_resolution_attribution import scaled_sample, summarize_resolution_attribution


def test_scaled_sample_scales_image_and_box_geometry(tmp_path: Path) -> None:
    sample = ScreenSpotSample(
        id="fixture",
        instruction="click",
        image_path=tmp_path / "native.png",
        bbox=(100, 50, 300, 150),
        image_size=(1000, 500),
        application="fixture",
        platform="test",
        ui_type="text",
        annotation_path=tmp_path / "fixture.json",
    )
    scaled = scaled_sample(sample, tmp_path / "half.png", 0.5)
    assert scaled.image_size == (500, 250)
    assert scaled.bbox == (50, 25, 150, 75)


def test_summary_splits_retained_and_lost_clicks() -> None:
    def row(sample_id: str, correct: bool, mass: float, rank: int) -> dict:
        return {
            "sample_id": sample_id,
            "linear_scale": 0.5,
            "strict_correct": correct,
            "metrics": {
                "prompt_difference": {
                    "target_mass": mass,
                    "target_lift": mass * 10,
                    "peak_distance_diagonal": 0.1,
                    "best_target_patch_rank": rank,
                }
            },
        }

    summary = summarize_resolution_attribution([row("a", True, 0.2, 1), row("b", False, 0.05, 8)])
    groups = summary["scales"][0]["groups"]
    assert groups["retained"]["median_target_mass"] == 0.2
    assert groups["lost"]["median_best_target_patch_rank"] == 8
