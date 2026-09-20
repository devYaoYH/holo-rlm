from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from attribution import (
    AttributionError,
    build_prompt_contrast,
    load_attribution,
    spatial_metrics,
    write_prompt_contrast_viewer,
)
from attribution.contrast import parameter_steps


def _coordinate_trace(tmp_path: Path, name: str, allocation: tuple[float, float], *, color: str = "white") -> Path:
    trace = tmp_path / name
    trace.mkdir()
    Image.new("RGB", (100, 50), color).save(trace / "model-input-000.png")
    np.save(trace / "input_ids.npy", np.asarray([[1, 9, 9, 2]], dtype=np.int64))
    tokens = [
        "<parameter=x>\n",
        "5",
        "0",
        "0\n</parameter>\n",
        "<parameter=y>\n",
        "4",
        "0",
        "0\n</parameter>",
    ]
    np.save(trace / "generated_ids.npy", np.asarray([range(40, 48)], dtype=np.int64))
    (trace / "generated_tokens.json").write_text(json.dumps({"token_ids": list(range(40, 48)), "tokens": tokens}))
    (trace / "positions.json").write_text(json.dumps({"image_token_id": 9, "prompt_token_count": 4}))
    (trace / "processor.json").write_text(
        json.dumps({"files": {"preprocessor_config.json": {"merge_size": 2}}})
    )
    (trace / "vision_inputs.json").write_text(
        json.dumps({"image_grid_thw": {"shape": [1, 3], "values": [[1, 2, 4]]}})
    )
    (trace / "model.json").write_text(json.dumps({"attention_layer_indices": [3], "model": "fixture"}))
    arrays = {}
    for step in range(8):
        row = np.zeros((1, 4 + step), dtype=np.float32)
        row[0, 1:3] = (0.5, 0.5) if step == 0 else allocation
        arrays[f"step_{step:03d}_layer_000"] = row
    np.savez_compressed(trace / "attention_last_query_rows.npz", **arrays)
    np.savez_compressed(trace / "value_norms.npz", layer_003=np.ones((1, 11), dtype=np.float32))
    return trace


def test_prompt_ensemble_difference_is_normalized_and_signed(tmp_path: Path) -> None:
    target = load_attribution(_coordinate_trace(tmp_path, "target", (0.9, 0.1)))
    control_a = load_attribution(_coordinate_trace(tmp_path, "control-a", (0.1, 0.9)))
    control_b = load_attribution(_coordinate_trace(tmp_path, "control-b", (0.5, 0.5)))

    assert parameter_steps(target, ("x", "y")) == (1, 2, 3, 5, 6, 7)
    contrast = build_prompt_contrast(
        target,
        (control_a, control_b),
        method="value_norm",
        control_instructions=("control A", "control B"),
    )
    np.testing.assert_allclose(contrast.raw_maps[0], [[0.9, 0.1]])
    np.testing.assert_allclose(contrast.prompt_baseline_maps[0], [[0.3, 0.7]])
    np.testing.assert_allclose(contrast.prompt_difference_maps[0], [[0.6, -0.6]])
    np.testing.assert_allclose(contrast.causal_maps[0], [[0.4, -0.4]])
    assert contrast.stability["minimum_leave_one_out_cosine"] is not None

    metrics = spatial_metrics(contrast.prompt_difference_maps[0], (0, 0, 50, 50), (100, 50))
    assert metrics["target_mass"] == pytest.approx(1.0)
    assert metrics["target_lift"] == pytest.approx(2.0)
    assert metrics["peak_inside_target"] is True

    output = tmp_path / "viewer"
    result = write_prompt_contrast_viewer(
        contrast,
        output,
        sample_id="fixture-1",
        target_instruction="Click target",
        bbox=(0, 0, 50, 50),
        predicted_click=(25, 25),
        correct=True,
    )
    assert Path(result["viewer"]).is_file()
    assert Path(result["previews"]["target-minus-prompt-baseline"]).is_file()
    assert "__PROMPT_CONTRAST_PAYLOAD__" not in (output / "viewer.html").read_text()
    analysis = json.loads((output / "analysis.json").read_text())
    assert analysis["controls"] == ["control A", "control B"]
    assert analysis["layer_head_statistics"]["heads"][0]["prompt_difference"]["target_lift"] == pytest.approx(2.0)


def test_prompt_ensemble_rejects_a_different_image(tmp_path: Path) -> None:
    target = load_attribution(_coordinate_trace(tmp_path, "target", (0.9, 0.1)))
    mismatch = load_attribution(_coordinate_trace(tmp_path, "mismatch", (0.1, 0.9), color="black"))
    with pytest.raises(AttributionError, match="exact same input image"):
        build_prompt_contrast(target, (mismatch,), method="value_norm")
