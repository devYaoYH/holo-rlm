from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from attribution import (
    AttributionError,
    load_attribution,
    resolve_trace_path,
    write_attribution_viewer,
    write_trajectory_viewer,
)
from attribution.viewer import _encoded_method


def _trace(tmp_path: Path, *, exact_grid: bool = True) -> Path:
    trace = tmp_path / "traces" / "trace-test"
    trace.mkdir(parents=True)
    Image.new("RGB", (80, 40), "#314159").save(trace / "model-input-000.png")
    np.save(trace / "input_ids.npy", np.asarray([[1, 9, 9, 9, 9, 9, 9, 9, 9, 2, 3]], dtype=np.int64))
    np.save(trace / "generated_ids.npy", np.asarray([[41, 42]], dtype=np.int64))
    (trace / "positions.json").write_text(json.dumps({"image_token_id": 9, "prompt_token_count": 11}))
    (trace / "generated_tokens.json").write_text(
        json.dumps(
            {
                "token_ids": [41, 42],
                "tokens": ["<parameter=action>\n", "click\n</parameter>"],
            }
        )
    )
    (trace / "processor.json").write_text(
        json.dumps({"files": {"preprocessor_config.json": {"merge_size": 2}}})
    )
    vision = (
        {"image_grid_thw": {"shape": [1, 3], "values": [[1, 4, 8]]}, "pixel_values": {"shape": [32, 1536]}}
        if exact_grid
        else {"image_grid_thw": [1, 3], "pixel_values": [32, 1536]}
    )
    (trace / "vision_inputs.json").write_text(json.dumps(vision))
    (trace / "model.json").write_text(json.dumps({"attention_layer_indices": [3, 7]}))
    arrays = {}
    for step in range(2):
        for layer in range(2):
            values = np.zeros((2, 11 + step), dtype=np.float32)
            values[:, 1:9] = np.arange(1, 9, dtype=np.float32) * (step + 1) * (layer + 1) / 100
            arrays[f"step_{step:03d}_layer_{layer:03d}"] = values
    np.savez_compressed(trace / "attention_last_query_rows.npz", **arrays)
    return trace


def _multi_frame_trace(tmp_path: Path) -> Path:
    trace = tmp_path / "traces" / "trace-multi"
    trace.mkdir(parents=True)
    Image.new("RGB", (80, 40), "#314159").save(trace / "model-input-000.png")
    Image.new("RGB", (80, 40), "#926535").save(trace / "model-input-001.png")
    tokens = [1, *([9] * 8), 4, 5, *([9] * 8), 2, 3]
    np.save(trace / "input_ids.npy", np.asarray([tokens], dtype=np.int64))
    np.save(trace / "generated_ids.npy", np.asarray([[41]], dtype=np.int64))
    (trace / "positions.json").write_text(json.dumps({"image_token_id": 9, "prompt_token_count": len(tokens)}))
    (trace / "generated_tokens.json").write_text(json.dumps({"token_ids": [41], "tokens": ["click"]}))
    (trace / "processor.json").write_text(
        json.dumps({"files": {"preprocessor_config.json": {"merge_size": 2}}})
    )
    (trace / "vision_inputs.json").write_text(
        json.dumps(
            {
                "image_grid_thw": {"shape": [2, 3], "values": [[1, 4, 8], [1, 4, 8]]},
                "pixel_values": {"shape": [64, 1536]},
            }
        )
    )
    (trace / "model.json").write_text(json.dumps({"attention_layer_indices": [3]}))
    values = np.zeros((2, len(tokens)), dtype=np.float32)
    values[:, 1:9] = 0.1
    values[:, 11:19] = 0.3
    np.savez_compressed(trace / "attention_last_query_rows.npz", step_000_layer_000=values)
    return trace


def _hand_computed_rollout_trace(tmp_path: Path) -> Path:
    """Two-layer trace whose direct and rollout maps are calculable by hand."""

    trace = tmp_path / "traces" / "trace-hand-computed"
    trace.mkdir(parents=True)
    Image.new("RGB", (80, 40), "#314159").save(trace / "model-input-000.png")
    # Prompt positions: BOS, image patch 0, image patch 1, generation query.
    np.save(trace / "input_ids.npy", np.asarray([[1, 9, 9, 2]], dtype=np.int64))
    np.save(trace / "generated_ids.npy", np.asarray([[41, 42, 43]], dtype=np.int64))
    (trace / "positions.json").write_text(
        json.dumps({"image_token_id": 9, "prompt_token_count": 4})
    )
    (trace / "generated_tokens.json").write_text(
        json.dumps(
            {
                "token_ids": [41, 42, 43],
                "tokens": ["<parameter=action>\n", "sc", "roll\n</parameter>"],
            }
        )
    )
    (trace / "processor.json").write_text(
        json.dumps({"files": {"preprocessor_config.json": {"merge_size": 2}}})
    )
    (trace / "vision_inputs.json").write_text(
        json.dumps(
            {
                "image_grid_thw": {"shape": [1, 3], "values": [[1, 2, 4]]},
                "pixel_values": {"shape": [8, 1536]},
            }
        )
    )
    (trace / "model.json").write_text(json.dumps({"attention_layer_indices": [2, 5]}))

    lower_prompt = np.eye(4, dtype=np.float32)
    lower_prompt[3] = [0.0, 0.8, 0.2, 0.0]
    upper_prompt = np.eye(4, dtype=np.float32)
    upper_prompt[3] = [0.0, 0.25, 0.75, 0.0]
    np.savez_compressed(
        trace / "attention_prompt_mean.npz",
        layer_002=lower_prompt,
        layer_005=upper_prompt,
    )
    np.savez_compressed(
        trace / "attention_prompt_value_weighted_mean.npz",
        layer_002=lower_prompt,
        layer_005=upper_prompt,
    )

    generated_row = np.asarray([[0.0, 0.2, 0.6, 0.0, 0.2]], dtype=np.float32)
    final_row = np.asarray([[0.0, 0.4, 0.2, 0.0, 0.2, 0.2]], dtype=np.float32)
    np.savez_compressed(
        trace / "attention_last_query_rows.npz",
        step_000_layer_000=lower_prompt[3:4],
        step_000_layer_001=upper_prompt[3:4],
        step_001_layer_000=generated_row,
        step_001_layer_001=generated_row,
        step_002_layer_000=final_row,
        step_002_layer_001=final_row,
    )
    ones = np.ones((1, 6), dtype=np.float32)
    np.savez_compressed(trace / "value_norms.npz", layer_002=ones, layer_005=ones)
    return trace


def test_attention_rows_map_to_exact_patch_grid(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    attribution = load_attribution(trace)
    assert (attribution.layout.rows, attribution.layout.columns) == (2, 4)
    assert attribution.layout.source == "captured_image_grid_thw"
    assert attribution.layer_indices == (3, 7)
    assert attribution.maps.shape == (2, 2, 2, 2, 4)
    expected = np.arange(1, 9, dtype=np.float32).reshape(2, 4) * 0.015
    np.testing.assert_allclose(attribution.aggregate(0), expected)


def test_multiple_image_token_spans_map_to_separate_frames(tmp_path: Path) -> None:
    attribution = load_attribution(_multi_frame_trace(tmp_path))
    assert attribution.frame_count == 2
    assert [frame.image_path.name for frame in attribution.frames] == [
        "model-input-000.png",
        "model-input-001.png",
    ]
    np.testing.assert_allclose(attribution.aggregate(0, frame_index=0), 0.1)
    np.testing.assert_allclose(attribution.aggregate(0, frame_index=1), 0.3)
    np.testing.assert_allclose(attribution.aggregate(0), 0.3)
    output = tmp_path / "multi-frame-viewer"
    result = write_attribution_viewer(attribution, output)
    assert result["frames"] == 2
    summary = json.loads((output / "attribution.json").read_text())
    assert [frame["role"] for frame in summary["frames"]] == ["history", "current"]
    html = (output / "viewer.html").read_text()
    assert 'id="input-frame"' in html
    assert 'id="baseline"' in html
    assert "Subtract previous-token baseline" in html
    assert "strictly before this target begins" in html
    assert "Frame 0 (history)" in html
    assert "Frame 1 (current)" in html


def test_value_norms_correct_attention_and_expand_grouped_query_heads(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    norms = np.ones((1, 12), dtype=np.float32)
    norms[0, 1] = 10.0
    np.savez_compressed(
        trace / "value_norms.npz",
        layer_003=norms,
        layer_007=norms,
    )

    attribution = load_attribution(trace)
    assert attribution.value_weighted_maps is not None
    expected = np.asarray([10, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    expected = (expected / expected.sum()).reshape(2, 4)
    np.testing.assert_allclose(attribution.aggregate(0, method="value_norm"), expected)
    with pytest.raises(AttributionError, match="unknown attribution method"):
        attribution.aggregate(0, method="bogus")

    output = tmp_path / "corrected-viewer"
    result = write_attribution_viewer(attribution, output)
    assert result["value_norm_correction"] is True
    summary = json.loads((output / "attribution.json").read_text())
    assert summary["default_method"] == "value_norm"
    assert set(summary["methods"]) == {"attention", "value_norm"}
    assert (output / "saliency-value-norm-step-000.png").is_file()


def test_cross_layer_rollout_and_parameter_spans(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    prompt = np.eye(11, dtype=np.float32)
    prompt[10] = 0
    prompt[10, 1:9] = 1 / 8
    weighted = prompt.copy()
    weighted[10, 1:9] = np.asarray([10, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    np.savez_compressed(trace / "attention_prompt_mean.npz", layer_003=prompt, layer_007=prompt)
    np.savez_compressed(
        trace / "attention_prompt_value_weighted_mean.npz",
        layer_003=weighted,
        layer_007=weighted,
    )
    norms = np.ones((1, 12), dtype=np.float32)
    np.savez_compressed(trace / "value_norms.npz", layer_003=norms, layer_007=norms)

    attribution = load_attribution(trace)
    assert attribution.rollout_maps is not None
    assert attribution.value_weighted_rollout_maps is not None
    assert attribution.aggregate(0, method="rollout").sum() == pytest.approx(0.75)
    assert attribution.generated_spans[0].parameter == "action"
    assert attribution.generated_spans[0].value == "click"
    assert attribution.generated_spans[0].steps == (1,)

    output = tmp_path / "rollout-viewer"
    write_attribution_viewer(attribution, output)
    summary = json.loads((output / "attribution.json").read_text())
    assert summary["default_method"] == "value_norm_rollout"
    assert summary["generated_spans"][0]["label"] == "action = click"


def test_hand_computed_projection_rollout_and_causal_baseline(tmp_path: Path) -> None:
    attribution = load_attribution(_hand_computed_rollout_trace(tmp_path))

    # Direct attention first selects prompt positions 1 and 2, reshapes them
    # left-to-right into the 1x2 patch grid, then averages the two layers.
    np.testing.assert_allclose(
        attribution.aggregate(0, method="attention"),
        np.asarray([[0.525, 0.475]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        attribution.aggregate(1, method="attention"),
        np.asarray([[0.2, 0.6]], dtype=np.float32),
    )

    # With residual weight 1/2, top-to-bottom composition gives:
    # step 0: [0.125, 0.375] + 0.5 * 0.5 * [0.8, 0.2]
    # step 1: [0.1, 0.3] + 0.6 * 0.5 * [0.2, 0.6]
    expected_rollout_0 = np.asarray([[0.325, 0.425]], dtype=np.float32)
    expected_rollout_1 = np.asarray([[0.16, 0.48]], dtype=np.float32)
    np.testing.assert_allclose(attribution.aggregate(0, method="rollout"), expected_rollout_0)
    np.testing.assert_allclose(attribution.aggregate(1, method="rollout"), expected_rollout_1)
    np.testing.assert_allclose(
        attribution.aggregate(1, method="value_norm_rollout"),
        expected_rollout_1,
    )

    # The multi-token action value occupies steps 1 and 2. Its baseline stops
    # strictly before the sequence starts, so it contains step 0 only.
    span = attribution.generated_spans[0]
    assert span.steps == (1, 2)
    expected_rollout_2 = np.asarray([[0.33, 0.19]], dtype=np.float32)
    np.testing.assert_allclose(attribution.aggregate(2, method="rollout"), expected_rollout_2)
    np.testing.assert_allclose(
        attribution.causal_difference(span.steps, method="rollout"),
        np.mean(np.stack([expected_rollout_1, expected_rollout_2]), axis=0)
        - expected_rollout_0,
        atol=1e-7,
    )
    # A single-token target at step 1 must not use the very different step 2
    # map as a reference: future-token leakage would change this exact result.
    np.testing.assert_allclose(
        attribution.causal_difference((1,), method="rollout"),
        expected_rollout_1 - expected_rollout_0,
        atol=1e-7,
    )
    with pytest.raises(AttributionError, match="no previous generated tokens"):
        attribution.causal_difference((0,), method="rollout")


def test_viewer_quantization_round_trip_has_a_known_error_bound() -> None:
    maps = np.asarray(
        [
            [[0.0, 0.1], [0.25, 0.5]],
            [[0.04, 0.08], [0.12, 0.16]],
        ],
        dtype=np.float32,
    )
    encoded = _encoded_method("rollout", "Rollout", maps, 4, supports_slice=False)
    quantized = np.frombuffer(base64.b64decode(encoded["mapsU8"]), dtype=np.uint8).reshape(2, 4)
    scales = np.frombuffer(base64.b64decode(encoded["mapScalesF32"]), dtype="<f4")
    decoded = quantized.astype(np.float32) / 255 * scales[:, None]
    expected = maps.reshape(2, 4)

    # Rounding to the nearest uint8 code introduces at most half a code step.
    assert np.all(np.abs(decoded - expected) < scales[:, None] / (2 * 255) + 1e-7)
    np.testing.assert_allclose(decoded.max(axis=1), scales)


def test_older_trace_grid_is_inferred_and_viewer_is_self_contained(tmp_path: Path) -> None:
    trace = _trace(tmp_path, exact_grid=False)
    attribution = load_attribution(trace)
    assert (attribution.layout.rows, attribution.layout.columns) == (2, 4)
    assert attribution.layout.source == "inferred_from_image_aspect_ratio"
    assert attribution.warnings
    assert attribution.value_weighted_maps is None
    output = tmp_path / "viewer"
    result = write_attribution_viewer(attribution, output)
    assert Path(result["viewer"]).is_file()
    html = (output / "viewer.html").read_text()
    assert "data:image/png;base64," in html
    assert "__ATTRIBUTION_PAYLOAD__" not in html
    summary = json.loads((output / "attribution.json").read_text())
    assert summary["patch_grid"]["rows"] == 2
    assert len(summary["steps"]) == 2
    assert (output / "saliency-attention-step-000.png").is_file()


def test_trajectory_bundle_resolves_trace_id(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    bundle = tmp_path / "trajectories" / "v0" / "traj-test"
    bundle.mkdir(parents=True)
    (bundle / "annotations.json").write_text(json.dumps({"instrumented_trace_ids": [trace.name]}))
    assert resolve_trace_path(bundle, trace_root=trace.parent) == trace.resolve()


def test_trajectory_viewer_links_actions_to_pre_action_frames(tmp_path: Path) -> None:
    trace = _trace(tmp_path)
    bundle = tmp_path / "trajectories" / "v0" / "traj-test"
    actions = bundle / "actions"
    actions.mkdir(parents=True)
    (actions / "0000.json").write_text(
        json.dumps(
            {
                "normalized": {"action": "click", "x": 500, "y": 600},
                "raw_output": {"instrumented_trace_id": trace.name},
            }
        )
    )
    output = tmp_path / "trajectory-viewer"
    result = write_trajectory_viewer(bundle, trace.parent, output)

    assert result["frames"] == 1
    assert result["captured_click"] is True
    assert (output / "trajectory.html").is_file()
    assert (output / trace.name / "viewer.html").is_file()
    summary = json.loads((output / "trajectory.json").read_text())
    assert summary["actions"] == [{"action": "click", "x": 500, "y": 600}]
    html = (output / "trajectory.html").read_text()
    assert "data:image/png;base64," in html
    assert "__TRAJECTORY_PAYLOAD__" not in html
