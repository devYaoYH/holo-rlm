"""The viewer must not confuse tuple ordinals with physical Holo blocks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from apps.trace_viewer.server import step_data, trace_metadata


def write(path, value):
    path.write_text(json.dumps(value))


def test_step_uses_physical_block_and_current_query(tmp_path):
    path = tmp_path / "trace-test"
    path.mkdir()
    write(path / "model.json", {"model_type": "qwen3_5", "attention_layer_indices": [3, 7],
                                "vision_config": {"spatial_merge_size": 2}})
    write(path / "positions.json", {"prompt_token_count": 7, "generated_token_count": 2, "image_token_id": 99})
    write(path / "vision_inputs.json", {"image_grid_thw": {"values": [[1, 2, 4]]}})
    write(path / "generated_tokens.json", {"token_ids": [10, 11], "tokens": ["a", "b"]})
    np.save(path / "input_ids.npy", np.array([[1, 99, 99, 2, 3, 4, 5]]))
    np.save(path / "generated_ids.npy", np.array([[10, 11]]))
    np.savez_compressed(path / "attention_last_query_rows.npz",
                        step_000_layer_000=np.full((2, 7), 0.1),
                        step_000_layer_001=np.full((2, 7), 0.2),
                        step_001_layer_000=np.full((2, 8), 0.3),
                        step_001_layer_001=np.full((2, 8), 0.4))
    np.savez_compressed(path / "hidden_state_last_query_rows.npz",
                        step_000_layer_003=np.array([0.0, -1.0]),
                        step_000_layer_004=np.array([1.0, -2.0]),
                        step_001_layer_003=np.array([1.0, -1.0]),
                        step_001_layer_004=np.array([3.0, -4.0]))
    meta = trace_metadata(path)
    assert meta["images"][0]["start"] == 1
    assert (meta["images"][0]["rows"], meta["images"][0]["cols"]) == (1, 2)
    first = step_data(path, 0, 3, -1)
    second = step_data(path, 1, 3, -1)
    assert first["query_position"] == 6
    assert second["query_position"] == 7
    assert first["attention"]["row_sum"] == pytest.approx(0.7)
    assert second["attention"]["row_sum"] == pytest.approx(2.4)
    assert second["vector"] == [3.0, -4.0]
    assert second["delta_vector"] == [2.0, -3.0]
    assert second["delta_norm"] == pytest.approx(np.sqrt(13))
    assert step_data(path, 1, 0, -1)["attention"] is None


def test_early_vision_metadata_without_grid_values(tmp_path):
    path = tmp_path / "trace-early"
    path.mkdir()
    write(path / "model.json", {"model_type": "qwen3_5"})
    write(path / "positions.json", {"prompt_token_count": 3, "generated_token_count": 1, "image_token_id": 99})
    write(path / "vision_inputs.json", {"image_grid_thw": [1, 3]})
    write(path / "generated_tokens.json", {"token_ids": [10], "tokens": ["a"]})
    np.save(path / "input_ids.npy", np.array([[1, 99, 2]]))
    meta = trace_metadata(path)
    assert meta["images"][0]["rows"] is None
    assert meta["full_attention_layers"] == list(range(3, 32, 4))
