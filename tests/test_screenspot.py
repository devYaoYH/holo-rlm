from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from demo.screenspot import (
    build_screenspot_request,
    list_screenspot_samples,
    load_screenspot_sample,
    parse_screenspot_click,
    point_hits_bbox,
    repair_screenspot_click,
)
from demo.screenspot_benchmark import _validate_benchmark_server_profile


def test_load_build_parse_and_score_screenspot_sample(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    images = tmp_path / "images"
    annotations.mkdir()
    images.mkdir()
    image = Image.new("RGB", (200, 100), "white")
    image.save(images / "sample-1.png")
    (annotations / "app.json").write_text(
        json.dumps(
            [
                {
                    "id": "sample-1",
                    "instruction": "Click the target.",
                    "img_filename": "missing/original.png",
                    "bbox": [80, 30, 120, 70],
                    "img_size": [200, 100],
                    "application": "fixture",
                    "platform": "test",
                    "ui_type": "text",
                }
            ]
        )
    )

    sample = load_screenspot_sample(annotations, images, "sample-1")
    request = build_screenspot_request(image, sample.instruction, "test-model", trace_generation_steps=64)
    assert request["temperature"] == 0
    assert request["trace"]["max_generation_steps"] == 64
    assert request["trace"]["capture_logprobs"] is True
    assert request["trace"]["capture_attentions"] is True
    assert "tools" not in request
    assert request["structured_outputs"]["json"]["required"] == ["x", "y"]
    assert request["messages"][0]["content"][0]["type"] == "image_url"
    assert "Localize an element on the GUI image" in request["messages"][0]["content"][1]["text"]
    assert [value.id for value in list_screenspot_samples(annotations, images)] == ["sample-1"]

    logprob_request = build_screenspot_request(
        image,
        sample.instruction,
        "test-model",
        trace_generation_steps=64,
        trace_profile="logprobs",
    )
    assert logprob_request["trace"]["capture_logprobs"] is True
    assert logprob_request["trace"]["capture_attentions"] is False

    response = {
        "choices": [
            {
                "message": {
                    "content": '{"x":500,"y":500}',
                }
            }
        ]
    }
    click = parse_screenspot_click(response, sample.image_size)
    assert click["pixel"] == {"x": 100.0, "y": 50.0}
    assert point_hits_bbox(click["pixel"], sample.bbox)
    assert not point_hits_bbox({"x": 121.0, "y": 50.0}, sample.bbox)


def test_repair_screenspot_click_is_narrow_and_explicit() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": '{"x":"484>","y":"351>"}',
                }
            }
        ]
    }
    click = repair_screenspot_click(response, (1000, 500))
    assert click["normalized"] == {"x": 484, "y": 351}
    assert click["pixel"] == {"x": 484.0, "y": 175.5}
    assert click["source"] == {"x": "484>", "y": "351>"}
    assert click["repair"] == "first_unsigned_integer_per_coordinate_field"


def test_benchmark_rejects_known_downsampled_server_profile() -> None:
    metadata = {"inference_configuration": {"image_max_pixels": 262_144}}
    try:
        _validate_benchmark_server_profile(metadata)
    except ValueError as exc:
        assert "HOLO_IMAGE_MAX_PIXELS=16777216" in str(exc)
    else:
        raise AssertionError("downsampled ScreenSpot server profile was accepted")

    _validate_benchmark_server_profile(
        {"inference_configuration": {"image_max_pixels": 16_777_216}}
    )
