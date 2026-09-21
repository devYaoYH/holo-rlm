from __future__ import annotations

from pathlib import Path

from instrumented_holo.interventions import InterventionConfig, RepresentationConfig
from instrumented_holo.model import decode_data_image
from instrumented_holo.patching_cli import (
    _apply_corruption,
    _candidates,
    _image_parts,
    _load_json_or_inline,
    load_manifest_plan,
    manifest_summary,
)
from PIL import Image


def test_portable_request_materializes_image_paths_and_swaps_exact_tiles(tmp_path: Path) -> None:
    image = Image.new("RGB", (6, 2), "black")
    for x in range(2):
        for y in range(2):
            image.putpixel((x, y), (255, 0, 0))
            image.putpixel((x + 4, y), (0, 0, 255))
    image.save(tmp_path / "screen.png")
    request = _load_json_or_inline(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_path", "path": "screen.png"}],
                }
            ]
        },
        tmp_path,
    )
    part = _image_parts(request)[0]
    clean = decode_data_image(part["image_url"]["url"])
    corrupted = _apply_corruption(
        request,
        {"type": "swap_equal_tiles", "image_index": 0, "first_box": [0, 0, 2, 2], "second_box": [4, 0, 6, 2]},
    )
    swapped = decode_data_image(_image_parts(corrupted)[0]["image_url"]["url"])
    assert clean.getpixel((0, 0)) == (255, 0, 0)
    assert swapped.getpixel((0, 0)) == (0, 0, 255)
    assert swapped.getpixel((4, 0)) == (255, 0, 0)


def test_manifest_candidates_support_coordinates_and_explicit_spans() -> None:
    candidates = _candidates(
        [
            {"label": "target", "coordinate": [482, 351]},
            {"label": "other", "text": "abc123", "scored_spans": [[3, 6]]},
        ]
    )
    assert candidates[0].label == "target"
    assert candidates[1].scored_spans == ((3, 6),)


def test_representation_configs_are_serializable_and_accept_legacy_schema() -> None:
    current = InterventionConfig.from_dict(
        {
            "name": "target-patches",
            "representation": {
                "layer": 19,
                "component": "residual_output",
                "unit": "image_region",
                "region": "target",
            },
        }
    )
    assert current.representation == RepresentationConfig(
        layer=19,
        component="residual_output",
        unit="image_region",
        region="target",
    )
    assert current.to_dict()["representation"]["region"] == "target"

    legacy = InterventionConfig.from_dict(
        {"name": "head", "kind": "attention_head", "layer": 19, "head": 10}
    )
    assert legacy.representation.component == "attention_head_output"
    assert legacy.representation.unit == "scored_token_predictions"
    assert legacy.ablation == "zero"


def test_manifest_can_be_validated_without_loading_model(tmp_path: Path) -> None:
    manifest = tmp_path / "experiment.json"
    manifest.write_text(
        """{
          "schema_version": 1,
          "clean_request": {"messages": [], "tools": []},
          "corrupted_request": {"messages": [], "tools": []},
          "candidates": [
            {"label": "target", "coordinate": [10, 20]},
            {"label": "distractor", "coordinate": [30, 40]}
          ],
          "interventions": [{
            "name": "head",
            "representation": {
              "layer": 19,
              "component": "attention_head_output",
              "unit": "scored_token_predictions",
              "head": 10
            },
            "ablation": "zero"
          }]
        }"""
    )
    summary = manifest_summary(load_manifest_plan(manifest))
    assert summary["valid"] is True
    assert summary["candidate_labels"] == ["target", "distractor"]
