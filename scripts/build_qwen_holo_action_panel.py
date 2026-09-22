"""Build the preregistered Qwen-to-Holo coordinate-action causal panel.

Every manifest has the exact official H Company element-localization request.
The remote runner supplies the checkpoint through ``HOLO_MODEL_PATH`` while
holding ``HOLO_PROCESSOR_PATH`` at Holo's processor for both checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image

from demo.screenspot import build_screenspot_request

ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "data" / "screenspot-pro" / "images" / "powerpoint_windows_59.png"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "activation_patching" / "qwen_holo_action_panel_v1"

CARDS = {
    "color_swatch": (849, 563, 1049, 735),
    "woven_fibers": (1069, 563, 1269, 735),
    "psychedelic_vibrant": (1289, 563, 1489, 735),
    "monochromatic_horizon": (1509, 563, 1709, 735),
    "artistic_neon": (1729, 563, 1929, 735),
    "quadratic_collection": (1949, 563, 2149, 735),
    "statistics_focus": (2169, 563, 2369, 735),
}

# Each tuple declares one lossless swap.  Both orientations are emitted so the
# same physical visual region is task-relevant under one prompt and irrelevant
# under its mirrored prompt.
PAIRS = (
    ("psychedelic_woven", "psychedelic_vibrant", "woven_fibers"),
    ("woven_color", "woven_fibers", "color_swatch"),
    ("monochromatic_artistic", "monochromatic_horizon", "artistic_neon"),
    ("quadratic_statistics", "quadratic_collection", "statistics_focus"),
)
DISPLAY = {
    "color_swatch": "Color swatch",
    "woven_fibers": "Woven fibers",
    "psychedelic_vibrant": "Psychedelic vibrant",
    "monochromatic_horizon": "Monochromatic horizon",
    "artistic_neon": "Artistic neon",
    "quadratic_collection": "Quadratic collection",
    "statistics_focus": "Statistics focus",
}
ACTION_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)
VISUAL_CONTROL_LAYERS = (3, 15, 27)


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _click_box(card: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return card[0], card[1], card[2], 702


def _coordinate(card: tuple[int, int, int, int]) -> list[int]:
    x1, y1, x2, y2 = _click_box(card)
    return [round((x1 + x2) * 500 / 2880), round((y1 + y2) * 500 / 1800)]


def _request(target: str, output_dir: Path) -> dict[str, Any]:
    with Image.open(IMAGE_PATH) as opened:
        request = build_screenspot_request(
            opened.convert("RGB"),
            f'Create a "{DISPLAY[target]}" presentation',
            "Hcompany/Holo-3.1-4B",
        )
    request = deepcopy(request)
    for message in request["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            if part.get("type") == "image_url":
                content[index] = {"type": "image_path", "path": _relative(IMAGE_PATH, output_dir)}
                return request
    raise RuntimeError("official ScreenSpot request did not contain one image")


def _interventions() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for layer in VISUAL_CONTROL_LAYERS:
        for region in ("target", "distractor"):
            items.append(
                {
                    "name": f"layer_{layer:02d}_{region}_visual_residuals",
                    "representation": {
                        "layer": layer,
                        "component": "residual_output",
                        "unit": "image_region",
                        "region": region,
                    },
                }
            )
    for layer in ACTION_LAYERS:
        items.append(
            {
                "name": f"layer_{layer:02d}_coordinate_span_residuals",
                "representation": {
                    "layer": layer,
                    "component": "residual_output",
                    "unit": "scored_token_predictions",
                },
                "ablation": "zero",
            }
        )
    return items


def _manifest(pair_id: str, first: str, second: str, target: str, output_dir: Path) -> dict[str, Any]:
    distractor = second if target == first else first
    return {
        "schema_version": 1,
        "name": f"qwen_holo_action_panel_v1_{pair_id}_{target}",
        "protocol": "hcompany_element_localization_v1",
        "phase": "A_visual_controls_and_coordinate_action_residuals",
        "score_reduction": "sum",
        "image_min_pixels": 65536,
        "image_max_pixels": 16777216,
        "clean_request": _request(target, output_dir),
        "corruption": {
            "type": "swap_equal_tiles",
            "image_index": 0,
            "first_box": list(CARDS[first]),
            "second_box": list(CARDS[second]),
        },
        "candidates": [
            {
                "label": "clean_instruction_target_slot",
                "json_coordinate": _coordinate(CARDS[target]),
                "scored_fields": ["x", "y"],
            },
            {
                "label": "other_swapped_slot",
                "json_coordinate": _coordinate(CARDS[distractor]),
                "scored_fields": ["x", "y"],
            },
        ],
        "regions": {
            "target": {"image_index": 0, "box": list(_click_box(CARDS[target])), "limit": 4},
            "distractor": {"image_index": 0, "box": list(_click_box(CARDS[distractor])), "limit": 4},
        },
        "interventions": _interventions(),
        "preregistration": {
            "checkpoints": ["Qwen3.5-4B base", "Holo3.1-4B fine-tuned"],
            "processor_and_template": "Holo3.1-4B processor and official H Company VisualLocalizerOutput request for both checkpoints",
            "input_order": "image tokens precede instruction tokens",
            "corruption": "Complete equal-size target/distractor tile swap with no resampling; prompt, candidate syntax, and coordinates otherwise fixed.",
            "coordinate_estimands": {
                "primary": "joint sum log p(target x,y) minus sum log p(other x,y)",
                "x": "same teacher-forced margin restricted to x digit tokens",
                "y": "conditional y-token diagnostic only; the paired y coordinate is identical in this horizontal panel",
            },
            "phase_a_sites": {
                "visual_controls": {"layers": list(VISUAL_CONTROL_LAYERS), "regions": ["target", "distractor"]},
                "action_residuals": {"layers": list(ACTION_LAYERS), "positions": "all x and y coordinate-token prediction positions"},
            },
            "phase_b_gate": (
                "Advance a layer to x-only residual patches and an all-head sweep only if, in at least 6 of 8 "
                "conditions for Holo, the full-coordinate action residual has positive restoration and positive "
                "clean ablation drop, with median restoration at least 0.10 nats; report all Phase-A rows regardless."
            ),
            "fine_tuning_hypothesis": (
                "Holo shows larger normalized action-residual recovery and paired ablation than Qwen, while "
                "visual-region effects follow the mirrored instruction rather than a fixed screen location."
            ),
            "scope_limit": "Eight within-image prompt/swap conditions are a causal panel, not an independent-image benchmark estimate.",
        },
    }


def build(output_dir: Path) -> list[Path]:
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"missing public ScreenSpot-Pro image: {IMAGE_PATH}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for pair_id, first, second in PAIRS:
        for target in (first, second):
            path = output_dir / f"{pair_id}_{target}.json"
            path.write_text(json.dumps(_manifest(pair_id, first, second, target, output_dir), indent=2) + "\n")
            paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in build(args.output.expanduser().resolve()):
        print(path)


if __name__ == "__main__":
    main()
