"""Build a preregistered, native-resolution spatial patching panel.

The panel uses several label cards on one public ScreenSpot-Pro PowerPoint
screenshot.  It is deliberately a *within-image* replication panel: it tests
whether clean-to-corrupted residual interchange is specific to the instructed
card across multiple visual locations, but does not pretend to be an
independent-screenshot benchmark.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = PROJECT_ROOT / "data" / "screenspot-pro" / "images" / "powerpoint_windows_59.png"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "activation_patching" / "template_swap_panel_v1"

# The source image is 2880 x 1800.  The rectangles are fixed cards in its
# template row.  We swap the complete equal-size card (172 px high), while
# attribution/activation regions retain only the clickable label height.
CARDS = {
    "color_swatch": (849, 563, 1049, 735),
    "woven_fibers": (1069, 563, 1269, 735),
    "psychedelic_vibrant": (1289, 563, 1489, 735),
    "monochromatic_horizon": (1509, 563, 1709, 735),
    "artistic_neon": (1729, 563, 1929, 735),
    "quadratic_collection": (1949, 563, 2149, 735),
    "statistics_focus": (2169, 563, 2369, 735),
}

# Frozen before model execution.  Several items reuse a card as a distractor
# only; each condition has a distinct target instruction and target location.
PANEL = (
    ("psychedelic_vs_woven", "Psychedelic vibrant", "psychedelic_vibrant", "woven_fibers"),
    ("woven_vs_color", "Woven fibers", "woven_fibers", "color_swatch"),
    ("monochromatic_vs_artistic", "Monochromatic horizon", "monochromatic_horizon", "artistic_neon"),
    ("quadratic_vs_statistics", "Quadratic collection", "quadratic_collection", "statistics_focus"),
)

# These are the full-attention layers.  Phase B may sweep heads only in a
# layer that shows a replicated target-specific residual effect in Phase A.
PHASE_A_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31)


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _click_box(card: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return card[0], card[1], card[2], 702


def _normalized_center(card: tuple[int, int, int, int]) -> list[int]:
    # The source screen is exactly 2880 x 1800; use the visible label/card
    # centre and the documented 0..1000 ScreenSpot coordinate convention.
    x1, y1, x2, y2 = _click_box(card)
    return [round((x1 + x2) * 500 / 2880), round((y1 + y2) * 500 / 1800)]


def _portable_request(instruction: str, output_dir: Path) -> dict[str, Any]:
    with Image.open(IMAGE_PATH) as opened:
        request = build_screenspot_request(
            opened.convert("RGB"),
            f'Create a "{instruction}" presentation',
            "Hcompany/Holo-3.1-4B",
        )
    request = deepcopy(request)
    for message in request["messages"]:
        if not isinstance(message.get("content"), list):
            continue
        for index, part in enumerate(message["content"]):
            if part.get("type") == "image_url":
                message["content"][index] = {"type": "image_path", "path": _relative(IMAGE_PATH, output_dir)}
                return request
    raise RuntimeError("official ScreenSpot request did not contain an image")


def _interventions() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for layer in PHASE_A_LAYERS:
        for region in ("target", "distractor"):
            result.append(
                {
                    "name": f"layer_{layer:02d}_{region}_residuals",
                    "representation": {
                        "layer": layer,
                        "component": "residual_output",
                        "unit": "image_region",
                        "region": region,
                    },
                }
            )
    return result


def _head_interventions(layer: int) -> list[dict[str, Any]]:
    """Exhaustive post-screen head panel for a predeclared selected layer."""

    return [
        {
            "name": f"layer_{layer:02d}_head_{head:02d}",
            "representation": {
                "layer": layer,
                "component": "attention_head_output",
                "unit": "scored_token_predictions",
                "head": head,
            },
            "ablation": "zero",
        }
        for head in range(16)
    ]


def _manifest(case: tuple[str, str, str, str], output_dir: Path) -> dict[str, Any]:
    case_id, instruction, target_name, distractor_name = case
    target = CARDS[target_name]
    distractor = CARDS[distractor_name]
    return {
        "schema_version": 1,
        "name": f"template_swap_panel_v1_{case_id}_official_json",
        "protocol": "hcompany_element_localization_v1",
        "phase": "A_spatial_residual_layer_sweep",
        "score_reduction": "mean",
        "image_min_pixels": 65536,
        "image_max_pixels": 16777216,
        "clean_request": _portable_request(instruction, output_dir),
        "corruption": {
            "type": "swap_equal_tiles",
            "image_index": 0,
            "first_box": list(target),
            "second_box": list(distractor),
        },
        "candidates": [
            {"label": "instructed_target_slot", "json_coordinate": _normalized_center(target)},
            {"label": "matched_distractor_slot", "json_coordinate": _normalized_center(distractor)},
        ],
        "regions": {
            "target": {"image_index": 0, "box": list(_click_box(target)), "limit": 4},
            "distractor": {"image_index": 0, "box": list(_click_box(distractor)), "limit": 4},
        },
        "interventions": _interventions(),
        "preregistration": {
            "panel_type": "within_image_spatial_replication",
            "selection_rule_for_phase_b": (
                "A layer advances only if its target residual interchange has positive restoration and "
                "positive clean-run ablation drop in at least 3 of 4 panel cases, and its median target "
                "restoration exceeds the same-layer distractor-region median."
            ),
            "phase_b_rule": (
                "Sweep all attention heads at every Phase-A-selected layer across all four frozen cases; "
                "do not select heads using only a single case."
            ),
            "scope_limit": "This panel is not an independent-image estimate or a ScreenSpot-Pro accuracy result.",
        },
    }


def build(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"missing public ScreenSpot-Pro image: {IMAGE_PATH}")
    paths: list[Path] = []
    for case in PANEL:
        path = output_dir / f"{case[0]}.json"
        path.write_text(json.dumps(_manifest(case, output_dir), indent=2) + "\n")
        paths.append(path)
    return paths


def build_head_sweep(output_dir: Path, layer: int) -> list[Path]:
    """Create an all-head sweep without changing the frozen cases or prompt."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"missing public ScreenSpot-Pro image: {IMAGE_PATH}")
    paths: list[Path] = []
    for case in PANEL:
        manifest = _manifest(case, output_dir)
        manifest["name"] = f"template_swap_panel_v1_{case[0]}_official_json_layer_{layer:02d}_head_sweep"
        manifest["phase"] = "B_exhaustive_attention_head_sweep"
        manifest["interventions"] = _head_interventions(layer)
        manifest["phase_b_selection"] = {
            "layer": layer,
            "rule": manifest["preregistration"]["selection_rule_for_phase_b"],
            "interpretation": (
                "The selection rule is permissive (sign-only). Report absolute effect sizes; a head result "
                "is not a circuit claim unless it is materially sized, directionally paired, and replicated."
            ),
        }
        path = output_dir / f"{case[0]}.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--head-sweep-layer",
        type=int,
        help="create a separate all-16-head phase-B manifest set for this layer",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    paths = (
        build_head_sweep(output, args.head_sweep_layer)
        if args.head_sweep_layer is not None
        else build(output)
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
