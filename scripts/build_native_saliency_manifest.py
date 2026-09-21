"""Build native-resolution paired prompt-control manifests for slides 7 and 8."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from src.demo.backends import MODEL_ACTION_SCHEMA, SYSTEM_PROMPT
from src.demo.screenspot import SCREENSPOT_LOCALIZATION_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks/attention_attribution/powerpoint48_hotel35_native_controls_v1.json"
LOCALIZER_PREFIX = (
    "Localize an element on the GUI image according to the provided target "
    "and output a click position.\n"
    f" * You must output a valid JSON following the format: {SCREENSPOT_LOCALIZATION_SCHEMA}\n"
    " Your target is:\n"
)


def _relative(path: Path, base: Path) -> str:
    return str(Path("../..") / path.relative_to(ROOT)) if base.name == "attention_attribution" else str(path)


def _normalized_center(box: list[int], size: tuple[int, int]) -> list[int]:
    x1, y1, x2, y2 = box
    width, height = size
    return [
        round(((x1 + x2) / 2) * 1000 / (width - 1)),
        round(((y1 + y2) / 2) * 1000 / (height - 1)),
    ]


def _ppt_case(
    *,
    case_id: str,
    instruction: str,
    bbox: list[int],
    role: str,
    output_dir: Path,
) -> dict[str, Any]:
    size = (2880, 1800)
    coordinate = _normalized_center(bbox, size)
    image_path = _relative(ROOT / "data/screenspot-pro/images/powerpoint_windows_48.png", output_dir)
    return {
        "id": case_id,
        "protocol": "hcompany_element_localization_v1",
        "request": {
            "model": "Hcompany/Holo-3.1-4B",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_path", "path": image_path},
                        {"type": "text", "text": LOCALIZER_PREFIX + instruction},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": {"json": SCREENSPOT_LOCALIZATION_SCHEMA},
        },
        "candidates": [
            {"label": "curated_oracle_center", "json_coordinate": coordinate},
            {"label": "target_anchor", "json_coordinate": [66, 58]},
        ],
        "case_metadata": {
            "ensemble_group": "powerpoint_windows_48",
            "ensemble_role": role,
            "instruction": instruction,
            "bbox": bbox,
            "bbox_source": (
                "ScreenSpot-Pro annotation" if role == "target" else "manually curated from the source screenshot"
            ),
            "image_size": list(size),
            "frame_bboxes": [bbox],
        },
    }


def _hotel_messages(instruction: str, output_dir: Path) -> list[dict[str, Any]]:
    trajectory = ROOT / "data/trajectories/v0/traj-20260921T080837Z-2a0fda4eb0/frames"
    images = [
        _relative(trajectory / f"{index:04d}-model-input.png", output_dir)
        for index in range(3)
    ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Inspect the hotel search results across the screenshots. Take exactly one desktop action per turn."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Exact hotel-search screenshot for frame 0."},
                {"type": "image_path", "path": images[0]},
            ],
        },
        {
            "role": "assistant",
            "content": (
                "<tool_call>\n<function=desktop_action>\n<parameter=action>\nscroll\n</parameter>\n"
                "<parameter=delta_y>\n-500\n</parameter>\n</function>\n</tool_call>"
            ),
        },
        {
            "role": "user",
            "content": "The previous scroll was applied. Continue inspecting the adjacent results.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Exact hotel-search screenshot for frame 1."},
                {"type": "image_path", "path": images[1]},
            ],
        },
        {
            "role": "assistant",
            "content": (
                "<tool_call>\n<function=desktop_action>\n<parameter=action>\nscroll\n</parameter>\n"
                "<parameter=delta_y>\n-500\n</parameter>\n</function>\n</tool_call>"
            ),
        },
        {"role": "user", "content": instruction},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Exact hotel-search screenshot for frame 2; this is the current frame."},
                {"type": "image_path", "path": images[2]},
            ],
        },
    ]


def _hotel_case(
    *,
    case_id: str,
    instruction: str,
    bbox: list[int],
    role: str,
    output_dir: Path,
) -> dict[str, Any]:
    coordinate = _normalized_center(bbox, (1024, 720))
    return {
        "id": case_id,
        "protocol": "holo_desktop_action_v1",
        "request": {
            "model": "Hcompany/Holo-3.1-4B",
            "messages": _hotel_messages(instruction, output_dir),
            "temperature": 0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "desktop_action",
                        "parameters": copy.deepcopy(MODEL_ACTION_SCHEMA),
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "desktop_action"}},
        },
        "candidates": [
            {
                "label": "curated_oracle_click",
                "tool_action": {"action": "click", "x": coordinate[0], "y": coordinate[1]},
                "scored_fields": ["x", "y"],
            },
            {
                "label": "target_anchor",
                "tool_action": {"action": "click", "x": 642, "y": 616},
                "scored_fields": ["x", "y"],
            },
        ],
        "case_metadata": {
            "ensemble_group": "hotel_test_0035_large_ui_step_2",
            "ensemble_role": role,
            "instruction": instruction,
            "bbox": bbox,
            "bbox_source": "fixture layout geometry",
            "image_size": [1024, 720],
            "frame_bboxes": [None, None, bbox],
            "source_trajectory": "data/trajectories/v0/traj-20260921T080837Z-2a0fda4eb0",
        },
    }


def build(output: Path) -> dict[str, Any]:
    output = output.expanduser().resolve()
    cases = [
        _ppt_case(
            case_id="powerpoint_windows_48_target_new_slide",
            instruction="Create new slide",
            bbox=[164, 68, 218, 139],
            role="target",
            output_dir=output.parent,
        ),
        _ppt_case(
            case_id="powerpoint_windows_48_control_title",
            instruction="Click the title placeholder",
            bbox=[735, 535, 2175, 914],
            role="control",
            output_dir=output.parent,
        ),
        _ppt_case(
            case_id="powerpoint_windows_48_control_subtitle",
            instruction="Click the subtitle placeholder",
            bbox=[735, 925, 2175, 1189],
            role="control",
            output_dir=output.parent,
        ),
        _ppt_case(
            case_id="powerpoint_windows_48_control_insert",
            instruction="Open the Insert tab",
            bbox=[258, 38, 314, 65],
            role="control",
            output_dir=output.parent,
        ),
        _ppt_case(
            case_id="powerpoint_windows_48_control_template",
            instruction="Select the red Standard Template in the right sidebar",
            bbox=[2480, 674, 2725, 837],
            role="control",
            output_dir=output.parent,
        ),
        _hotel_case(
            case_id="hotel35_target_cheapest_button",
            instruction="Every hotel price has appeared. Click View details for the cheapest hotel.",
            bbox=[601, 414, 714, 472],
            role="target",
            output_dir=output.parent,
        ),
        _hotel_case(
            case_id="hotel35_control_juniper_button",
            instruction="Click View details for Juniper Signal Inn.",
            bbox=[601, 156, 714, 214],
            role="control",
            output_dir=output.parent,
        ),
        _hotel_case(
            case_id="hotel35_control_ember_button",
            instruction="Click View details for Ember Quay Inn.",
            bbox=[601, 672, 714, 719],
            role="control",
            output_dir=output.parent,
        ),
        _hotel_case(
            case_id="hotel35_control_lumen_name",
            instruction="Click the Lumen Harbor Rooms hotel name.",
            bbox=[495, 296, 645, 316],
            role="control",
            output_dir=output.parent,
        ),
        _hotel_case(
            case_id="hotel35_control_logo",
            instruction="Click the StayLocal logo.",
            bbox=[31, 31, 192, 50],
            role="control",
            output_dir=output.parent,
        ),
    ]
    manifest = {
        "schema_version": 1,
        "name": "powerpoint48_hotel35_native_controls_v1",
        "processor_path": "../../models/Holo-3.1-4B",
        "models": [
            {"label": "Qwen3.5-4B", "role": "base", "path": "../../models/Qwen3.5-4B"},
            {"label": "Holo3.1-4B", "role": "tuned", "path": "../../models/Holo-3.1-4B"},
        ],
        "image_min_pixels": 65536,
        "image_max_pixels": 16777216,
        "score_reduction": "mean",
        "experiment_design": {
            "primary_method": "value_norm_attention",
            "construction": (
                "For each checkpoint and group, subtract the mean of four separately L1-normalized "
                "same-image alternate-instruction maps from the separately normalized target map."
            ),
            "teacher_forcing": (
                "Every target and control uses an independently image-derived oracle coordinate; no model rollout "
                "selects a coordinate."
            ),
            "resolution": (
                "Checkpoint-native image_max_pixels=16777216; no diagnostic max-frame-width downsampling."
            ),
            "hotel_history": (
                "All five hotel conditions use identical three-frame image and scroll history. The generic official "
                "hotel system prompt is retained; only the final visible-target instruction changes."
            ),
        },
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "cases": len(result["cases"])}, indent=2))


if __name__ == "__main__":
    main()
