"""Build the base-Qwen versus Holo delta-lens manifest from official harness requests."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from PIL import Image

from demo.backends import build_request
from demo.renderer import hotel_click
from demo.runner import CHEAPEST_TASK, cheapest_progress_message
from demo.screenspot import build_screenspot_request, load_screenspot_sample

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "delta_lens" / "qwen35_4b_vs_holo31_4b.json"
HOTEL_TRAJECTORY = (
    PROJECT_ROOT / "data" / "trajectories" / "v0" / "traj-20260921T080837Z-2a0fda4eb0"
)

SCREENSPOT_CASES = (
    ("powerpoint_windows_59", (406, 351), "adjacent matched theme tile from the slide-8 intervention"),
    ("powerpoint_windows_48", (115, 150), "prior Holo diagnostic click on the same screenshot"),
    ("photoshop_windows_1", (11, 151), "prior Holo diagnostic click on the same screenshot"),
    ("photoshop_windows_9", (10, 70), "prior Holo diagnostic click on the same screenshot"),
    ("vscode_macos_3", (144, 89), "prior Holo diagnostic click on the same screenshot"),
)


def _relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _portable_image_request(request: dict[str, Any], image_path: Path, base: Path) -> dict[str, Any]:
    result = deepcopy(request)
    replaced = 0
    for message in result["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            if part.get("type") != "image_url":
                continue
            content[index] = {"type": "image_path", "path": _relative(image_path, base)}
            replaced += 1
    if replaced != 1:
        raise ValueError(f"expected exactly one new image in request, found {replaced}")
    return result


def _normalized_center(bbox: tuple[float, float, float, float], size: tuple[int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    width, height = size
    return round((x1 + x2) * 500 / width), round((y1 + y2) * 500 / height)


def _screenspot_cases(output_dir: Path) -> list[dict[str, Any]]:
    annotation_root = PROJECT_ROOT / "data" / "screenspot-pro" / "annotations"
    image_root = PROJECT_ROOT / "data" / "screenspot-pro" / "images"
    cases = []
    for sample_id, distractor, rationale in SCREENSPOT_CASES:
        sample = load_screenspot_sample(annotation_root, image_root, sample_id)
        with Image.open(sample.image_path) as opened:
            image = opened.convert("RGB")
        request = build_screenspot_request(image, sample.instruction, "Hcompany/Holo-3.1-4B")
        request = _portable_image_request(request, sample.image_path, output_dir)
        oracle = _normalized_center(sample.bbox, sample.image_size)
        cases.append(
            {
                "id": sample.id,
                "protocol": "hcompany_element_localization_v1",
                "score_reduction": "mean",
                "image_min_pixels": 65_536,
                "image_max_pixels": 16_777_216,
                "request": request,
                "candidates": [
                    {"label": "oracle_bbox_center", "json_coordinate": list(oracle)},
                    {"label": "diagnostic_distractor", "json_coordinate": list(distractor)},
                ],
                "case_metadata": {
                    "instruction": sample.instruction,
                    "application": sample.application,
                    "platform": sample.platform,
                    "ui_type": sample.ui_type,
                    "bbox": list(sample.bbox),
                    "image_size": list(sample.image_size),
                    "distractor_selection": rationale,
                    "interpretation": "exploratory matched-token diagnostic; distractors other than the slide-8 tile are post-hoc",
                },
            }
        )
    return cases


def _replace_current_image(request: dict[str, Any], frame_path: Path, output_dir: Path) -> dict[str, Any]:
    result = deepcopy(request)
    current = result["messages"][-1]
    content = current.get("content")
    if not isinstance(content, list):
        raise ValueError("hotel request does not end in a multimodal observation")
    images = [index for index, part in enumerate(content) if part.get("type") == "image_url"]
    if len(images) != 1:
        raise ValueError("hotel request must contain exactly one current screenshot")
    content[images[0]] = {"type": "image_path", "path": _relative(frame_path, output_dir)}
    return result


def _pixel_to_normalized(point: tuple[int, int], size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    return round(point[0] * 1000 / (width - 1)), round(point[1] * 1000 / (height - 1))


def _structured_step(tool_call: dict[str, Any]) -> str:
    return json.dumps(
        {"note": None, "thought": "", "tool_calls": [tool_call]},
        separators=(",", ":"),
    )


def _hotel_cases(output_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((HOTEL_TRAJECTORY / "manifest.json").read_text())
    config = manifest["fixture"]["scenario_config"]
    viewport = (int(config["viewport"]["width"]), int(config["viewport"]["height"]))
    target_id = str(config["cheapest_id"])
    distractor_id = str(config["hotels"][3]["id"])
    target_pixel = hotel_click(config, {"scroll_y": 1000}, target_id)
    distractor_pixel = hotel_click(config, {"scroll_y": 1000}, distractor_id)
    target_click = _pixel_to_normalized(target_pixel, viewport)
    distractor_click = _pixel_to_normalized(distractor_pixel, viewport)

    messages: list[dict[str, Any]] = [{"role": "user", "content": CHEAPEST_TASK}]
    cases: list[dict[str, Any]] = []
    scroll_native = {
        "tool_name": "scroll_desktop",
        "element": "hotel search results list",
        "x": 500,
        "y": 500,
        "direction": "down",
        "scroll_size": 10,
    }
    early_click = {
        "tool_name": "click_desktop",
        "element": "cheapest hotel's View details button",
        "x": target_click[0],
        "y": target_click[1],
        "button": "left",
    }
    for step in range(3):
        frame_path = HOTEL_TRAJECTORY / "frames" / f"{step:04d}-model-input.png"
        with Image.open(frame_path) as opened:
            image = opened.convert("RGB")
        request = build_request(
            messages,
            image,
            "Hcompany/Holo-3.1-4B",
            normalized_coordinates=True,
            frame_index=step,
        )
        request["chat_template_kwargs"] = {"enable_thinking": False}
        request = _replace_current_image(request, frame_path, output_dir)
        if step < 2:
            candidates = [
                {
                    "label": "oracle_scroll_down",
                    "tool_action": scroll_native,
                    "scored_fields": ["tool_name"],
                },
                {
                    "label": "premature_click",
                    "tool_action": early_click,
                    "scored_fields": ["tool_name"],
                },
            ]
        else:
            candidates = [
                {
                    "label": "oracle_cheapest_click",
                    "tool_action": {
                        "tool_name": "click_desktop",
                        "element": "Lumen Harbor Rooms View details button",
                        "x": target_click[0],
                        "y": target_click[1],
                        "button": "left",
                    },
                    "scored_fields": ["x", "y"],
                },
                {
                    "label": "visible_noncheapest_click",
                    "tool_action": {
                        "tool_name": "click_desktop",
                        "element": "visible non-cheapest hotel View details button",
                        "x": distractor_click[0],
                        "y": distractor_click[1],
                        "button": "left",
                    },
                    "scored_fields": ["x", "y"],
                },
            ]
        cases.append(
            {
                "id": f"hotel_test_0035_large_ui_step_{step}",
                "protocol": "holo_desktop_structured_v0_1_10",
                "score_reduction": "mean",
                "image_min_pixels": 65_536,
                "image_max_pixels": 16_777_216,
                "request": request,
                "candidates": candidates,
                "case_metadata": {
                    "source_trajectory": _relative(HOTEL_TRAJECTORY, output_dir),
                    "fixture_item": config["item_id"],
                    "step": step,
                    "scroll_offset": step * 500,
                    "oracle_target": config["cheapest_id"],
                    "distractor_target": distractor_id if step == 2 else "premature action type",
                    "coordinate_space": "normalized_0_1000",
                    "viewport": list(viewport),
                    "oracle_click_pixel": list(target_pixel) if step == 2 else None,
                    "oracle_click_normalized": list(target_click) if step == 2 else None,
                    "distractor_click_pixel": list(distractor_pixel) if step == 2 else None,
                    "distractor_click_normalized": list(distractor_click) if step == 2 else None,
                },
            }
        )
        if step == 2:
            continue
        messages.append(deepcopy(request["messages"][-1]))
        messages.append({"role": "assistant", "content": _structured_step(scroll_native)})
        progress = cheapest_progress_message(
            scroll_y=(step + 1) * 500,
            action={"action": "scroll", "delta_y": 500},
            all_results_seen=step == 1,
            objective_visible=step == 1,
            no_scroll_movement=False,
        )
        messages.append(
            {
                "role": "user",
                "content": f'<tool_output tool="scroll_desktop">\n{progress}\n</tool_output>',
            }
        )
    return cases


def build_manifest(output: Path) -> dict[str, Any]:
    output = output.expanduser().resolve()
    output_dir = output.parent
    return {
        "schema_version": 1,
        "name": "qwen35_4b_vs_holo31_4b_official_protocol_delta_lens",
        "processor_path": _relative(PROJECT_ROOT / "models" / "Holo-3.1-4B", output_dir),
        "models": [
            {
                "label": "Qwen3.5-4B base",
                "role": "base",
                "path": _relative(PROJECT_ROOT / "models" / "Qwen3.5-4B", output_dir),
            },
            {
                "label": "Holo3.1-4B",
                "role": "tuned",
                "path": _relative(PROJECT_ROOT / "models" / "Holo-3.1-4B", output_dir),
            },
        ],
        "comparison_contract": {
            "processor_and_chat_template": "Holo3.1-4B for both checkpoints",
            "model_loading": "sequential",
            "delta_sign": "Holo3.1-4B minus Qwen3.5-4B base",
            "screen_prompt": "H Company's official hcompany_element_localization_v1 image-first protocol",
            "hotel_prompt": "checked-in hotel prompt with official HoloDesktop 0.1.10 structured tool schema",
            "thinking": False,
            "temperature": 0,
        },
        "cases": [*_screenspot_cases(output_dir), *_hotel_cases(output_dir)],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_manifest(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
