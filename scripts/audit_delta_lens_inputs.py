"""Audit delta-lens image and coordinate invariants without loading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from instrumented_holo.delta_lens import load_delta_lens_plan, override_plan_paths
from instrumented_holo.model import normalise_messages


def _images(messages: list[dict[str, Any]]) -> list[Any]:
    return [
        part["image"]
        for message in normalise_messages(messages)
        for part in (message["content"] if isinstance(message["content"], list) else [])
        if part.get("type") == "image"
    ]


def _coordinate_schema_ok(raw_case: dict[str, Any]) -> bool:
    request = raw_case["request"]
    if raw_case["protocol"] == "hcompany_element_localization_v1":
        schema = request["structured_outputs"]["json"]
        return all(
            schema["properties"][axis]["minimum"] == 0
            and schema["properties"][axis]["maximum"] == 1000
            for axis in ("x", "y")
        )
    click = request["tools"][0]["function"]["parameters"]["oneOf"][0]
    return all(
        click["properties"][axis]["minimum"] == 0
        and click["properties"][axis]["maximum"] == 1000
        for axis in ("x", "y")
    )


def _candidate_coordinates_ok(raw_case: dict[str, Any]) -> bool:
    for candidate in raw_case["candidates"]:
        coordinate = candidate.get("json_coordinate")
        if coordinate is not None and (
            len(coordinate) != 2 or any(type(value) is not int or not 0 <= value <= 1000 for value in coordinate)
        ):
            return False
        action = candidate.get("tool_action")
        if (
            isinstance(action, dict)
            and action.get("action") == "click"
            and any(type(action.get(axis)) is not int or not 0 <= action[axis] <= 1000 for axis in ("x", "y"))
        ):
            return False
    return True


def audit(manifest_path: Path, *, processor_path: Path | None = None) -> dict[str, Any]:
    from transformers import AutoProcessor

    plan = override_plan_paths(
        load_delta_lens_plan(manifest_path, require_processor=processor_path is None),
        processor_path=processor_path,
    )
    if not plan.processor_path.is_dir():
        raise FileNotFoundError(f"processor checkpoint is missing: {plan.processor_path}")
    processor = AutoProcessor.from_pretrained(plan.processor_path, local_files_only=True)
    patch_size = int(processor.image_processor.patch_size)
    rows = []
    raw_by_id = {str(case["id"]): case for case in plan.raw["cases"]}
    for case in plan.cases:
        raw_case = raw_by_id[case.id]
        processor.image_processor.size = {
            "shortest_edge": case.image_min_pixels,
            "longest_edge": case.image_max_pixels,
        }
        normalized = normalise_messages(case.request["messages"])
        images = _images(case.request["messages"])
        encoded = processor.apply_chat_template(
            normalized,
            tools=case.request.get("tools", []),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **{"enable_thinking": False, **case.request.get("chat_template_kwargs", {})},
        )
        grids = encoded["image_grid_thw"].detach().cpu().tolist()
        if len(images) != len(grids):
            raise RuntimeError(f"case {case.id!r} image count and processor grid count differ")
        image_rows = []
        for image, (_temporal, grid_h, grid_w) in zip(images, grids, strict=True):
            source_width, source_height = image.size
            effective_width = int(grid_w) * patch_size
            effective_height = int(grid_h) * patch_size
            ratio = effective_width * effective_height / (source_width * source_height)
            image_rows.append(
                {
                    "source_size": [source_width, source_height],
                    "source_pixels": source_width * source_height,
                    "pixel_ceiling": case.image_max_pixels,
                    "source_below_ceiling": source_width * source_height <= case.image_max_pixels,
                    "processor_grid_thw": [int(value) for value in (_temporal, grid_h, grid_w)],
                    "processor_effective_size": [effective_width, effective_height],
                    "effective_area_ratio": ratio,
                    "no_material_downsample": ratio >= 0.90,
                }
            )
        rows.append(
            {
                "id": case.id,
                "protocol": case.protocol,
                "coordinate_schema_0_1000": _coordinate_schema_ok(raw_case),
                "candidate_coordinates_0_1000": _candidate_coordinates_ok(raw_case),
                "images": image_rows,
            }
        )
        del encoded
    ok = all(
        row["coordinate_schema_0_1000"]
        and row["candidate_coordinates_0_1000"]
        and all(image["source_below_ceiling"] and image["no_material_downsample"] for image in row["images"])
        for row in rows
    )
    return {
        "schema_version": 1,
        "ok": ok,
        "manifest": str(plan.path),
        "processor": str(plan.processor_path),
        "patch_size": patch_size,
        "cases": rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--processor", type=Path, help="override the shared Holo processor path")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit(args.manifest, processor_path=args.processor)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
