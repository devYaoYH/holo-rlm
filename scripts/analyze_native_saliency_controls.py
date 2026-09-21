"""Analyze and render native-resolution paired prompt-control saliency ensembles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


METHODS = ("value_norm_attention", "direct_attention")
ROLES = ("base", "tuned")


def _flatten(frames: list[list[float]]) -> np.ndarray:
    values = np.concatenate([np.asarray(frame, dtype=np.float64) for frame in frames])
    if not math.isclose(float(values.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"attention map must be L1-normalized across frames, got {values.sum():.9f}")
    return values


def _split(values: np.ndarray, grids: list[list[int]]) -> list[np.ndarray]:
    result = []
    offset = 0
    for rows, columns in grids:
        count = rows * columns
        result.append(values[offset : offset + count].reshape(rows, columns))
        offset += count
    if offset != len(values):
        raise ValueError("attention vector length does not match image grids")
    return result


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _patch_indices(
    bbox: list[float], image_size: list[int], grid: list[int]
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    width, height = image_size
    rows, columns = grid
    indices = []
    for row in range(rows):
        patch_y1 = row * height / rows
        patch_y2 = (row + 1) * height / rows
        if patch_y2 <= y1 or patch_y1 >= y2:
            continue
        for column in range(columns):
            patch_x1 = column * width / columns
            patch_x2 = (column + 1) * width / columns
            if patch_x2 <= x1 or patch_x1 >= x2:
                continue
            indices.append(row * columns + column)
    if not indices:
        raise ValueError(f"bbox {bbox} does not overlap grid {grid}")
    return np.asarray(indices, dtype=np.int64)


def _signed_overlay(values: np.ndarray, ceiling: float) -> Image.Image:
    normalized = np.clip(np.abs(values) / max(ceiling, 1e-12), 0.0, 1.0) ** 0.55
    positive = values >= 0
    red = np.where(positive, 240, 45) * normalized
    green = np.where(positive, 90, 132) * normalized
    blue = np.where(positive, 57, 211) * normalized
    alpha = np.clip(normalized * 196, 0, 196)
    rgba = np.stack((red, green, blue, alpha), axis=-1).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _composite(source: Image.Image, heatmap: Image.Image, bbox: list[float] | None) -> Image.Image:
    resized = heatmap.resize(source.size, Image.Resampling.BILINEAR)
    result = Image.alpha_composite(source.convert("RGBA"), resized)
    if bbox is not None:
        draw = ImageDraw.Draw(result)
        width = max(4, round(source.width / 500))
        box = tuple(round(value) for value in bbox)
        draw.rectangle(box, outline=(255, 255, 255, 255), width=width)
        draw.rectangle(
            (box[0] - width, box[1] - width, box[2] + width, box[3] + width),
            outline=(24, 168, 117, 255),
            width=width,
        )
    return result.convert("RGB")


def _resolve_images(manifest_path: Path, case: dict[str, Any]) -> list[Path]:
    result = []
    for message in case["request"]["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image_path":
                continue
            path = Path(part["path"])
            result.append(path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve())
    return result


def _group_summary(
    *,
    group: str,
    target_row: dict[str, Any],
    control_rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    target_id = target_row["id"]
    control_ids = [row["id"] for row in control_rows]
    cases_by_role = {
        role: {case["id"]: case for case in payloads[role]["cases"]}
        for role in ROLES
    }
    target_case = cases_by_role["tuned"][target_id]
    grids = target_case["image_grids"]
    sizes = target_case["image_sizes"]
    frame_bboxes = target_row["case_metadata"].get("frame_bboxes", [target_row["case_metadata"]["bbox"]])
    if len(frame_bboxes) != len(grids):
        raise ValueError(f"group {group!r} frame bbox count does not match image grids")

    target_frame_index = next(
        (index for index, bbox in reversed(list(enumerate(frame_bboxes))) if bbox is not None), None
    )
    if target_frame_index is None:
        raise ValueError(f"group {group!r} has no target bbox")
    target_indices = _patch_indices(
        frame_bboxes[target_frame_index], sizes[target_frame_index], grids[target_frame_index]
    )
    frame_offsets = np.cumsum([0] + [rows * columns for rows, columns in grids])
    global_target_indices = target_indices + frame_offsets[target_frame_index]
    total_patches = int(frame_offsets[-1])
    target_area_fraction = len(global_target_indices) / total_patches

    method_results: dict[str, Any] = {}
    render_paths: dict[str, Any] = {}
    images = _resolve_images(manifest_path, target_row)
    if len(images) != len(grids):
        raise ValueError(f"group {group!r} image count does not match grids")
    group_dir = output_dir / group
    group_dir.mkdir(parents=True, exist_ok=True)

    for method in METHODS:
        role_maps: dict[str, np.ndarray] = {}
        role_results: dict[str, Any] = {}
        for role in ROLES:
            target = _flatten(cases_by_role[role][target_id]["maps"][method])
            controls = np.stack(
                [_flatten(cases_by_role[role][case_id]["maps"][method]) for case_id in control_ids]
            )
            control_mean = controls.mean(axis=0)
            difference = target - control_mean
            positive = np.clip(difference, 0.0, None)
            positive_mass = float(positive.sum())
            target_positive = float(positive[global_target_indices].sum())
            leave_one_out = []
            for index, control_id in enumerate(control_ids):
                reduced = target - np.delete(controls, index, axis=0).mean(axis=0)
                leave_one_out.append(
                    {
                        "excluded_control": control_id,
                        "cosine_similarity": _cosine(difference, reduced),
                    }
                )
            frame_maps = _split(difference, grids)
            role_maps[role] = difference
            role_results[role] = {
                "target_region_mass_by_instruction": {
                    target_id: float(target[global_target_indices].sum()),
                    **{
                        control_id: float(controls[index, global_target_indices].sum())
                        for index, control_id in enumerate(control_ids)
                    },
                },
                "control_mean_target_region_mass": float(control_mean[global_target_indices].sum()),
                "prompt_difference_target_region_mass": float(difference[global_target_indices].sum()),
                "target_region_lift_over_global_patch_area": (
                    float(difference[global_target_indices].sum()) / target_area_fraction
                ),
                "target_share_of_positive_prompt_difference": (
                    target_positive / positive_mass if positive_mass else 0.0
                ),
                "frame_prompt_difference_mass": [float(frame.sum()) for frame in frame_maps],
                "frame_positive_prompt_difference_mass": [
                    float(np.clip(frame, 0.0, None).sum()) for frame in frame_maps
                ],
                "leave_one_control_out": leave_one_out,
                "minimum_leave_one_out_cosine": min(row["cosine_similarity"] for row in leave_one_out),
                "map": difference.tolist(),
            }

        delta = role_maps["tuned"] - role_maps["base"]
        method_results[method] = {
            **role_results,
            "delta_prompt_difference_target_region_mass": float(delta[global_target_indices].sum()),
            "delta_map": delta.tolist(),
        }

        maps_to_render = {"qwen": role_maps["base"], "holo": role_maps["tuned"], "holo-minus-qwen": delta}
        split_maps = {name: _split(values, grids) for name, values in maps_to_render.items()}
        shared = np.concatenate([np.abs(frame).ravel() for name in ("qwen", "holo") for frame in split_maps[name]])
        shared_ceiling = float(np.percentile(shared, 99.5))
        delta_values = np.concatenate([np.abs(frame).ravel() for frame in split_maps["holo-minus-qwen"]])
        delta_ceiling = float(np.percentile(delta_values, 99.5))
        slug = method.replace("_attention", "").replace("_", "-")
        render_paths[method] = {}
        for name, frame_maps in split_maps.items():
            ceiling = delta_ceiling if name == "holo-minus-qwen" else shared_ceiling
            paths = []
            for frame_index, (image_path, frame_map) in enumerate(zip(images, frame_maps, strict=True)):
                with Image.open(image_path) as opened:
                    source = opened.convert("RGB")
                destination = group_dir / f"{name}-{slug}-frame-{frame_index}.png"
                _composite(source, _signed_overlay(frame_map, ceiling), frame_bboxes[frame_index]).save(
                    destination, quality=94
                )
                paths.append(str(destination))
            render_paths[method][name] = paths

    return {
        "target_case": target_id,
        "control_cases": control_ids,
        "source_images": [str(path) for path in images],
        "image_sizes": sizes,
        "image_grids": grids,
        "frame_bboxes": frame_bboxes,
        "target_frame_index": target_frame_index,
        "target_patch_indices": global_target_indices.tolist(),
        "target_patch_area_fraction": target_area_fraction,
        "methods": method_results,
        "rendered": render_paths,
    }


def analyze(result_dir: Path, output_dir: Path) -> dict[str, Any]:
    result_dir = result_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    snapshot_path = result_dir / "manifest.snapshot.json"
    manifest = json.loads(snapshot_path.read_text())
    run_record = json.loads((result_dir / "attention-delta.json").read_text())
    manifest_path = Path(run_record["manifest"]).expanduser().resolve()
    payloads = {role: json.loads((result_dir / f"{role}.json").read_text()) for role in ROLES}
    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    for case in manifest["cases"]:
        rows_by_group.setdefault(case["case_metadata"]["ensemble_group"], []).append(case)
    groups: dict[str, Any] = {}
    for group, rows in rows_by_group.items():
        targets = [row for row in rows if row["case_metadata"]["ensemble_role"] == "target"]
        controls = [row for row in rows if row["case_metadata"]["ensemble_role"] == "control"]
        if len(targets) != 1 or len(controls) < 2:
            raise ValueError(f"group {group!r} must contain one target and at least two controls")
        groups[group] = _group_summary(
            group=group,
            target_row=targets[0],
            control_rows=controls,
            payloads=payloads,
            manifest_path=manifest_path,
            output_dir=output_dir,
        )
    result = {
        "schema_version": 1,
        "definition": (
            "Within each checkpoint, separately L1-normalized target value-norm maps minus the mean of four "
            "separately normalized same-image alternate-instruction maps. Holo-minus-Qwen is the difference "
            "between those prompt-conditioned maps."
        ),
        "primary_method": "value_norm_attention",
        "manifest": str(manifest_path),
        "manifest_snapshot": str(snapshot_path),
        "groups": groups,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "native-saliency-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.result_dir, args.output_dir)
    compact = {}
    for group, values in result["groups"].items():
        primary = values["methods"]["value_norm_attention"]
        compact[group] = {
            "grid": values["image_grids"],
            "qwen_target_mass": primary["base"]["prompt_difference_target_region_mass"],
            "holo_target_mass": primary["tuned"]["prompt_difference_target_region_mass"],
            "holo_minus_qwen": primary["delta_prompt_difference_target_region_mass"],
            "qwen_min_loo_cosine": primary["base"]["minimum_leave_one_out_cosine"],
            "holo_min_loo_cosine": primary["tuned"]["minimum_leave_one_out_cosine"],
        }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
