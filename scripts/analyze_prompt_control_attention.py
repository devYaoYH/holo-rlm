"""Aggregate and render paired same-image instruction-control attention maps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def _signed_overlay(values: np.ndarray, ceiling: float) -> Image.Image:
    normalized = np.clip(np.abs(values) / max(ceiling, 1e-12), 0.0, 1.0) ** 0.55
    positive = values >= 0
    red = np.where(positive, 240, 45) * normalized
    green = np.where(positive, 90, 132) * normalized
    blue = np.where(positive, 57, 211) * normalized
    alpha = np.clip(normalized * 196, 0, 196)
    rgba = np.stack((red, green, blue, alpha), axis=-1).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _composite(source: Image.Image, heatmap: Image.Image, bbox: list[float]) -> Image.Image:
    resized = heatmap.resize(source.size, Image.Resampling.BILINEAR)
    result = Image.alpha_composite(source.convert("RGBA"), resized)
    draw = ImageDraw.Draw(result)
    width = max(5, round(source.width / 480))
    box = tuple(round(value) for value in bbox)
    draw.rectangle(box, outline=(255, 255, 255, 255), width=width)
    draw.rectangle(
        (box[0] - width, box[1] - width, box[2] + width, box[3] + width),
        outline=(24, 168, 117, 255),
        width=width,
    )
    return result.convert("RGB")


def _case_map(case: dict[str, Any], method: str) -> np.ndarray:
    frames = case["maps"][method]
    if len(frames) != 1:
        raise ValueError(f"case {case['id']!r} must contain exactly one image")
    values = np.asarray(frames[0], dtype=np.float64)
    if not math.isclose(float(values.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"case {case['id']!r} {method} map is not L1-normalized")
    return values


def _role_summary(
    cases: dict[str, dict[str, Any]],
    target_id: str,
    control_ids: list[str],
    method: str,
    target_indices: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    target = _case_map(cases[target_id], method)
    controls = np.stack([_case_map(cases[case_id], method) for case_id in control_ids])
    control_mean = controls.mean(axis=0)
    prompt_difference = target - control_mean
    leave_one_out = []
    for index, control_id in enumerate(control_ids):
        reduced_mean = np.delete(controls, index, axis=0).mean(axis=0)
        reduced_difference = target - reduced_mean
        leave_one_out.append(
            {
                "excluded_control": control_id,
                "cosine_similarity": _cosine(prompt_difference, reduced_difference),
            }
        )
    instruction_target_mass = {
        target_id: float(target[target_indices].sum()),
        **{
            case_id: float(controls[index, target_indices].sum())
            for index, case_id in enumerate(control_ids)
        },
    }
    positive = np.clip(prompt_difference, 0.0, None)
    positive_mass = float(positive.sum())
    target_positive_mass = float(positive[target_indices].sum())
    summary = {
        "target_region_mass_by_instruction": instruction_target_mass,
        "control_mean_target_region_mass": float(control_mean[target_indices].sum()),
        "prompt_difference_target_region_mass": float(prompt_difference[target_indices].sum()),
        "prompt_difference_positive_mass": positive_mass,
        "target_share_of_positive_prompt_difference": (
            target_positive_mass / positive_mass if positive_mass else 0.0
        ),
        "leave_one_control_out": leave_one_out,
        "minimum_leave_one_out_cosine": min(row["cosine_similarity"] for row in leave_one_out),
        "map": prompt_difference.tolist(),
    }
    return summary, prompt_difference


def analyze(result_dir: Path, source_image: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads((result_dir / "manifest.snapshot.json").read_text())
    target_rows = [case for case in manifest["cases"] if case["case_metadata"]["ensemble_role"] == "target"]
    control_rows = [case for case in manifest["cases"] if case["case_metadata"]["ensemble_role"] == "control"]
    if len(target_rows) != 1 or len(control_rows) < 2:
        raise ValueError("manifest must contain one target and at least two controls")
    target_row = target_rows[0]
    target_id = target_row["id"]
    control_ids = [case["id"] for case in control_rows]
    model_payloads = {
        role: json.loads((result_dir / f"{role}.json").read_text())
        for role in ("base", "tuned")
    }
    cases_by_role = {
        role: {case["id"]: case for case in payload["cases"]}
        for role, payload in model_payloads.items()
    }
    target_case = cases_by_role["base"][target_id]
    grid = target_case["image_grids"][0]
    target_indices = np.asarray(target_case["target"]["patch_indices"], dtype=np.int64)

    methods: dict[str, Any] = {}
    rendered: dict[str, dict[str, str]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_image) as opened:
        source = opened.convert("RGB")
    for method in ("value_norm_attention", "direct_attention"):
        role_summaries: dict[str, Any] = {}
        role_maps: dict[str, np.ndarray] = {}
        for role in ("base", "tuned"):
            role_summaries[role], role_maps[role] = _role_summary(
                cases_by_role[role], target_id, control_ids, method, target_indices
            )
        delta = role_maps["tuned"] - role_maps["base"]
        method_summary = {
            **role_summaries,
            "delta_prompt_difference_target_region_mass": float(delta[target_indices].sum()),
            "delta_map": delta.tolist(),
        }
        methods[method] = method_summary

        rows, columns = grid
        base_grid = role_maps["base"].reshape(rows, columns)
        tuned_grid = role_maps["tuned"].reshape(rows, columns)
        delta_grid = delta.reshape(rows, columns)
        shared_ceiling = float(
            np.percentile(np.abs(np.concatenate((base_grid.ravel(), tuned_grid.ravel()))), 99.5)
        )
        delta_ceiling = float(np.percentile(np.abs(delta_grid), 99.5))
        slug = method.replace("_attention", "").replace("_", "-")
        paths = {
            "base": output_dir / f"prompt-difference-qwen-{slug}.png",
            "tuned": output_dir / f"prompt-difference-holo-{slug}.png",
            "delta": output_dir / f"prompt-difference-holo-minus-qwen-{slug}.png",
        }
        bbox = target_row["case_metadata"]["bbox"]
        _composite(source, _signed_overlay(base_grid, shared_ceiling), bbox).save(paths["base"], quality=94)
        _composite(source, _signed_overlay(tuned_grid, shared_ceiling), bbox).save(paths["tuned"], quality=94)
        _composite(source, _signed_overlay(delta_grid, delta_ceiling), bbox).save(paths["delta"], quality=94)
        rendered[method] = {name: str(path) for name, path in paths.items()}

    result = {
        "schema_version": 1,
        "definition": (
            "For each checkpoint: L1-normalized target map minus the mean of four separately "
            "L1-normalized same-image alternate-instruction maps. Delta is Holo minus Qwen."
        ),
        "primary_method": "value_norm_attention",
        "target_case": target_id,
        "control_cases": control_ids,
        "grid": grid,
        "target_bbox": target_row["case_metadata"]["bbox"],
        "target_patch_indices": target_indices.tolist(),
        "methods": methods,
        "rendered": rendered,
    }
    (output_dir / "paired-control-attention-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.result_dir, args.source_image, args.output_dir)
    compact = {
        method: {
            "qwen_prompt_difference_target_mass": values["base"]["prompt_difference_target_region_mass"],
            "holo_prompt_difference_target_mass": values["tuned"]["prompt_difference_target_region_mass"],
            "holo_minus_qwen": values["delta_prompt_difference_target_region_mass"],
            "qwen_min_loo_cosine": values["base"]["minimum_leave_one_out_cosine"],
            "holo_min_loo_cosine": values["tuned"]["minimum_leave_one_out_cosine"],
        }
        for method, values in summary["methods"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
