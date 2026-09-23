#!/usr/bin/env python3
"""Compare Holo attention with guarded stochastic clicks for vscode_macos_0."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _map(case: dict[str, Any], method: str) -> np.ndarray:
    values = np.asarray(case["maps"][method][0], dtype=np.float64)
    if not math.isclose(float(values.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{case['id']} map is not normalized")
    return values


def _raw_overlay(values: np.ndarray, rows: int, columns: int) -> Image.Image:
    grid = values.reshape(rows, columns)
    ceiling = max(float(np.percentile(grid, 99.0)), 1e-12)
    normalized = np.clip(grid / ceiling, 0.0, 1.0) ** 0.55
    red = np.clip(normalized * 255, 0, 255)
    green = np.clip(np.maximum(normalized - 0.35, 0) / 0.65 * 210, 0, 210)
    blue = np.clip(np.maximum(normalized - 0.78, 0) / 0.22 * 80, 0, 80)
    alpha = np.clip(normalized * 205, 0, 205)
    rgba = np.stack((red, green, blue, alpha), axis=-1).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _signed_overlay(values: np.ndarray, rows: int, columns: int) -> Image.Image:
    grid = values.reshape(rows, columns)
    ceiling = max(float(np.percentile(np.abs(grid), 99.0)), 1e-12)
    normalized = np.clip(np.abs(grid) / ceiling, 0.0, 1.0) ** 0.55
    positive = grid >= 0
    red = np.where(positive, 255, 37) * normalized
    green = np.where(positive, 85, 138) * normalized
    blue = np.where(positive, 45, 230) * normalized
    alpha = np.clip(normalized * 205, 0, 205)
    rgba = np.stack((red, green, blue, alpha), axis=-1).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _annotate(
    source: Image.Image,
    overlay: Image.Image,
    oracle_bbox: list[float],
    sample_points: list[tuple[float, float]],
) -> Image.Image:
    result = Image.alpha_composite(source.convert("RGBA"), overlay.resize(source.size, Image.Resampling.BILINEAR))
    draw = ImageDraw.Draw(result)
    width = max(5, round(source.width / 450))
    bbox = tuple(round(value) for value in oracle_bbox)
    draw.rectangle(bbox, outline=(255, 138, 43, 255), width=width)
    radius = max(12, round(source.width / 180))
    for x, y in sample_points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 166, 255, 255))
    return result.convert("RGB")


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    result = image.copy()
    result.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "#0B1118")
    canvas.paste(result, ((width - result.width) // 2, (height - result.height) // 2))
    return canvas


def analyze(result_dir: Path, source_image: Path, sampling_summary: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads((result_dir / "manifest.snapshot.json").read_text())
    tuned = json.loads((result_dir / "tuned.json").read_text())
    cases = {case["id"]: case for case in tuned["cases"]}
    target_rows = [row for row in manifest["cases"] if row["case_metadata"]["ensemble_role"] == "target_sample"]
    control_rows = [row for row in manifest["cases"] if row["case_metadata"]["ensemble_role"] == "control"]
    target_ids = [row["id"] for row in target_rows]
    control_ids = [row["id"] for row in control_rows]
    if len(target_ids) != 8 or len(control_ids) != 4:
        raise ValueError("expected eight samples and four controls")

    method = "value_norm_attention"
    target_maps = np.stack([_map(cases[case_id], method) for case_id in target_ids])
    control_maps = np.stack([_map(cases[case_id], method) for case_id in control_ids])
    target_mean = target_maps.mean(axis=0)
    control_mean = control_maps.mean(axis=0)
    prompt_difference = target_mean - control_mean
    grid = cases[target_ids[0]]["image_grids"][0]
    rows, columns = grid
    target_indices = np.asarray(cases[target_ids[0]]["target"]["patch_indices"], dtype=np.int64)
    target_area_fraction = len(target_indices) / target_mean.size
    positive = np.clip(prompt_difference, 0.0, None)
    positive_total = float(positive.sum())
    raw_target_mass = float(target_mean[target_indices].sum())
    control_target_mass = float(control_mean[target_indices].sum())
    difference_target_mass = float(prompt_difference[target_indices].sum())
    raw_target_rank = int(np.count_nonzero(target_mean > target_mean[target_indices[0]]) + 1)
    difference_target_rank = int(
        np.count_nonzero(prompt_difference > prompt_difference[target_indices[0]]) + 1
    )
    leave_one_out = []
    for index, control_id in enumerate(control_ids):
        reduced_difference = target_mean - np.delete(control_maps, index, axis=0).mean(axis=0)
        denominator = float(np.linalg.norm(prompt_difference) * np.linalg.norm(reduced_difference))
        leave_one_out.append(
            {
                "excluded_control": control_id,
                "target_mass": float(reduced_difference[target_indices].sum()),
                "cosine_similarity": (
                    float(np.dot(prompt_difference, reduced_difference) / denominator) if denominator else 0.0
                ),
            }
        )

    sampling = json.loads(sampling_summary.read_text())
    sample = next(row for row in sampling["results"] if row["sample_id"] == "vscode_macos_0")
    sample_points = [(row["source_pixel"]["x"], row["source_pixel"]["y"]) for row in sample["candidates"]]
    oracle_bbox = sample["oracle_bbox_source_pixels"]
    with Image.open(source_image) as opened:
        source = opened.convert("RGB")

    raw_image = _annotate(source, _raw_overlay(target_mean, rows, columns), oracle_bbox, sample_points)
    difference_image = _annotate(source, _signed_overlay(prompt_difference, rows, columns), oracle_bbox, sample_points)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "vscode-refresh-mean-value-norm-attention.png"
    difference_path = output_dir / "vscode-refresh-diverse-instruction-subtracted-value-norm.png"
    raw_image.save(raw_path, quality=94)
    difference_image.save(difference_path, quality=94)

    canvas = Image.new("RGB", (2560, 1080), "#F5F2EC")
    draw = ImageDraw.Draw(canvas)
    draw.text((80, 45), "Refresh the file explorer · attention versus sampled actions", fill="#131A22", font=_font(44, bold=True))
    draw.text((80, 104), "25% linear resolution · Holo-3.1-4B · value-norm attention over all 8 sampled coordinate strings", fill="#617080", font=_font(24))
    left = _fit(raw_image, 1160, 754)
    right = _fit(difference_image, 1160, 754)
    canvas.paste(left, (80, 180))
    canvas.paste(right, (1320, 180))
    draw.text((80, 145), "MEAN VALUE-NORM ATTENTION", fill="#1680D8", font=_font(22, bold=True))
    draw.text((1320, 145), "MINUS DIVERSE-INSTRUCTION BASELINE", fill="#E25A3C", font=_font(22, bold=True))
    metrics = (
        f"Oracle patch: {raw_target_mass * 100:.2f}% attention · "
        f"{raw_target_mass / target_area_fraction:.1f}× area lift · rank {raw_target_rank}/{target_mean.size} · "
        f"control mean {control_target_mass * 100:.2f}% · sampled hits 0/8"
    )
    draw.text((80, 955), metrics, fill="#131A22", font=_font(23, bold=True))
    legend = (
        "Orange = oracle refresh icon   ·   cyan = 8 sampled clicks   ·   red = instruction-specific positive attribution   ·   blue = baseline-dominant"
    )
    draw.text((80, 1000), legend, fill="#43505D", font=_font(22))
    comparison_path = output_dir / "vscode-refresh-attention-vs-sampling.png"
    canvas.save(comparison_path, quality=94)

    peak_index = int(np.argmax(target_mean))
    diff_peak_index = int(np.argmax(prompt_difference))
    summary = {
        "schema_version": 1,
        "case": "vscode_macos_0",
        "instruction": "Refresh the file explorer.",
        "model": tuned["model"],
        "resolution": [640, 416],
        "grid": grid,
        "method": "mean value-norm attention over eight sampled action strings",
        "baseline": "mean of four same-image alternate-instruction value-norm maps",
        "sample_count": len(target_ids),
        "control_count": len(control_ids),
        "sampling_hit_count": sample["spread"]["hit_count"],
        "sampling_rms_radius_normalized": sample["spread"]["rms_radius"],
        "target_patch_indices": target_indices.tolist(),
        "target_patch_area_fraction": target_area_fraction,
        "raw_target_mass": raw_target_mass,
        "raw_target_lift": float(raw_target_mass / target_area_fraction),
        "raw_target_rank": raw_target_rank,
        "raw_target_mass_by_sample": [float(row[target_indices].sum()) for row in target_maps],
        "control_mean_target_mass": control_target_mass,
        "control_target_mass_by_instruction": {
            control_id: float(control_maps[index, target_indices].sum())
            for index, control_id in enumerate(control_ids)
        },
        "prompt_difference_target_mass": difference_target_mass,
        "prompt_difference_target_rank": difference_target_rank,
        "target_share_of_positive_prompt_difference": (
            float(positive[target_indices].sum()) / positive_total if positive_total else 0.0
        ),
        "leave_one_control_out": leave_one_out,
        "minimum_leave_one_out_cosine": min(row["cosine_similarity"] for row in leave_one_out),
        "raw_peak_patch_index": peak_index,
        "raw_peak_inside_target": peak_index in set(target_indices.tolist()),
        "prompt_difference_peak_patch_index": diff_peak_index,
        "prompt_difference_peak_inside_target": diff_peak_index in set(target_indices.tolist()),
        "rendered": {
            "raw": str(raw_path),
            "prompt_difference": str(difference_path),
            "comparison": str(comparison_path),
        },
        "claim_boundary": "Single selected miss at 25% linear resolution; descriptive, not causal or population-level.",
    }
    summary_path = output_dir / "vscode-refresh-attention-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--sampling-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.result_dir, args.source_image, args.sampling_summary, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
