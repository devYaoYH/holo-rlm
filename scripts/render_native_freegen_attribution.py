"""Render x-token, y-token, and joint views for native ScreenSpot free generations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _positive_overlay(values: np.ndarray, ceiling: float) -> Image.Image:
    normalized = np.clip(values / max(ceiling, 1e-12), 0.0, 1.0) ** 0.55
    red = 245 * normalized + 38 * (1 - normalized)
    green = 92 * normalized + 90 * (1 - normalized)
    blue = 38 * normalized + 130 * (1 - normalized)
    alpha = np.clip(normalized * 188, 0, 188)
    return Image.fromarray(np.stack((red, green, blue, alpha), axis=-1).astype(np.uint8), mode="RGBA")


def _patch_indices(bbox: list[int], size: list[int], grid: list[int]) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox
    width, height = size
    rows, columns = grid
    row_ids = np.asarray(
        [row for row in range(rows) if (row + 1) * height / rows > y1 and row * height / rows < y2]
    )
    column_ids = np.asarray(
        [column for column in range(columns) if (column + 1) * width / columns > x1 and column * width / columns < x2]
    )
    return row_ids, column_ids


def _composite(
    source: Image.Image,
    heatmap: np.ndarray,
    ceiling: float,
    bbox: list[int],
    click: tuple[float, float],
) -> Image.Image:
    overlay = _positive_overlay(heatmap, ceiling).resize(source.size, Image.Resampling.BILINEAR)
    result = Image.alpha_composite(source.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(result)
    width = max(5, round(source.width / 480))
    draw.rectangle(tuple(bbox), outline=(255, 255, 255, 255), width=width)
    draw.rectangle(
        (bbox[0] - width, bbox[1] - width, bbox[2] + width, bbox[3] + width),
        outline=(24, 168, 117, 255),
        width=width,
    )
    x, y = click
    radius = max(14, round(source.width / 115))
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 255, 255, 255), width=width)
    draw.line((x - radius * 1.4, y, x + radius * 1.4, y), fill=(255, 255, 255, 255), width=width)
    draw.line((x, y - radius * 1.4, x, y + radius * 1.4), fill=(255, 255, 255, 255), width=width)
    return result.convert("RGB")


def render(result_dir: Path, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    result_dir = result_dir.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    tuned = json.loads((result_dir / "tuned.json").read_text())
    cases = {row["id"]: row for row in tuned["cases"]}
    metadata = {row["id"]: row["case_metadata"] for row in manifest["cases"]}
    sample_ids = sorted({row["sample_id"] for row in metadata.values()})
    output_dir.mkdir(parents=True, exist_ok=True)
    samples: dict[str, Any] = {}
    for sample_id in sample_ids:
        x_case = cases[f"{sample_id}_x"]
        y_case = cases[f"{sample_id}_y"]
        meta = metadata[f"{sample_id}_x"]
        grid = x_case["image_grids"][0]
        size = x_case["image_sizes"][0]
        x_map = np.asarray(x_case["maps"]["value_norm_attention"][0], dtype=np.float64).reshape(grid)
        y_map = np.asarray(y_case["maps"]["value_norm_attention"][0], dtype=np.float64).reshape(grid)
        joint = np.sqrt(np.maximum(x_map, 0) * np.maximum(y_map, 0))
        joint /= joint.sum()
        row_ids, column_ids = _patch_indices(meta["bbox"], size, grid)
        target_indices = np.ix_(row_ids, column_ids)
        x_band_mass = float(x_map[:, column_ids].sum())
        y_band_mass = float(y_map[row_ids, :].sum())
        target_mass = float(joint[target_indices].sum())
        target_area = len(row_ids) * len(column_ids) / (grid[0] * grid[1])
        coordinate = meta["free_generated_coordinate"]
        click = (coordinate[0] * size[0] / 1000, coordinate[1] * size[1] / 1000)
        source_path = output_dir / "source" / f"{sample_id}.png"
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        maps = {"x-token": x_map, "y-token": y_map, "joint-xy": joint}
        shared_axis_ceiling = float(np.percentile(np.concatenate((x_map.ravel(), y_map.ravel())), 99.5))
        for name, values in maps.items():
            ceiling = float(np.percentile(values, 99.5)) if name == "joint-xy" else shared_axis_ceiling
            _composite(source, values, ceiling, meta["bbox"], click).save(
                sample_dir / f"holo-value-norm-{name}.png", quality=94
            )
        samples[sample_id] = {
            "instruction": meta["instruction"],
            "strict_correct": meta["strict_correct"],
            "bbox": meta["bbox"],
            "free_generated_coordinate": coordinate,
            "free_generated_click_pixels": list(click),
            "image_size": size,
            "grid": grid,
            "x_token_mass_in_target_columns": x_band_mass,
            "y_token_mass_in_target_rows": y_band_mass,
            "joint_xy_target_mass": target_mass,
            "joint_xy_target_lift": target_mass / target_area,
            "rendered": {
                name: str(sample_dir / f"holo-value-norm-{name}.png") for name in maps
            },
        }
    result = {
        "schema_version": 1,
        "definition": (
            "Value-norm attention reconstructed over Holo's own free-generated coordinate tokens. "
            "x and y are shown separately; joint-xy is the normalized geometric mean used only as a "
            "two-axis visualization."
        ),
        "resolution": "2880x1800 source; 16777216 processor pixel ceiling; no downsampling",
        "samples": samples,
    }
    (output_dir / "native-freegen-attribution-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.result_dir, args.manifest, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
