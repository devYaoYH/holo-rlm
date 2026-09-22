"""Analyze the x/y decomposition of the PowerPoint-48 prompt-control estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _overlap_ids(bbox: list[int], size: list[int], grid: list[int]) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox
    width, height = size
    rows, columns = grid
    row_ids = np.asarray([r for r in range(rows) if (r + 1) * height / rows > y1 and r * height / rows < y2])
    col_ids = np.asarray([c for c in range(columns) if (c + 1) * width / columns > x1 and c * width / columns < x2])
    return row_ids, col_ids


def _signed_overlay(values: np.ndarray, ceiling: float) -> Image.Image:
    normalized = np.clip(np.abs(values) / max(ceiling, 1e-12), 0.0, 1.0) ** 0.55
    positive = values >= 0
    red = np.where(positive, 240, 45) * normalized
    green = np.where(positive, 90, 132) * normalized
    blue = np.where(positive, 57, 211) * normalized
    alpha = np.clip(normalized * 196, 0, 196)
    return Image.fromarray(np.stack((red, green, blue, alpha), axis=-1).astype(np.uint8), mode="RGBA")


def _composite(source: Image.Image, values: np.ndarray, bbox: list[int], ceiling: float) -> Image.Image:
    overlay = _signed_overlay(values, ceiling).resize(source.size, Image.Resampling.BILINEAR)
    result = Image.alpha_composite(source.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(result)
    width = max(5, round(source.width / 480))
    draw.rectangle(tuple(bbox), outline=(255, 255, 255, 255), width=width)
    draw.rectangle(
        (bbox[0] - width, bbox[1] - width, bbox[2] + width, bbox[3] + width),
        outline=(24, 168, 117, 255),
        width=width,
    )
    return result.convert("RGB")


def analyze(result_dir: Path, manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    result_dir = result_dir.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    payload = json.loads((result_dir / "tuned.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    cases = {row["id"]: row for row in payload["cases"]}
    rows = manifest["cases"]
    target_rows = [row for row in rows if row["case_metadata"]["ensemble_role"] == "target"]
    control_rows = [row for row in rows if row["case_metadata"]["ensemble_role"] == "control"]
    if len(target_rows) != 2:
        raise ValueError("expected one x target and one y target")
    source = Image.open(
        Path(__file__).resolve().parents[1] / "data/screenspot-pro/images/powerpoint_windows_48.png"
    ).convert("RGB")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for axis in ("x", "y"):
        target_row = next(row for row in target_rows if row["case_metadata"]["attributed_field"] == axis)
        controls = [row for row in control_rows if row["case_metadata"]["attributed_field"] == axis]
        target_case = cases[target_row["id"]]
        target = np.asarray(target_case["maps"]["value_norm_attention"][0], dtype=np.float64)
        control_maps = np.stack(
            [np.asarray(cases[row["id"]]["maps"]["value_norm_attention"][0], dtype=np.float64) for row in controls]
        )
        grid = target_case["image_grids"][0]
        size = target_case["image_sizes"][0]
        difference = (target - control_maps.mean(axis=0)).reshape(grid)
        positive = np.maximum(difference, 0)
        positive_mass = float(positive.sum())
        bbox = target_row["case_metadata"]["bbox"]
        row_ids, col_ids = _overlap_ids(bbox, size, grid)
        target_mass = float(positive[np.ix_(row_ids, col_ids)].sum())
        column_mass = float(positive[:, col_ids].sum())
        row_mass = float(positive[row_ids, :].sum())
        ceiling = float(np.percentile(np.abs(difference), 99.5))
        path = output_dir / f"powerpoint48-holo-value-norm-{axis}-difference.png"
        _composite(source, difference, bbox, ceiling).save(path, quality=94)
        results[axis] = {
            "positive_mass": positive_mass,
            "target_box_share_of_positive_mass": target_mass / positive_mass,
            "target_column_share_of_positive_mass": column_mass / positive_mass,
            "target_row_share_of_positive_mass": row_mass / positive_mass,
            "rendered": str(path),
        }
    result = {
        "schema_version": 1,
        "definition": "Target instruction minus the mean of four same-image controls, split by scored coordinate field.",
        "interpretation": (
            "A dominant x-column in the x-token map and y-row in the y-token map indicates axis-factorized "
            "coordinate routing. Averaging those queries creates a cross/stripe visualization and should not be "
            "interpreted as object-wide residual activation."
        ),
        "axes": results,
    }
    (output_dir / "axis-diagnostic-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.result_dir, args.manifest, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
