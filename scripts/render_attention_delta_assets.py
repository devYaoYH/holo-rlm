"""Render paired Qwen/Holo attention maps for the presentation evidence slide."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _reshape(values: list[float], grid: list[int]) -> np.ndarray:
    rows, columns = grid
    array = np.asarray(values, dtype=np.float32)
    if array.size != rows * columns:
        raise ValueError(f"map has {array.size} values but grid is {rows}x{columns}")
    return array.reshape(rows, columns)


def _positive_overlay(values: np.ndarray, ceiling: float) -> Image.Image:
    normalized = np.clip(values / max(ceiling, 1e-12), 0.0, 1.0) ** 0.55
    red = 245 * normalized + 38 * (1 - normalized)
    green = 92 * normalized + 90 * (1 - normalized)
    blue = 38 * normalized + 130 * (1 - normalized)
    alpha = np.clip(normalized * 178, 0, 178)
    rgba = np.stack((red, green, blue, alpha), axis=-1).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


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


def render(
    screen_result: Path,
    hotel_result: Path,
    source_image: Path,
    output_dir: Path,
    method_name: str,
) -> dict:
    screen = json.loads(screen_result.read_text())["comparison"]["cases"][0]
    if screen["id"] != "powerpoint_windows_59":
        raise ValueError("the ScreenSpot result is not powerpoint_windows_59")
    if method_name not in screen["methods"]:
        raise ValueError(f"unknown attribution method: {method_name}")
    method = screen["methods"][method_name]
    grid = screen["image_grids"][0]
    base = _reshape(method["base"][0], grid)
    tuned = _reshape(method["tuned"][0], grid)
    delta = _reshape(method["delta"][0], grid)
    shared_ceiling = float(np.percentile(np.concatenate((base.ravel(), tuned.ravel())), 99.5))
    delta_ceiling = float(np.percentile(np.abs(delta), 99.5))

    bbox = json.loads((screen_result.parent / "base.json").read_text())["cases"][0]["target"]["bbox_pixels"]
    with Image.open(source_image) as opened:
        source = opened.convert("RGB")
    output_dir.mkdir(parents=True, exist_ok=True)
    method_slug = method_name.replace("_attention", "").replace("_", "-")
    files = {
        "base": output_dir / f"attention-qwen-{method_slug}.png",
        "tuned": output_dir / f"attention-holo-{method_slug}.png",
        "delta": output_dir / f"attention-holo-minus-qwen-{method_slug}.png",
    }
    _composite(source, _positive_overlay(base, shared_ceiling), bbox).save(files["base"], quality=94)
    _composite(source, _positive_overlay(tuned, shared_ceiling), bbox).save(files["tuned"], quality=94)
    _composite(source, _signed_overlay(delta, delta_ceiling), bbox).save(files["delta"], quality=94)

    hotel = json.loads(hotel_result.read_text())["comparison"]["cases"][0]
    hotel_methods = {}
    for name, values in hotel["methods"].items():
        hotel_methods[name] = {
            "base_frame_mass": [sum(frame) for frame in values["base"]],
            "tuned_frame_mass": [sum(frame) for frame in values["tuned"]],
            "delta_frame_mass": [sum(frame) for frame in values["delta"]],
        }
    summary = {
        "schema_version": 1,
        "screen_case": screen["id"],
        "screen_grid": grid,
        "screen_target_delta": screen["target_delta"],
        "hotel_case": hotel["id"],
        "hotel_frame_allocation": hotel_methods,
        "rendering": {
            "method": method_name,
            "base_and_tuned_shared_percentile": 99.5,
            "delta_symmetric_absolute_percentile": 99.5,
            "target_box": bbox,
            "positive_delta_color": "orange-red",
            "negative_delta_color": "blue",
        },
        "files": {name: str(path) for name, path in files.items()},
    }
    (output_dir / "attention-delta-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-result", type=Path, required=True)
    parser.add_argument("--hotel-result", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("direct_attention", "value_norm_attention"),
        default="value_norm_attention",
    )
    args = parser.parse_args()
    summary = render(
        args.screen_result,
        args.hotel_result,
        args.source_image,
        args.output_dir,
        args.method,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
