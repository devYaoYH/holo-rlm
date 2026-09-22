#!/usr/bin/env python3
"""Create a compact, slide-ready analysis of a ScreenSpot resolution ablation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> list[float] | None:
    if count == 0:
        return None
    proportion = successes / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    return [center - half_width, center + half_width]


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _cohort(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row["strict_correct"]) for row in rows)
    return {
        "count": len(rows),
        "strict_correct": successes,
        "retention": successes / len(rows) if rows else None,
        "wilson_95": wilson_interval(successes, len(rows)),
    }


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload["results"]
    by_scale: dict[float, list[dict[str, Any]]] = defaultdict(list)
    by_sample: dict[str, dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in results:
        scale = float(row["linear_scale"])
        by_scale[scale].append(row)
        by_sample[str(row["sample_id"])][scale] = row

    scale_rows: list[dict[str, Any]] = []
    for scale in sorted(by_scale, reverse=True):
        rows = by_scale[scale]
        drifts = [float(row["drift_from_native_normalized"]) for row in rows if row.get("drift_from_native_normalized") is not None]
        distances = [float(row["point_to_box_distance_normalized"]) for row in rows if row.get("point_to_box_distance_normalized") is not None]
        areas = {
            True: [],
            False: [],
        }
        for row in rows:
            x1, y1, x2, y2 = (float(value) for value in row["bbox"])
            width, height = (int(value) for value in row["native_image_size"])
            areas[bool(row["strict_correct"])].append(((x2 - x1) * (y2 - y1)) / (width * height))
        applications = {
            key: _cohort([row for row in rows if row["application"] == key])
            for key in sorted({str(row["application"]) for row in rows})
        }
        ui_types = {
            key: _cohort([row for row in rows if row["ui_type"] == key])
            for key in sorted({str(row["ui_type"]) for row in rows})
        }
        scale_rows.append(
            {
                "linear_scale": scale,
                "area_fraction": scale * scale,
                **_cohort(rows),
                "format_valid": sum(bool(row["format_valid"]) for row in rows),
                "median_drift_from_native_normalized": statistics.median(drifts) if drifts else None,
                "p90_drift_from_native_normalized": quantile(drifts, 0.9),
                "median_point_to_box_distance_normalized": statistics.median(distances) if distances else None,
                "p90_point_to_box_distance_normalized": quantile(distances, 0.9),
                "median_target_area_retained": statistics.median(areas[True]) if areas[True] else None,
                "median_target_area_failed": statistics.median(areas[False]) if areas[False] else None,
                "by_application": applications,
                "by_ui_type": ui_types,
            }
        )

    ordered_scales = sorted(by_scale, reverse=True)
    monotone = 0
    for rows in by_sample.values():
        outcomes = [bool(rows[scale]["strict_correct"]) for scale in ordered_scales]
        monotone += all(not outcomes[index] or outcomes[index - 1] for index in range(1, len(outcomes)))

    return {
        "schema_version": 1,
        "experiment": payload["experiment"],
        "model_id": payload["model_id"],
        "model_revision": payload.get("server_model", {}).get("revision"),
        "inference_configuration": payload.get("server_model", {}).get("inference_configuration"),
        "runtime_seconds": payload["runtime_seconds"],
        "selected_count": payload["selected_count"],
        "request_count": payload["request_count"],
        "selection_condition": "native successes from the reference run; native rerun was 36/36",
        "estimand": "paired success retention under controlled downsampling, not unconditional benchmark accuracy",
        "monotone_items": monotone,
        "nonmonotone_items": len(by_sample) - monotone,
        "scales": scale_rows,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("summary", type=Path)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    result = analyze(json.loads(args.summary.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
