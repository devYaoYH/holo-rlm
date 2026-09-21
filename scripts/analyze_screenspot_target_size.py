#!/usr/bin/env python3
"""Quantify target-box size as a confound in ScreenSpot-Pro UI-type accuracy."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("summary", type=Path, help="completed benchmark summary.json")
    result.add_argument("annotations", type=Path, help="directory of official annotation JSON files")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--bins", type=int, default=10, help="pooled target-area rank bins")
    result.add_argument("--annotation-source", help="stable URL or revision label recorded in the output")
    return result


def accuracy(rows: list[dict[str, Any]]) -> float:
    return sum(bool(row["strict_correct"]) for row in rows) / len(rows)


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.bins < 2:
        raise SystemExit("--bins must be at least 2")

    summary = json.loads(args.summary.read_text())
    runs = summary.get("items")
    if not isinstance(runs, list) or not runs:
        raise SystemExit("summary must contain a non-empty items list")

    annotations: dict[str, dict[str, Any]] = {}
    for path in sorted(args.annotations.glob("*.json")):
        for row in json.loads(path.read_text()):
            sample_id = str(row["id"])
            if sample_id in annotations:
                raise SystemExit(f"duplicate annotation id: {sample_id}")
            annotations[sample_id] = row

    joined: list[dict[str, Any]] = []
    for run in runs:
        sample_id = str(run["sample_id"])
        annotation = annotations.get(sample_id)
        if annotation is None:
            raise SystemExit(f"missing annotation: {sample_id}")
        if annotation["ui_type"] != run["ui_type"]:
            raise SystemExit(f"ui_type mismatch: {sample_id}")
        x1, y1, x2, y2 = (float(value) for value in annotation["bbox"])
        image_width, image_height = (float(value) for value in annotation["img_size"])
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
            raise SystemExit(f"invalid geometry: {sample_id}")
        joined.append(
            {
                **run,
                "bbox_width_px": width,
                "bbox_height_px": height,
                "bbox_area_px": width * height,
                "bbox_area_fraction": width * height / (image_width * image_height),
                "bbox_width_fraction": width / image_width,
                "bbox_height_fraction": height / image_height,
            }
        )

    ranked = sorted(joined, key=lambda row: (row["bbox_area_fraction"], row["sample_id"]))
    for rank, row in enumerate(ranked):
        row["area_rank_bin"] = min(args.bins - 1, math.floor(rank * args.bins / len(ranked)))

    by_ui: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_ui_bin: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    bin_counts: dict[int, int] = defaultdict(int)
    for row in joined:
        ui_type = str(row["ui_type"])
        size_bin = int(row["area_rank_bin"])
        by_ui[ui_type].append(row)
        by_ui_bin[(ui_type, size_bin)].append(row)
        bin_counts[size_bin] += 1

    ui_summary: dict[str, Any] = {}
    for ui_type, rows in sorted(by_ui.items()):
        ui_summary[ui_type] = {
            "count": len(rows),
            "accuracy": accuracy(rows),
            "median_bbox_width_px": statistics.median(row["bbox_width_px"] for row in rows),
            "median_bbox_height_px": statistics.median(row["bbox_height_px"] for row in rows),
            "median_bbox_area_px": statistics.median(row["bbox_area_px"] for row in rows),
            "median_bbox_area_fraction": statistics.median(row["bbox_area_fraction"] for row in rows),
            "median_bbox_width_fraction": statistics.median(row["bbox_width_fraction"] for row in rows),
            "median_bbox_height_fraction": statistics.median(row["bbox_height_fraction"] for row in rows),
        }

    area_buckets = [
        ("<0.01%", 0.0, 0.0001),
        ("0.01–0.02%", 0.0001, 0.0002),
        ("0.02–0.05%", 0.0002, 0.0005),
        ("0.05–0.10%", 0.0005, 0.001),
        ("0.10–0.20%", 0.001, 0.002),
        ("0.20–0.50%", 0.002, 0.005),
        ("≥0.50%", 0.005, None),
    ]
    distribution_buckets: list[dict[str, Any]] = []
    for label, lower, upper in area_buckets:
        bucket: dict[str, Any] = {
            "label": label,
            "lower_area_fraction_inclusive": lower,
            "upper_area_fraction_exclusive": upper,
        }
        for ui_type, rows in sorted(by_ui.items()):
            count = sum(
                1
                for row in rows
                if row["bbox_area_fraction"] >= lower
                and (upper is None or row["bbox_area_fraction"] < upper)
            )
            bucket[ui_type] = {"count": count, "share": count / len(rows)}
        distribution_buckets.append(bucket)

    strata: list[dict[str, Any]] = []
    standardized: dict[str, float] = {ui_type: 0.0 for ui_type in by_ui}
    for size_bin in range(args.bins):
        weight = bin_counts[size_bin] / len(joined)
        row: dict[str, Any] = {
            "area_rank_bin": size_bin + 1,
            "pooled_count": bin_counts[size_bin],
            "pooled_weight": weight,
        }
        for ui_type in sorted(by_ui):
            cells = by_ui_bin[(ui_type, size_bin)]
            if not cells:
                raise SystemExit(f"empty size stratum for {ui_type}: {size_bin + 1}")
            value = accuracy(cells)
            row[ui_type] = {"count": len(cells), "accuracy": value}
            standardized[ui_type] += weight * value
        strata.append(row)

    icon = ui_summary["icon"]
    text = ui_summary["text"]
    result = {
        "schema_version": 1,
        "source": {
            "summary": str(args.summary),
            "annotations": args.annotation_source or str(args.annotations),
            "joined_count": len(joined),
        },
        "definition": "UI type is the official target-element label. Target size is the annotated bounding-box area divided by screenshot area.",
        "by_ui_type": ui_summary,
        "median_area_ratio_text_over_icon": text["median_bbox_area_fraction"] / icon["median_bbox_area_fraction"],
        "raw_accuracy_gap_text_minus_icon": text["accuracy"] - icon["accuracy"],
        "target_area_distribution": {
            "unit": "share within each official UI type",
            "buckets": distribution_buckets,
        },
        "size_standardization": {
            "method": f"Direct standardization over {args.bins} pooled target-area rank bins",
            "accuracy": standardized,
            "gap_text_minus_icon": standardized["text"] - standardized["icon"],
            "strata": strata,
            "caveat": "This adjusts only for annotated target area. Application, platform, action family, visual density, aspect ratio, and annotation noise remain possible confounds.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
