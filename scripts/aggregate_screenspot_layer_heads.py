#!/usr/bin/env python3
"""Aggregate ScreenSpot prompt-contrast layer/head statistics across a frozen cohort."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("manifest", type=Path, help="Frozen cohort JSON")
    result.add_argument("--contrasts-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--focus-layer", type=int, default=19)
    result.add_argument("--focus-head", type=int, default=10)
    result.add_argument("--bootstrap-samples", type=int, default=10_000)
    result.add_argument("--seed", type=int, default=19)
    return result


def _mean(values: Iterable[float]) -> float:
    return float(statistics.fmean(values))


def _summary(values: list[float], *, bootstrap_samples: int, rng: random.Random) -> dict[str, float]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("metric values must be a non-empty finite list")
    boot = sorted(
        _mean(rng.choice(values) for _ in values)
        for _ in range(bootstrap_samples)
    )
    lower = boot[round(0.025 * (len(boot) - 1))]
    upper = boot[round(0.975 * (len(boot) - 1))]
    return {
        "count": len(values),
        "mean": _mean(values),
        "median": float(statistics.median(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "bootstrap_mean_ci95_lower": lower,
        "bootstrap_mean_ci95_upper": upper,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ranked(rows: list[dict[str, Any]], metric: str = "mean") -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (-float(row[metric]), row.get("transformer_layer", -1), row.get("head", -1)))


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit("manifest must contain a non-empty items list")
    if args.bootstrap_samples < 100:
        raise SystemExit("--bootstrap-samples must be at least 100")

    rng = random.Random(args.seed)
    per_item: list[dict[str, Any]] = []
    layer_values: dict[int, list[float]] = defaultdict(list)
    layer_outcome_values: dict[tuple[int, bool], list[float]] = defaultdict(list)
    head_values: dict[tuple[int, int], list[float]] = defaultdict(list)
    head_outcome_values: dict[tuple[int, int, bool], list[float]] = defaultdict(list)
    expected_layers: tuple[int, ...] | None = None
    expected_heads: tuple[tuple[int, int], ...] | None = None

    for item in items:
        sample_id = item["sample_id"]
        analysis_path = args.contrasts_root / sample_id / "analysis.json"
        if not analysis_path.is_file():
            raise SystemExit(f"missing contrast analysis: {analysis_path}")
        analysis = json.loads(analysis_path.read_text())
        if analysis.get("sample_id") != sample_id:
            raise SystemExit(f"sample mismatch in {analysis_path}")
        benchmark_correct = bool(item["benchmark_correct"])
        if analysis.get("correct") is not benchmark_correct:
            raise SystemExit(f"benchmark outcome mismatch for {sample_id}")
        if analysis.get("controls") != item.get("controls"):
            raise SystemExit(f"control prompt mismatch for {sample_id}")
        if analysis.get("excluded_controls"):
            raise SystemExit(f"contrast excluded controls for {sample_id}: {analysis['excluded_controls']}")

        layers = analysis["layer_head_statistics"]["layers"]
        heads = analysis["layer_head_statistics"]["heads"]
        item_layers = tuple(int(row["transformer_layer"]) for row in layers)
        item_heads = tuple((int(row["transformer_layer"]), int(row["head"])) for row in heads)
        if expected_layers is None:
            expected_layers = item_layers
            expected_heads = item_heads
        elif item_layers != expected_layers or item_heads != expected_heads:
            raise SystemExit(f"layer/head layout mismatch for {sample_id}")

        by_layer = {
            int(row["transformer_layer"]): float(row["prompt_mean_target_lift"])
            for row in layers
        }
        by_head = {
            (int(row["transformer_layer"]), int(row["head"])): float(
                row["prompt_difference"]["target_lift"]
            )
            for row in heads
        }
        for layer, value in by_layer.items():
            layer_values[layer].append(value)
            layer_outcome_values[(layer, benchmark_correct)].append(value)
        for (layer, head), value in by_head.items():
            head_values[(layer, head)].append(value)
            head_outcome_values[(layer, head, benchmark_correct)].append(value)

        focus_layer_value = by_layer[args.focus_layer]
        focus_head_value = by_head[(args.focus_layer, args.focus_head)]
        layer_rank = next(
            index
            for index, (layer, _) in enumerate(
                sorted(by_layer.items(), key=lambda pair: (-pair[1], pair[0])), start=1
            )
            if layer == args.focus_layer
        )
        head_rank_in_layer = next(
            index
            for index, ((layer, head), _) in enumerate(
                sorted(
                    (pair for pair in by_head.items() if pair[0][0] == args.focus_layer),
                    key=lambda pair: (-pair[1], pair[0][1]),
                ),
                start=1,
            )
            if layer == args.focus_layer and head == args.focus_head
        )
        per_item.append(
            {
                "sample_id": sample_id,
                "label": item.get("label", sample_id),
                "benchmark_correct": benchmark_correct,
                "focus_layer_target_lift": focus_layer_value,
                "focus_layer_rank": layer_rank,
                "focus_head_target_lift": focus_head_value,
                "focus_head_rank_within_layer": head_rank_in_layer,
                "control_stability_mean_leave_one_out_cosine": analysis["stability"][
                    "mean_leave_one_out_cosine"
                ],
                "control_stability_minimum_leave_one_out_cosine": analysis["stability"][
                    "minimum_leave_one_out_cosine"
                ],
            }
        )

    correct_count = sum(bool(item["benchmark_correct"]) for item in items)
    declared = manifest.get("balance", {})
    if correct_count != declared.get("benchmark_successes") or len(items) - correct_count != declared.get(
        "benchmark_failures"
    ):
        raise SystemExit("manifest balance declaration does not match items")

    layer_rows: list[dict[str, Any]] = []
    for layer in sorted(layer_values):
        stats = _summary(layer_values[layer], bootstrap_samples=args.bootstrap_samples, rng=rng)
        layer_rows.append(
            {
                "transformer_layer": layer,
                **stats,
                "success_mean": _mean(layer_outcome_values[(layer, True)]),
                "failure_mean": _mean(layer_outcome_values[(layer, False)]),
            }
        )
    ranked_layers = _ranked(layer_rows)
    for rank, row in enumerate(ranked_layers, start=1):
        row["aggregate_rank"] = rank

    head_rows: list[dict[str, Any]] = []
    for layer, head in sorted(head_values):
        stats = _summary(head_values[(layer, head)], bootstrap_samples=args.bootstrap_samples, rng=rng)
        head_rows.append(
            {
                "transformer_layer": layer,
                "head": head,
                **stats,
                "success_mean": _mean(head_outcome_values[(layer, head, True)]),
                "failure_mean": _mean(head_outcome_values[(layer, head, False)]),
            }
        )
    ranked_heads = _ranked(head_rows)
    for rank, row in enumerate(ranked_heads, start=1):
        row["aggregate_rank"] = rank

    focus_layer = next(row for row in layer_rows if row["transformer_layer"] == args.focus_layer)
    focus_head = next(
        row
        for row in head_rows
        if row["transformer_layer"] == args.focus_layer and row["head"] == args.focus_head
    )
    focus_heads_in_layer = _ranked(
        [row for row in head_rows if row["transformer_layer"] == args.focus_layer]
    )
    focus_head_rank_within_layer = next(
        rank
        for rank, row in enumerate(focus_heads_in_layer, start=1)
        if row["head"] == args.focus_head
    )
    next_best_layer = next(row for row in ranked_layers if row["transformer_layer"] != args.focus_layer)

    result = {
        "schema_version": 1,
        "cohort": {
            "name": manifest.get("name"),
            "items": len(items),
            "successes": correct_count,
            "failures": len(items) - correct_count,
            "controls_per_item": manifest.get("capture", {}).get("controls_per_item"),
            "benchmark_protocol": manifest.get("benchmark_protocol"),
            "max_pixels": manifest.get("capture", {}).get("max_pixels"),
            "model": manifest.get("model"),
            "repository_commit": manifest.get("capture", {}).get("repository_commit"),
        },
        "metric": {
            "name": "same-image prompt-differential target lift",
            "definition": "Positive target-minus-control attribution mass inside the target box, divided by target-box area fraction.",
            "interpretation": "1.0 is spatially uniform positive mass; values above 1.0 are enriched inside the target box.",
            "aggregation": "Arithmetic mean across items, with deterministic item-level bootstrap intervals.",
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
        },
        "focus": {
            "layer": args.focus_layer,
            "head": args.focus_head,
            "layer_summary": focus_layer,
            "layer_top_item_count": sum(row["focus_layer_rank"] == 1 for row in per_item),
            "layer_median_item_rank": statistics.median(row["focus_layer_rank"] for row in per_item),
            "next_best_aggregate_layer": next_best_layer,
            "mean_ratio_to_next_best_layer": focus_layer["mean"] / next_best_layer["mean"],
            "head_summary": focus_head,
            "head_rank_within_layer": focus_head_rank_within_layer,
            "head_top_within_layer_item_count": sum(
                row["focus_head_rank_within_layer"] == 1 for row in per_item
            ),
        },
        "per_item": per_item,
        "layers": sorted(layer_rows, key=lambda row: row["transformer_layer"]),
        "heads": sorted(head_rows, key=lambda row: (row["transformer_layer"], row["head"])),
        "caveat": "This is a four-item balanced pilot. Bootstrap intervals describe this frozen cohort and are not population confidence intervals.",
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "aggregate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_csv(args.output / "items.csv", per_item)
    _write_csv(args.output / "layers.csv", sorted(layer_rows, key=lambda row: row["transformer_layer"]))
    _write_csv(args.output / "heads.csv", sorted(head_rows, key=lambda row: (row["transformer_layer"], row["head"])))
    print(json.dumps({"output": str(args.output), "focus": result["focus"]}, indent=2))


if __name__ == "__main__":
    main()
