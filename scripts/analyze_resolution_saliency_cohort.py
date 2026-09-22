#!/usr/bin/env python3
"""Combine resolution-saliency runs and quantify uncertainty and click/map disagreement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

METRICS = (
    "target_mass",
    "target_lift",
    "best_target_patch_rank",
    "peak_distance_diagonal",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("summaries", nargs="+", type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--markdown-output", type=Path)
    result.add_argument("--bootstrap-resamples", type=int, default=10_000)
    result.add_argument("--seed", type=int, default=20260922)
    return result


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_interval(
    values: list[float],
    statistic: Callable[[Iterable[float]], float],
    *,
    resamples: int,
    rng: random.Random,
) -> list[float] | None:
    if not values:
        return None
    estimates = [
        statistic(rng.choice(values) for _ in values)
        for _ in range(resamples)
    ]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> list[float] | None:
    if count == 0:
        return None
    proportion = successes / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def prompt_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return row["metrics"]["prompt_difference"]


def summarize_group(rows: list[dict[str, Any]], *, resamples: int, rng: random.Random) -> dict[str, Any]:
    strict_correct = sum(bool(row["strict_correct"]) for row in rows)
    peak_inside = sum(bool(prompt_metrics(row)["peak_inside_target"]) for row in rows)
    rank_one = sum(int(prompt_metrics(row)["best_target_patch_rank"]) == 1 for row in rows)
    result: dict[str, Any] = {
        "count": len(rows),
        "strict_correct_count": strict_correct,
        "strict_correct_rate": strict_correct / len(rows) if rows else None,
        "strict_correct_wilson_95": wilson_interval(strict_correct, len(rows)),
        "peak_inside_count": peak_inside,
        "peak_inside_rate": peak_inside / len(rows) if rows else None,
        "peak_inside_wilson_95": wilson_interval(peak_inside, len(rows)),
        "rank_one_count": rank_one,
        "rank_one_rate": rank_one / len(rows) if rows else None,
        "rank_one_wilson_95": wilson_interval(rank_one, len(rows)),
    }
    for metric in METRICS:
        values = [float(prompt_metrics(row)[metric]) for row in rows]
        result[f"median_{metric}"] = statistics.median(values) if values else None
        result[f"median_{metric}_bootstrap_95"] = bootstrap_interval(
            values,
            statistics.median,
            resamples=resamples,
            rng=rng,
        )
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combine(paths: list[Path], *, resamples: int, seed: int) -> dict[str, Any]:
    rows_by_key: dict[tuple[str, float], dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    model_ids: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text())
        model_ids.add(str(payload["model_id"]))
        sources.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "experiment": payload["experiment"],
                "case_count": payload["case_count"],
                "condition_count": len(payload["results"]),
            }
        )
        for row in payload["results"]:
            key = (str(row["sample_id"]), float(row["linear_scale"]))
            if key in rows_by_key and rows_by_key[key] != row:
                raise SystemExit(f"conflicting duplicate condition: {key}")
            rows_by_key[key] = row
    if len(model_ids) != 1:
        raise SystemExit(f"model mismatch: {sorted(model_ids)}")

    rows = sorted(rows_by_key.values(), key=lambda row: (row["sample_id"], -float(row["linear_scale"])))
    sample_ids = sorted({str(row["sample_id"]) for row in rows})
    scales = sorted({float(row["linear_scale"]) for row in rows}, reverse=True)
    expected = {(sample_id, scale) for sample_id in sample_ids for scale in scales}
    missing = sorted(expected - set(rows_by_key))
    if missing:
        raise SystemExit(f"missing conditions: {missing}")

    native_by_sample = {
        sample_id: rows_by_key[(sample_id, 1.0)]
        for sample_id in sample_ids
    }
    rng = random.Random(seed)
    scale_summaries: list[dict[str, Any]] = []
    for scale in scales:
        scale_rows = [row for row in rows if float(row["linear_scale"]) == scale]
        lost = [row for row in scale_rows if not row["strict_correct"]]
        retained = [row for row in scale_rows if row["strict_correct"]]
        ratios: dict[str, list[float]] = defaultdict(list)
        differences: dict[str, list[float]] = defaultdict(list)
        for row in scale_rows:
            native = native_by_sample[str(row["sample_id"])]
            current_metrics = prompt_metrics(row)
            native_metrics = prompt_metrics(native)
            for metric in ("target_mass", "target_lift"):
                current_value = float(current_metrics[metric])
                native_value = float(native_metrics[metric])
                differences[metric].append(current_value - native_value)
                if native_value > 0:
                    ratios[metric].append(current_value / native_value)
        lost_peak_inside = sum(bool(prompt_metrics(row)["peak_inside_target"]) for row in lost)
        lost_rank_one = sum(int(prompt_metrics(row)["best_target_patch_rank"]) == 1 for row in lost)
        scale_summaries.append(
            {
                "linear_scale": scale,
                "groups": {
                    "all": summarize_group(scale_rows, resamples=resamples, rng=rng),
                    "retained": summarize_group(retained, resamples=resamples, rng=rng),
                    "lost": summarize_group(lost, resamples=resamples, rng=rng),
                },
                "paired_to_native": {
                    metric: {
                        "median_ratio": statistics.median(ratios[metric]) if ratios[metric] else None,
                        "median_ratio_bootstrap_95": bootstrap_interval(
                            ratios[metric], statistics.median, resamples=resamples, rng=rng
                        ),
                        "median_difference": statistics.median(differences[metric]),
                        "median_difference_bootstrap_95": bootstrap_interval(
                            differences[metric], statistics.median, resamples=resamples, rng=rng
                        ),
                    }
                    for metric in ("target_mass", "target_lift")
                },
                "lost_click_map_disagreement": {
                    "lost_click_count": len(lost),
                    "peak_inside_count": lost_peak_inside,
                    "peak_inside_rate": lost_peak_inside / len(lost) if lost else None,
                    "rank_one_count": lost_rank_one,
                    "rank_one_rate": lost_rank_one / len(lost) if lost else None,
                },
            }
        )

    nonnative_lost = [row for row in rows if float(row["linear_scale"]) < 1.0 and not row["strict_correct"]]
    return {
        "schema_version": 1,
        "estimand": "Resolution sensitivity within a balanced sample conditioned on native-resolution strict success.",
        "model_id": model_ids.pop(),
        "sources": sources,
        "sample_count": len(sample_ids),
        "condition_count": len(rows),
        "linear_scales": scales,
        "bootstrap": {"resamples": resamples, "seed": seed, "interval": "percentile 95%"},
        "scales": scale_summaries,
        "nonnative_lost_click_map_disagreement": {
            "lost_click_count": len(nonnative_lost),
            "peak_inside_count": sum(bool(prompt_metrics(row)["peak_inside_target"]) for row in nonnative_lost),
            "rank_one_count": sum(int(prompt_metrics(row)["best_target_patch_rank"]) == 1 for row in nonnative_lost),
        },
        "results": rows,
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Resolution-saliency cohort",
        "",
        f"{result['sample_count']} native-success cases, {result['condition_count']} conditions. "
        "Attribution is value-norm rollout minus the mean of four same-image instruction controls.",
        "",
        "| Linear resolution | Strict hits | Median target mass | Median target lift | Median best-patch rank | Lost clicks with peak inside |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for scale in result["scales"]:
        group = scale["groups"]["all"]
        disagreement = scale["lost_click_map_disagreement"]
        lost = disagreement["lost_click_count"]
        peak = disagreement["peak_inside_count"]
        lines.append(
            f"| {scale['linear_scale'] * 100:.0f}% | {group['strict_correct_count']}/{group['count']} | "
            f"{group['median_target_mass'] * 100:.2f}% | {group['median_target_lift']:.1f}x | "
            f"{group['median_best_target_patch_rank']:.1f} | {peak}/{lost} |"
        )
    quarter = min(result["scales"], key=lambda row: row["linear_scale"])
    half = next((row for row in result["scales"] if row["linear_scale"] == 0.5), None)
    paired_mass = quarter["paired_to_native"]["target_mass"]
    paired_interval = paired_mass["median_ratio_bootstrap_95"]
    disagreement = result["nonnative_lost_click_map_disagreement"]
    lost_count = disagreement["lost_click_count"]
    rank_one_count = disagreement["rank_one_count"]
    lines.extend(
        [
            "",
            "## Key comparisons",
            "",
            f"At quarter resolution, the median within-case target-mass ratio is "
            f"{paired_mass['median_ratio'] * 100:.2f}% of native "
            f"(bootstrap 95% interval {paired_interval[0] * 100:.2f}% to "
            f"{paired_interval[1] * 100:.2f}%).",
            "",
            f"Across {lost_count} non-native missed clicks, {rank_one_count} "
            f"({rank_one_count / lost_count * 100:.1f}%) still rank an oracle-overlapping "
            "patch first after diverse-instruction subtraction."
            + (
                f" At half resolution this occurs in "
                f"{half['lost_click_map_disagreement']['rank_one_count']}/"
                f"{half['lost_click_map_disagreement']['lost_click_count']} misses; at quarter "
                f"resolution it occurs in {quarter['lost_click_map_disagreement']['rank_one_count']}/"
                f"{quarter['lost_click_map_disagreement']['lost_click_count']}."
                if half
                else ""
            ),
            "",
            "Most resolution-induced misses coincide with weaker target localization, while a repeatable minority retain a top-ranked target patch and fail at coordinate readout.",
            "",
            "The cohort is selected on native success, so these are paired resolution-sensitivity diagnostics rather than unconditional benchmark estimates.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.bootstrap_resamples < 100:
        raise SystemExit("--bootstrap-resamples must be at least 100")
    result = combine(args.summaries, resamples=args.bootstrap_resamples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown(result))
    print(json.dumps({key: result[key] for key in ("sample_count", "condition_count", "linear_scales")}, indent=2))


if __name__ == "__main__":
    main()
