#!/usr/bin/env python3
"""Summarize a ScreenSpot-Pro benchmark run by UI and instruction action family."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ACTION_FAMILIES: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "Navigate / reveal",
        frozenset(
            "open show view go switch expand collapse hide minimize maximize back forward launch preview "
            "refresh find search navigate display return access"
            .split()
        ),
    ),
    (
        "Select / activate",
        frozenset(
            "select choose click press tap check mark use pick activate focus hover enable turn confirm apply"
            .split()
        ),
    ),
    (
        "Create / insert",
        frozenset("add create insert draw generate make input enter type plot build new duplicate attach".split()),
    ),
    (
        "Edit / format",
        frozenset(
            "change set edit modify adjust rename align rotate zoom stretch update color fill blur crop filter "
            "convert resize increase decrease move arrange flip"
            .split()
        ),
    ),
    (
        "Remove / close",
        frozenset("close delete remove clear cancel stop disable unpin dismiss exit erase".split()),
    ),
    (
        "File / transfer",
        frozenset("save copy paste cut import export share send install download upload print".split()),
    ),
)
OTHER_FAMILY = "Other"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("summary", type=Path, help="screenspot-benchmark summary.json")
    result.add_argument("--output", type=Path, required=True)
    return result


def action_family(instruction: str) -> str:
    """Assign the first recognized instruction verb to one fixed family."""

    lookup = {word: family for family, words in ACTION_FAMILIES for word in words}
    for token in re.findall(r"[a-z]+", instruction.lower()):
        if token in lookup:
            return lookup[token]
    return OTHER_FAMILY


def wilson95(correct: int, count: int) -> tuple[float, float]:
    if count <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    proportion = correct / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    return (center - radius, center + radius)


def metric(correct: int, count: int) -> dict[str, float | int]:
    lower, upper = wilson95(correct, count)
    return {
        "count": count,
        "correct": correct,
        "accuracy": correct / count if count else 0.0,
        "wilson95_lower": lower,
        "wilson95_upper": upper,
    }


def grouped(rows: list[dict[str, Any]], key) -> dict[Any, dict[str, float | int]]:
    counts: dict[Any, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        bucket = counts[key(row)]
        bucket[0] += 1
        bucket[1] += int(bool(row["strict_correct"]))
    return {name: metric(correct, count) for name, (count, correct) in counts.items()}


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    source_bytes = args.summary.read_bytes()
    source = json.loads(source_bytes)
    rows = source.get("items")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("summary must contain a non-empty items list")
    if source.get("completed_count") != len(rows) or source.get("failed_count") != 0:
        raise SystemExit("benchmark run is incomplete or contains failed requests")

    families = [family for family, _ in ACTION_FAMILIES] + [OTHER_FAMILY]
    enriched = [dict(row, action_family=action_family(row["instruction"])) for row in rows]
    by_ui = grouped(enriched, lambda row: row["ui_type"])
    by_platform_ui = grouped(enriched, lambda row: (row["platform"], row["ui_type"]))
    by_action = grouped(enriched, lambda row: row["action_family"])
    by_application = grouped(enriched, lambda row: row["application"])
    matrix = grouped(enriched, lambda row: (row["ui_type"], row["action_family"]))

    cells = [
        {"ui_type": ui_type, "action_family": family, **matrix[(ui_type, family)]}
        for ui_type in ("icon", "text")
        for family in families
    ]
    populated_cells = [cell for cell in cells if cell["count"] >= 25 and cell["action_family"] != OTHER_FAMILY]
    weakest_cell = min(populated_cells, key=lambda cell: (cell["accuracy"], -cell["count"]))

    result = {
        "schema_version": 1,
        "source": {
            "path": str(args.summary),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "inference_protocol": source.get("inference_protocol"),
            "model_id": source.get("model_id"),
            "model_revision": source.get("server_model", {}).get("revision"),
            "runtime_seconds": source.get("runtime_seconds"),
            "image_max_pixels": source.get("server_model", {})
            .get("inference_configuration", {})
            .get("image_max_pixels"),
        },
        "overall": metric(int(source["strict_correct"]), len(enriched)),
        "format_valid": metric(int(source["format_valid"]), len(enriched)),
        "reported_reference_accuracy": 0.665,
        "reported_reference_note": "Reference value supplied by the project owner from H Company's Holo3.1-4B benchmark report.",
        "by_ui_type": {key: by_ui[key] for key in sorted(by_ui)},
        "by_platform_ui": [
            {"platform": platform, "ui_type": ui_type, **values}
            for (platform, ui_type), values in sorted(by_platform_ui.items())
        ],
        "action_family_order": families,
        "by_action_family": {family: by_action[family] for family in families},
        "ui_action_matrix": cells,
        "weakest_labeled_cell_minimum_n_25": weakest_cell,
        "by_application": [
            {"application": application, **values}
            for application, values in sorted(by_application.items(), key=lambda pair: (pair[1]["accuracy"], pair[0]))
        ],
        "taxonomy": {
            "method": "Tokenize the English instruction, scan left to right, and assign the first verb found in the fixed keyword dictionary. Unmatched instructions enter Other.",
            "families": {family: sorted(words) for family, words in ACTION_FAMILIES},
            "official": False,
            "caveat": "Action families are a deterministic diagnostic taxonomy, not an official ScreenSpot-Pro label.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "overall": result["overall"], "weakest": weakest_cell}, indent=2))


if __name__ == "__main__":
    main()
