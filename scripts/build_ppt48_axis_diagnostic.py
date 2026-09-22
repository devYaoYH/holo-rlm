"""Split the PowerPoint-48 prompt-control saliency estimator into x/y token queries."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "benchmarks/attention_attribution/powerpoint48_hotel35_native_controls_v1.json"
DEFAULT_OUTPUT = ROOT / "benchmarks/attention_attribution/powerpoint48_native_axis_diagnostic_v1.json"


def _candidate(coordinate: list[int], field: str, label: str) -> dict:
    text = json.dumps({"x": coordinate[0], "y": coordinate[1]}, separators=(",", ":"))
    value = str(coordinate[0] if field == "x" else coordinate[1])
    start = text.index(value, text.index(f'"{field}"'))
    return {"label": label, "text": text, "scored_spans": [[start, start + len(value)]]}


def build(source: Path, output: Path) -> dict:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    manifest = json.loads(source.read_text())
    cases = []
    for row in manifest["cases"]:
        if row["case_metadata"]["ensemble_group"] != "powerpoint_windows_48":
            continue
        coordinate = list(row["candidates"][0]["json_coordinate"])
        for field in ("x", "y"):
            case = copy.deepcopy(row)
            case["id"] = f"{row['id']}_{field}"
            candidate = _candidate(coordinate, field, "curated_coordinate")
            case["candidates"] = [candidate, {**candidate, "label": "alignment_duplicate"}]
            case["case_metadata"]["axis_source_case"] = row["id"]
            case["case_metadata"]["attributed_field"] = field
            cases.append(case)
    result = {
        **manifest,
        "name": "powerpoint48_native_axis_diagnostic_v1",
        "experiment_design": {
            "purpose": (
                "Diagnostic decomposition of the existing PowerPoint-48 prompt-control map into separate "
                "x-coordinate and y-coordinate token queries."
            ),
            "resolution": "Original 2880x1800 image; no downsampling.",
        },
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.source, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "cases": len(result["cases"])}, indent=2))


if __name__ == "__main__":
    main()
