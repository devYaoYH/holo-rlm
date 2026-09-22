"""Build native-resolution post-hoc attribution cases for two free Holo generations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.demo.screenspot import SCREENSPOT_LOCALIZATION_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks/attention_attribution/screenspot_freegen_native_xy_v1.json"
LOCALIZER_PREFIX = (
    "Localize an element on the GUI image according to the provided target "
    "and output a click position.\n"
    f" * You must output a valid JSON following the format: {SCREENSPOT_LOCALIZATION_SCHEMA}\n"
    " Your target is:\n"
)


def _relative(path: Path, base: Path) -> str:
    return str(Path("../..") / path.relative_to(ROOT)) if base.name == "attention_attribution" else str(path)


def _candidate(coordinate: list[int], field: str, label: str) -> dict[str, Any]:
    text = json.dumps({"x": coordinate[0], "y": coordinate[1]}, separators=(",", ":"))
    value = str(coordinate[0] if field == "x" else coordinate[1])
    start = text.index(value, text.index(f'"{field}"'))
    return {"label": label, "text": text, "scored_spans": [[start, start + len(value)]]}


def _case(
    *,
    sample_id: str,
    instruction: str,
    bbox: list[int],
    coordinate: list[int],
    field: str,
    strict_correct: bool,
    output_dir: Path,
) -> dict[str, Any]:
    image_path = ROOT / f"artifacts/screenspot-presentation/native-freegen-v1/source/{sample_id}.png"
    candidate = _candidate(coordinate, field, "free_generated_action")
    return {
        "id": f"{sample_id}_{field}",
        "protocol": "hcompany_element_localization_v1_posthoc_generated_action",
        "request": {
            "model": "Hcompany/Holo-3.1-4B",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_path", "path": _relative(image_path, output_dir)},
                        {"type": "text", "text": LOCALIZER_PREFIX + instruction},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": {"json": SCREENSPOT_LOCALIZATION_SCHEMA},
        },
        # The second candidate is intentionally identical. The attention-only
        # runner reconstructs candidate 0; duplication preserves the shared
        # two-candidate preparation API without introducing a distractor.
        "candidates": [candidate, {**candidate, "label": "alignment_duplicate"}],
        "case_metadata": {
            "sample_id": sample_id,
            "instruction": instruction,
            "bbox": bbox,
            "free_generated_coordinate": coordinate,
            "attributed_field": field,
            "strict_correct": strict_correct,
            "source_run": "data/remote-results/run/screenspot-full-official/summary.json",
            "source_resolution": [2880, 1800],
        },
    }


def build(output: Path) -> dict[str, Any]:
    output = output.expanduser().resolve()
    rows = [
        {
            "sample_id": "powerpoint_windows_63",
            "instruction": "Fill color",
            "bbox": [585, 419, 619, 480],
            "coordinate": [209, 241],
            "strict_correct": True,
            "trace_id": "trace-20260921T111129Z-5e703d33e804",
        },
        {
            "sample_id": "powerpoint_windows_54",
            "instruction": "Choose the language for proofing tools",
            "bbox": [333, 69, 391, 124],
            "coordinate": [124, 72],
            "strict_correct": False,
            "trace_id": "trace-20260921T111102Z-7175e3911919",
        },
    ]
    cases = [
        _case(output_dir=output.parent, field=field, **{key: value for key, value in row.items() if key != "trace_id"})
        for row in rows
        for field in ("x", "y")
    ]
    manifest = {
        "schema_version": 1,
        "name": "screenspot_freegen_native_xy_v1",
        "processor_path": "../../models/Holo-3.1-4B",
        "models": [
            {"label": "Qwen3.5-4B", "role": "base", "path": "../../models/Qwen3.5-4B"},
            {"label": "Holo3.1-4B", "role": "tuned", "path": "../../models/Holo-3.1-4B"},
        ],
        "image_min_pixels": 65536,
        "image_max_pixels": 16777216,
        "score_reduction": "mean",
        "experiment_design": {
            "generation": "Coordinates come from the official free-generation full benchmark run.",
            "attribution": (
                "Post-hoc deterministic replay of Holo's own generated JSON action, with x and y token "
                "queries reconstructed separately and no oracle or distractor candidate comparison."
            ),
            "resolution": "Original 2880x1800 images at image_max_pixels=16777216; no downsampling.",
            "source_traces": {row["sample_id"]: row["trace_id"] for row in rows},
        },
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "cases": len(result["cases"])}, indent=2))


if __name__ == "__main__":
    main()
