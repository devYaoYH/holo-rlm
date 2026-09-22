#!/usr/bin/env python3
"""Freeze a size-spread cohort of native ScreenSpot-Pro successes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from demo.screenspot_resolution import DEFAULT_LINEAR_SCALES, build_native_success_cohort


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--summary", type=Path, default=Path("data/remote-results/run/screenspot-full-official/summary.json"))
    result.add_argument("--annotations", type=Path, default=Path("data/screenspot-pro/annotations"))
    result.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/resolution_ablation/screenspot_success_retention_v1.json"),
    )
    result.add_argument("--per-stratum", type=int, default=6)
    return result


def main() -> None:
    args = parser().parse_args()
    summary = json.loads(args.summary.read_text())
    annotation_paths = sorted(args.annotations.glob("*.json"))
    annotations = [record for path in annotation_paths for record in json.loads(path.read_text())]
    applications = ("photoshop", "powerpoint", "vscode")
    cases = build_native_success_cohort(
        summary,
        annotations,
        per_stratum=args.per_stratum,
        applications=applications,
    )
    payload = {
        "schema_version": 1,
        "id": "screenspot_success_retention_v1",
        "dataset": "likaixin/ScreenSpot-Pro",
        "inference_protocol": "hcompany_element_localization_v1",
        "model": "Hcompany/Holo-3.1-4B",
        "selection": {
            "condition": "strictly correct in the 1,581-item checkpoint-native reference run",
            "estimand": "paired retention under downsampling, not unconditional benchmark accuracy",
            "applications": list(applications),
            "ui_types": ["icon", "text"],
            "per_application_ui_stratum": args.per_stratum,
            "within_stratum_sampling": "evenly spaced ranks by normalized target-box area",
            "reference_summary_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
            "reference_model_revision": summary.get("server_model", {}).get("revision"),
            "reference_strict_accuracy": summary.get("strict_accuracy"),
        },
        "linear_scales": list(DEFAULT_LINEAR_SCALES),
        "area_fractions": [value * value for value in DEFAULT_LINEAR_SCALES],
        "interpolation": "PIL.Image.Resampling.LANCZOS",
        "controls": [
            "same image content and aspect ratio",
            "same instruction and official localization prompt",
            "same structured output schema and normalized 0-1000 coordinates",
            "same model process and deterministic decoding",
            "server pixel ceiling must exceed every native source image",
        ],
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
