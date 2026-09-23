#!/usr/bin/env python3
"""Build the matched 25%-resolution attention manifest for the VS Code sampling miss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from src.demo.screenspot import SCREENSPOT_LOCALIZATION_SCHEMA


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks/attention_attribution/vscode_refresh_sampling_25pct_v1.json"
DEFAULT_IMAGE = ROOT / "data/local-results/vscode-refresh-sampling-attention-25pct-v1/source/vscode_macos_0_25pct.png"
SOURCE_IMAGE = ROOT / "data/screenspot-pro/images/vscode_macos_0.png"
SAMPLING_SUMMARY = ROOT / "data/local-results/coordinate-stochastic-widthguard-v1/summary.json"
SCALE = 0.25
LOCALIZER_PREFIX = (
    "Localize an element on the GUI image according to the provided target "
    "and output a click position.\n"
    f" * You must output a valid JSON following the format: {SCREENSPOT_LOCALIZATION_SCHEMA}\n"
    " Your target is:\n"
)


def _scaled_bbox(bbox: list[float]) -> list[float]:
    return [value * SCALE for value in bbox]


def _normalized_center(bbox: list[float], image_size: tuple[int, int]) -> list[int]:
    x = ((bbox[0] + bbox[2]) / 2) / image_size[0] * 1000
    y = ((bbox[1] + bbox[3]) / 2) / image_size[1] * 1000
    return [round(x), round(y)]


def _request(image_path: str, instruction: str) -> dict[str, Any]:
    return {
        "model": "Hcompany/Holo-3.1-4B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_path", "path": image_path},
                    {"type": "text", "text": LOCALIZER_PREFIX + instruction},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
        "structured_outputs": {"json": SCREENSPOT_LOCALIZATION_SCHEMA},
    }


def _case(
    *,
    case_id: str,
    role: str,
    instruction: str,
    bbox_source: list[float],
    coordinate: list[int],
    image_path: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate = {"label": "declared_action", "json_coordinate": coordinate}
    return {
        "id": case_id,
        "protocol": "hcompany_element_localization_v1_sampling_attention",
        "request": _request(image_path, instruction),
        "candidates": [candidate, {**candidate, "label": "alignment_duplicate"}],
        "case_metadata": {
            "ensemble_group": "vscode_macos_0_refresh_25pct",
            "ensemble_role": role,
            "instruction": instruction,
            "bbox": _scaled_bbox(bbox_source),
            "bbox_source_pixels": bbox_source,
            "image_size": [640, 416],
            "source_image_size": [2560, 1664],
            "declared_coordinate": coordinate,
            **(extra_metadata or {}),
        },
    }


def build(output: Path, low_image_path: Path) -> dict[str, Any]:
    output = output.expanduser().resolve()
    low_image_path = low_image_path.expanduser().resolve()
    low_image_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(SOURCE_IMAGE) as opened:
        source = opened.convert("RGB")
    low_image = source.resize((640, 416), Image.Resampling.LANCZOS)
    low_image.save(low_image_path, optimize=True)

    sampling = json.loads(SAMPLING_SUMMARY.read_text())
    result = next(row for row in sampling["results"] if row["sample_id"] == "vscode_macos_0")
    target_bbox = [473.0, 183.0, 503.0, 219.0]
    relative_image = str(Path("../..") / low_image_path.relative_to(ROOT))
    cases = [
        _case(
            case_id=f"vscode_refresh_sample_{index:02d}",
            role="target_sample",
            instruction="Refresh the file explorer.",
            bbox_source=target_bbox,
            coordinate=[candidate["coordinate"]["x"], candidate["coordinate"]["y"]],
            image_path=relative_image,
            extra_metadata={
                "sample_index": index,
                "source": "data/local-results/coordinate-stochastic-widthguard-v1/summary.json",
                "hits_oracle": candidate["hits_oracle"],
            },
        )
        for index, candidate in enumerate(result["candidates"])
    ]

    controls = [
        ("get_score_tab", "Open the get_score.py editor tab.", [914.0, 120.0, 1162.0, 165.0]),
        ("disable_extension", "Click Disable for the Bracket Pair Color DLW extension.", [2040.0, 379.0, 2113.0, 423.0]),
        ("call_gpt_tab", "Open the call_gpt4o.py editor tab.", [2173.0, 120.0, 2397.0, 165.0]),
        ("readme", "Open README.md in the Explorer.", [88.0, 747.0, 199.0, 780.0]),
    ]
    cases.extend(
        _case(
            case_id=f"vscode_refresh_control_{slug}",
            role="control",
            instruction=instruction,
            bbox_source=bbox,
            coordinate=_normalized_center(bbox, source.size),
            image_path=relative_image,
            extra_metadata={"bbox_source": "manually curated from the same screenshot"},
        )
        for slug, instruction, bbox in controls
    )

    manifest = {
        "schema_version": 1,
        "name": "vscode_refresh_sampling_25pct_attention_v1",
        "processor_path": "../../models/Holo-3.1-4B",
        "models": [
            {"label": "Qwen3.5-4B", "role": "base", "path": "../../models/Qwen3.5-4B"},
            {"label": "Holo3.1-4B", "role": "tuned", "path": "../../models/Holo-3.1-4B"},
        ],
        "image_min_pixels": 65536,
        "image_max_pixels": 16777216,
        "score_reduction": "mean",
        "experiment_design": {
            "primary_method": "value_norm_attention",
            "target_aggregation": "Mean of eight separately normalized maps, one per guarded stochastic sample.",
            "control_baseline": "Mean of four separately normalized same-image alternate-instruction maps.",
            "resolution": "Exact 640x416 PIL-Lanczos image used by the 25%-linear-scale sampling pilot.",
            "coordinate_contract": "Official ScreenSpot-Pro 0-1000 JSON localization schema.",
            "claim_boundary": "Selected single-case diagnostic; not a population estimate or causal intervention.",
        },
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-output", type=Path, default=DEFAULT_IMAGE)
    args = parser.parse_args()
    result = build(args.output, args.image_output)
    print(json.dumps({"output": str(args.output.resolve()), "cases": len(result["cases"])}, indent=2))


if __name__ == "__main__":
    main()
