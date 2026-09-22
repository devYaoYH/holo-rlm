#!/usr/bin/env python3
"""Measure low-resolution coordinate spread from Holo beam candidates on ScreenSpot-Pro."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "instrumented_server" / "src"))

from instrumented_holo.model import InstrumentedHolo, generation_stop_strings  # noqa: E402
from instrumented_holo.settings import Settings  # noqa: E402
from demo.screenspot import (  # noqa: E402
    SCREENSPOT_PROTOCOL,
    build_screenspot_request,
    load_screenspot_sample,
    point_hits_bbox,
)

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_COORDINATE_RE = re.compile(r'["\']?(x|y)["\']?\s*:\s*(-?\d+)')


def parse_coordinate_text(text: str) -> dict[str, int] | None:
    """Parse one exact or lightly wrapped x/y JSON completion."""

    for match in _JSON_OBJECT_RE.finditer(text):
        fragment = match.group(0)
        try:
            payload = json.loads(fragment)
        except json.JSONDecodeError:
            pairs = {key: int(value) for key, value in _COORDINATE_RE.findall(fragment)}
            payload = pairs if set(pairs) == {"x", "y"} else None
        if not isinstance(payload, dict) or set(payload) != {"x", "y"}:
            continue
        if type(payload["x"]) is not int or type(payload["y"]) is not int:
            continue
        if 0 <= payload["x"] <= 1000 and 0 <= payload["y"] <= 1000:
            return {"x": payload["x"], "y": payload["y"]}
    return None


def pairwise_distances(points: list[tuple[int, int]]) -> list[float]:
    return [
        math.hypot(points[left][0] - points[right][0], points[left][1] - points[right][1])
        for left in range(len(points))
        for right in range(left + 1, len(points))
    ]


def summarize_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [candidate for candidate in candidates if candidate.get("coordinate")]
    points = [(row["coordinate"]["x"], row["coordinate"]["y"]) for row in valid]
    if not points:
        return {
            "valid_count": 0,
            "unique_count": 0,
            "hit_count": 0,
            "hit_fraction": 0.0,
            "centroid": None,
            "std_x": None,
            "std_y": None,
            "rms_radius": None,
            "max_pairwise_distance": None,
            "max_pairwise_fraction_of_diagonal": None,
        }
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    std_x = math.sqrt(sum((point[0] - mean_x) ** 2 for point in points) / len(points))
    std_y = math.sqrt(sum((point[1] - mean_y) ** 2 for point in points) / len(points))
    rms_radius = math.sqrt(
        sum((point[0] - mean_x) ** 2 + (point[1] - mean_y) ** 2 for point in points) / len(points)
    )
    distances = pairwise_distances(points)
    max_pairwise = max(distances, default=0.0)
    hits = sum(bool(row.get("hits_oracle")) for row in valid)
    return {
        "valid_count": len(valid),
        "unique_count": len(set(points)),
        "hit_count": hits,
        "hit_fraction": hits / len(valid),
        "centroid": {"x": mean_x, "y": mean_y},
        "std_x": std_x,
        "std_y": std_y,
        "rms_radius": rms_radius,
        "max_pairwise_distance": max_pairwise,
        "max_pairwise_fraction_of_diagonal": max_pairwise / math.hypot(1000, 1000),
    }


def source_point(coordinate: dict[str, int], image_size: tuple[int, int]) -> dict[str, float]:
    width, height = image_size
    return {
        "x": coordinate["x"] * width / 1000,
        "y": coordinate["y"] * height / 1000,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = Settings(
        model_path=args.model.expanduser().resolve(),
        trace_dir=args.output.expanduser().resolve() / "traces",
        device=args.device,
        dtype=args.dtype,
        eager_attention=False,
        image_min_pixels=args.image_min_pixels,
        image_max_pixels=args.image_max_pixels,
    )
    engine = InstrumentedHolo(settings)
    engine.load()
    assert engine.model is not None and engine.processor is not None and engine.torch is not None

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for sample_id in args.sample_ids:
        sample = load_screenspot_sample(args.annotations, args.images, sample_id)
        with Image.open(sample.image_path) as source:
            source_image = source.convert("RGB")
        low_size = (
            max(1, round(source_image.width * args.scale)),
            max(1, round(source_image.height * args.scale)),
        )
        low_image = source_image.resize(low_size, Image.Resampling.LANCZOS)
        request = build_screenspot_request(low_image, sample.instruction, args.model_id)
        inputs = engine._prepare_inputs(  # noqa: SLF001 - experiment intentionally reuses the official engine path.
            request["messages"],
            request.get("tools"),
            enable_thinking=False,
        )
        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_beams,
            "early_stopping": True,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        stop_strings = generation_stop_strings(request.get("tools"), request)
        if stop_strings:
            generate_kwargs["stop_strings"] = list(stop_strings)
            generate_kwargs["tokenizer"] = engine.processor.tokenizer

        sample_started = time.perf_counter()
        with engine.torch.inference_mode():
            generated = engine.model.generate(**generate_kwargs)
        sample_runtime = time.perf_counter() - sample_started
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        generated_tokens = generated.sequences[:, prompt_tokens:]
        texts = engine.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        sequence_scores = generated.sequences_scores.detach().float().cpu().tolist()
        candidates: list[dict[str, Any]] = []
        for rank, (text, score) in enumerate(zip(texts, sequence_scores, strict=True), start=1):
            coordinate = parse_coordinate_text(text)
            candidate: dict[str, Any] = {
                "rank": rank,
                "sequence_score": score,
                "text": text,
                "coordinate": coordinate,
                "format_valid": coordinate is not None,
            }
            if coordinate is not None:
                point = source_point(coordinate, sample.image_size)
                candidate["source_pixel"] = point
                candidate["hits_oracle"] = point_hits_bbox(point, sample.bbox)
            candidates.append(candidate)
        result = {
            "sample_id": sample.id,
            "instruction": sample.instruction,
            "application": sample.application,
            "ui_type": sample.ui_type,
            "source_image_size": list(source_image.size),
            "low_resolution_size": list(low_image.size),
            "linear_scale": args.scale,
            "source_image_sha256": hashlib.sha256(sample.image_path.read_bytes()).hexdigest(),
            "oracle_bbox_source_pixels": list(sample.bbox),
            "runtime_seconds": sample_runtime,
            "prompt_tokens": prompt_tokens,
            "candidates": candidates,
        }
        result["spread"] = summarize_candidates(candidates)
        result["top1_hits_oracle"] = bool(candidates[0].get("hits_oracle"))
        result["any_beam_hits_oracle"] = any(bool(row.get("hits_oracle")) for row in candidates)
        results.append(result)
        print(json.dumps({
            "sample_id": sample.id,
            "runtime_seconds": round(sample_runtime, 2),
            "valid": result["spread"]["valid_count"],
            "unique": result["spread"]["unique_count"],
            "rms_radius": result["spread"]["rms_radius"],
            "top1_hit": result["top1_hits_oracle"],
            "any_hit": result["any_beam_hits_oracle"],
        }), flush=True)

    valid_spreads = [row["spread"]["rms_radius"] for row in results if row["spread"]["rms_radius"] is not None]
    output = {
        "schema_version": 1,
        "experiment": "coordinate_beam_spread_pilot_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "inference_protocol": SCREENSPOT_PROTOCOL,
        "method": "deterministic beam search over the official normalized x/y JSON action sequence",
        "model": engine.model_metadata(),
        "configuration": {
            "sample_ids": args.sample_ids,
            "linear_scale": args.scale,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "image_min_pixels": args.image_min_pixels,
            "image_max_pixels": args.image_max_pixels,
            "resize": "PIL Lanczos before the checkpoint processor",
            "coordinate_contract": "integer x/y in normalized 0-1000 space",
        },
        "aggregate": {
            "case_count": len(results),
            "top1_hit_count": sum(row["top1_hits_oracle"] for row in results),
            "any_beam_hit_count": sum(row["any_beam_hits_oracle"] for row in results),
            "all_beams_format_valid_count": sum(
                row["spread"]["valid_count"] == args.num_beams for row in results
            ),
            "mean_unique_coordinate_count": sum(row["spread"]["unique_count"] for row in results) / len(results),
            "mean_rms_radius": sum(valid_spreads) / len(valid_spreads) if valid_spreads else None,
            "runtime_seconds": time.perf_counter() - started,
        },
        "results": results,
        "claim_boundary": (
            "Small feasibility pilot on a selected native-success cohort. Beam spread is a diagnostic uncertainty "
            "signal, not yet a calibrated confidence estimate or evidence that adaptive cropping improves accuracy."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "summary.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "Holo-3.1-4B")
    parser.add_argument("--model-id", default="Hcompany/Holo-3.1-4B")
    parser.add_argument("--annotations", type=Path, default=REPO_ROOT / "data" / "screenspot-pro" / "annotations")
    parser.add_argument("--images", type=Path, default=REPO_ROOT / "data" / "screenspot-pro" / "images")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "local-results" / "coordinate-beam-spread-pilot-v1")
    parser.add_argument("--sample-ids", nargs="+", default=[
        "powerpoint_windows_59",
        "vscode_macos_0",
    ])
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--image-min-pixels", type=int, default=65_536)
    parser.add_argument("--image-max-pixels", type=int, default=16_777_216)
    args = parser.parse_args()
    if not 0 < args.scale <= 1:
        parser.error("--scale must be in (0, 1]")
    if args.num_beams < 2:
        parser.error("--num-beams must be at least 2")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    payload = run(cli_args)
    print(json.dumps({
        "output": str((cli_args.output / "summary.json").resolve()),
        "aggregate": payload["aggregate"],
    }, indent=2))
