"""Paired ScreenSpot-Pro resolution ablations over a frozen native-success cohort."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from .screenspot import (
    SCREENSPOT_PROTOCOL,
    build_screenspot_request,
    load_screenspot_sample,
    parse_screenspot_click,
    point_hits_bbox,
)
from .screenspot_benchmark import _server_model_metadata

DEFAULT_LINEAR_SCALES = (1.0, 0.75, 0.5, 0.25)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def evenly_spaced(values: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Choose deterministic size-spread representatives from one stratum."""

    if count < 1 or len(values) < count:
        raise ValueError("each stratum must contain at least the requested number of cases")
    ordered = sorted(values, key=lambda row: (row["normalized_target_area"], row["id"]))
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise ValueError("evenly spaced selection produced duplicate ranks")
    return [ordered[index] for index in indices]


def build_native_success_cohort(
    benchmark_summary: dict[str, Any],
    annotations: list[dict[str, Any]],
    *,
    per_stratum: int,
    applications: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Select native successes across application, UI type, and target size."""

    records = {str(record["id"]): record for record in annotations}
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in benchmark_summary["items"]:
        sample_id = str(item["sample_id"])
        if not item.get("strict_correct") or sample_id not in records:
            continue
        record = records[sample_id]
        application = str(record["application"])
        if application not in applications:
            continue
        width, height = (int(value) for value in record["img_size"])
        x1, y1, x2, y2 = (float(value) for value in record["bbox"])
        case = {
            "id": sample_id,
            "instruction": str(record["instruction"]),
            "img_filename": str(record["img_filename"]),
            "image_size": [width, height],
            "bbox": [x1, y1, x2, y2],
            "application": application,
            "platform": str(record["platform"]),
            "ui_type": str(record["ui_type"]),
            "normalized_target_area": ((x2 - x1) * (y2 - y1)) / (width * height),
            "selection_native_strict_correct": True,
        }
        strata[(application, case["ui_type"])].append(case)

    selected: list[dict[str, Any]] = []
    expected = {(application, ui_type) for application in applications for ui_type in ("icon", "text")}
    if set(strata) != expected:
        missing = sorted(expected - set(strata))
        raise ValueError(f"missing native-success strata: {missing}")
    for key in sorted(expected):
        selected.extend(evenly_spaced(strata[key], per_stratum))
    return sorted(selected, key=lambda row: row["id"])


def scaled_size(size: tuple[int, int], linear_scale: float) -> tuple[int, int]:
    if not 0 < linear_scale <= 1:
        raise ValueError("linear scale must be within (0, 1]")
    return tuple(max(1, round(value * linear_scale)) for value in size)  # type: ignore[return-value]


def point_to_bbox_distance_normalized(
    point: dict[str, float],
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> float:
    """Distance to the nearest point in the target box on a 0-1000 canvas."""

    width, height = image_size
    x1, y1, x2, y2 = bbox
    px = point["x"] * 1000 / width
    py = point["y"] * 1000 / height
    nx1, nx2 = x1 * 1000 / width, x2 * 1000 / width
    ny1, ny2 = y1 * 1000 / height, y2 * 1000 / height
    dx = max(nx1 - px, 0.0, px - nx2)
    dy = max(ny1 - py, 0.0, py - ny2)
    return math.hypot(dx, dy)


def summarize_resolution_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_scale: dict[float, list[dict[str, Any]]] = defaultdict(list)
    by_id: dict[str, dict[float, dict[str, Any]]] = defaultdict(dict)
    for result in results:
        scale = float(result["linear_scale"])
        by_scale[scale].append(result)
        by_id[str(result["sample_id"])][scale] = result

    native_correct = {
        sample_id
        for sample_id, rows in by_id.items()
        if 1.0 in rows and rows[1.0].get("strict_correct")
    }
    summaries: list[dict[str, Any]] = []
    for scale in sorted(by_scale, reverse=True):
        rows = by_scale[scale]
        valid = [row for row in rows if row.get("format_valid")]
        correct = [row for row in rows if row.get("strict_correct")]
        paired = [row for row in rows if row["sample_id"] in native_correct]
        paired_correct = [row for row in paired if row.get("strict_correct")]
        distances = [float(row["point_to_box_distance_normalized"]) for row in valid]
        drifts = [float(row["drift_from_native_normalized"]) for row in rows if row.get("drift_from_native_normalized") is not None]
        summaries.append(
            {
                "linear_scale": scale,
                "area_fraction": scale * scale,
                "count": len(rows),
                "format_valid": len(valid),
                "strict_correct": len(correct),
                "selection_retention": len(correct) / len(rows) if rows else None,
                "paired_native_rerun_count": len(paired),
                "paired_native_rerun_retention": len(paired_correct) / len(paired) if paired else None,
                "median_point_to_box_distance_normalized": statistics.median(distances) if distances else None,
                "mean_point_to_box_distance_normalized": statistics.fmean(distances) if distances else None,
                "median_drift_from_native_normalized": statistics.median(drifts) if drifts else None,
            }
        )
    return {
        "native_rerun_correct_count": len(native_correct),
        "scales": summaries,
    }


def _validate_server_profile(metadata: dict[str, Any] | None, maximum_source_pixels: int) -> None:
    if not metadata or not isinstance(metadata.get("inference_configuration"), dict):
        raise ValueError("resolution ablation requires server provenance from /models")
    maximum = metadata["inference_configuration"].get("image_max_pixels")
    if not isinstance(maximum, int) or maximum < maximum_source_pixels:
        raise ValueError(
            "server image_max_pixels must exceed every native source image so the server does not add "
            f"an uncontrolled resize; need at least {maximum_source_pixels}, got {maximum}"
        )


def run_resolution_ablation(
    *,
    manifest_path: Path,
    annotation_root: Path,
    image_root: Path,
    base_url: str,
    model_id: str,
    output_dir: Path,
    resume: bool = True,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    annotation_root = annotation_root.expanduser().resolve()
    image_root = image_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]
    scales = tuple(float(value) for value in manifest["linear_scales"])
    if 1.0 not in scales:
        raise ValueError("resolution ablation must include a same-run native scale")
    maximum_source_pixels = max(int(case["image_size"][0]) * int(case["image_size"][1]) for case in cases)
    server_model = _server_model_metadata(base_url, model_id)
    _validate_server_profile(server_model, maximum_source_pixels)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "schema_version": 1,
        "experiment": "screenspot_success_retention_resolution_ablation_v1",
        "inference_protocol": SCREENSPOT_PROTOCOL,
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "base_url": base_url,
        "model_id": model_id,
        "server_model": server_model,
        "linear_scales": list(scales),
        "selected_count": len(cases),
        "request_count": len(cases) * len(scales),
        "interpolation": "PIL.Image.Resampling.LANCZOS",
        "coordinate_contract": "model emits 0-1000; scoring maps to the original annotated image",
    }
    _write_json(output_dir / "run.json", run_config)

    started = time.time()
    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=3600, trust_env=False) as client:
        for case in cases:
            sample = load_screenspot_sample(annotation_root, image_root, str(case["id"]))
            with Image.open(sample.image_path) as source:
                native_image = source.convert("RGB")
            if native_image.size != sample.image_size:
                raise ValueError(f"image size mismatch for {sample.id}: {native_image.size} != {sample.image_size}")
            native_click: dict[str, int] | None = None
            for scale in sorted(scales, reverse=True):
                scale_label = f"{scale:.4f}".rstrip("0").rstrip(".").replace(".", "p")
                result_path = output_dir / "items" / sample.id / f"scale-{scale_label}.json"
                if resume and result_path.is_file():
                    result = json.loads(result_path.read_text())
                else:
                    input_size = scaled_size(native_image.size, scale)
                    image = native_image if scale == 1.0 else native_image.resize(input_size, Image.Resampling.LANCZOS)
                    request = build_screenspot_request(image, sample.instruction, model_id)
                    response = client.post(f"{base_url.rstrip('/')}/chat/completions", json=request)
                    response.raise_for_status()
                    payload = response.json()
                    result = {
                        "sample_id": sample.id,
                        "application": sample.application,
                        "platform": sample.platform,
                        "ui_type": sample.ui_type,
                        "instruction": sample.instruction,
                        "linear_scale": scale,
                        "area_fraction": scale * scale,
                        "native_image_size": list(sample.image_size),
                        "input_image_size": list(input_size),
                        "input_pixel_count": input_size[0] * input_size[1],
                        "bbox": list(sample.bbox),
                        "selection_native_strict_correct": True,
                        "response": payload,
                    }
                    try:
                        click = parse_screenspot_click(payload, sample.image_size)
                        result["click"] = click
                        result["format_valid"] = True
                        result["strict_correct"] = point_hits_bbox(click["pixel"], sample.bbox)
                        result["point_to_box_distance_normalized"] = point_to_bbox_distance_normalized(
                            click["pixel"], sample.bbox, sample.image_size
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        result["format_valid"] = False
                        result["strict_correct"] = False
                        result["error"] = str(exc)
                        result["point_to_box_distance_normalized"] = None
                    _write_json(result_path, result)

                if scale == 1.0 and result.get("format_valid"):
                    native_click = result["click"]["normalized"]
                if native_click is not None and result.get("format_valid"):
                    click = result["click"]["normalized"]
                    result["drift_from_native_normalized"] = math.hypot(
                        float(click["x"] - native_click["x"]),
                        float(click["y"] - native_click["y"]),
                    )
                    _write_json(result_path, result)
                results.append(result)
                print(
                    json.dumps(
                        {
                            "sample_id": sample.id,
                            "scale": scale,
                            "strict_correct": result["strict_correct"],
                            "completed": len(results),
                            "total": len(cases) * len(scales),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    payload = {
        **run_config,
        "runtime_seconds": time.time() - started,
        "results": results,
        "summary": summarize_resolution_results(results),
    }
    _write_json(output_dir / "summary.json", payload)
    return payload
