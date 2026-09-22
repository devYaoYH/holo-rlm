"""Resolution-scaled ScreenSpot attribution with matched instruction controls."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from attribution import AttributionError, build_prompt_contrast, load_attribution, write_prompt_contrast_viewer
from attribution.contrast import parameter_steps
from PIL import Image

from .screenspot import ScreenSpotSample, load_screenspot_sample, run_screenspot_case
from .screenspot_benchmark import _server_model_metadata
from .screenspot_resolution import _validate_server_profile, _write_json, scaled_size


def scaled_sample(sample: ScreenSpotSample, image_path: Path, linear_scale: float) -> ScreenSpotSample:
    """Return a sample whose image geometry matches one explicit downsample."""

    width, height = scaled_size(sample.image_size, linear_scale)
    x1, y1, x2, y2 = sample.bbox
    return replace(
        sample,
        image_path=image_path,
        bbox=(x1 * linear_scale, y1 * linear_scale, x2 * linear_scale, y2 * linear_scale),
        image_size=(width, height),
    )


def summarize_resolution_attribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate attribution geometry separately for retained and lost clicks."""

    result: list[dict[str, Any]] = []
    for scale in sorted({float(row["linear_scale"]) for row in rows}, reverse=True):
        members = [row for row in rows if float(row["linear_scale"]) == scale]
        groups = {}
        for name, selected in (
            ("all", members),
            ("retained", [row for row in members if row["strict_correct"]]),
            ("lost", [row for row in members if not row["strict_correct"]]),
        ):
            metrics = [row["metrics"]["prompt_difference"] for row in selected]
            groups[name] = {
                "count": len(selected),
                "median_target_mass": statistics.median(row["target_mass"] for row in metrics) if metrics else None,
                "median_target_lift": statistics.median(row["target_lift"] for row in metrics) if metrics else None,
                "median_peak_distance_diagonal": statistics.median(
                    row["peak_distance_diagonal"] for row in metrics
                ) if metrics else None,
                "median_best_target_patch_rank": statistics.median(
                    row["best_target_patch_rank"] for row in metrics
                ) if metrics else None,
            }
        result.append({"linear_scale": scale, "groups": groups})
    return {"scales": result}


def _render_contrast(case: dict[str, Any], trace_root: Path, output_dir: Path) -> dict[str, Any]:
    target_run = next(run for run in case["runs"] if run["role"] == "target")
    control_runs = tuple(run for run in case["runs"] if run["role"].startswith("control-"))
    target_id = target_run.get("trace_id")
    if not isinstance(target_id, str):
        raise AttributionError("resolution attribution target is missing its trace ID")
    target = load_attribution(trace_root / target_id)
    controls = []
    compatible_runs = []
    excluded = []
    for run in control_runs:
        trace_id = run.get("trace_id")
        if not isinstance(trace_id, str):
            excluded.append({"instruction": run["instruction"], "trace_id": "", "reason": "missing trace ID"})
            continue
        control = load_attribution(trace_root / trace_id)
        try:
            parameter_steps(control, ("x", "y"))
        except AttributionError as exc:
            excluded.append({"instruction": run["instruction"], "trace_id": trace_id, "reason": str(exc)})
        else:
            controls.append(control)
            compatible_runs.append(run)
    if len(controls) < 3:
        raise AttributionError("resolution attribution requires at least three compatible controls")
    comparison = build_prompt_contrast(
        target,
        tuple(controls),
        control_instructions=tuple(run["instruction"] for run in compatible_runs),
    )
    click_record = target_run.get("click") or target_run.get("repaired_click")
    if not isinstance(click_record, dict):
        raise AttributionError("resolution attribution target has no usable click")
    click = click_record["pixel"]
    sample = case["sample"]
    return write_prompt_contrast_viewer(
        comparison,
        output_dir,
        sample_id=sample["id"],
        target_instruction=sample["instruction"],
        bbox=tuple(float(value) for value in sample["bbox"]),
        predicted_click=(float(click["x"]), float(click["y"])),
        correct=bool(target_run.get("grounding_correct", target_run.get("correct", False))),
        format_valid=bool(target_run.get("format_valid", True)),
        excluded_controls=tuple(excluded),
        include_layer_head_statistics=False,
    )


def run_resolution_attribution(
    *,
    manifest_path: Path,
    annotation_root: Path,
    image_root: Path,
    base_url: str,
    model_id: str,
    trace_root: Path,
    output_dir: Path,
    offset: int = 0,
    count: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Capture and analyze target/control saliency at every declared image scale."""

    manifest_path = manifest_path.expanduser().resolve()
    annotation_root = annotation_root.expanduser().resolve()
    image_root = image_root.expanduser().resolve()
    trace_root = trace_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    all_cases = list(manifest["cases"])
    cases = all_cases[offset : offset + count if count is not None else None]
    scales = tuple(float(value) for value in manifest["linear_scales"])
    controls_by_application = manifest["controls_by_application"]
    if len(scales) < 2 or 1.0 not in scales:
        raise ValueError("resolution attribution requires native scale plus at least one downsample")
    if any(len(values) != 4 for values in controls_by_application.values()):
        raise ValueError("every application must declare exactly four control instructions")
    samples = [load_screenspot_sample(annotation_root, image_root, str(case["id"])) for case in cases]
    maximum_source_pixels = max(sample.image_size[0] * sample.image_size[1] for sample in samples)
    server_model = _server_model_metadata(base_url, model_id)
    _validate_server_profile(server_model, maximum_source_pixels)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "schema_version": 1,
        "experiment": manifest["experiment"],
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "server_model": server_model,
        "model_id": model_id,
        "linear_scales": list(scales),
        "case_offset": offset,
        "case_count": len(cases),
        "prompt_count_per_condition": 5,
        "request_count": len(cases) * len(scales) * 5,
        "trace_profile": "attribution",
        "trace_generation_steps": 64,
        "interpolation": "PIL.Image.Resampling.LANCZOS",
    }
    _write_json(output_dir / "run.json", run_config)
    rows: list[dict[str, Any]] = []
    started = time.time()
    for sample in samples:
        controls = tuple(str(value) for value in controls_by_application[sample.application])
        with Image.open(sample.image_path) as source:
            native_image = source.convert("RGB")
        for scale in sorted(scales, reverse=True):
            scale_label = f"{scale:.4f}".rstrip("0").rstrip(".").replace(".", "p")
            condition_dir = output_dir / "items" / sample.id / f"scale-{scale_label}"
            image_path = condition_dir / "input.png"
            case_path = condition_dir / "case.json"
            contrast_dir = condition_dir / "contrast"
            analysis_path = contrast_dir / "analysis.json"
            if not image_path.is_file():
                condition_dir.mkdir(parents=True, exist_ok=True)
                size = scaled_size(native_image.size, scale)
                image = native_image if scale == 1.0 else native_image.resize(size, Image.Resampling.LANCZOS)
                image.save(image_path, format="PNG", optimize=True)
            condition_sample = scaled_sample(sample, image_path, scale)
            if not resume or not case_path.is_file():
                run_screenspot_case(
                    sample=condition_sample,
                    controls=controls,
                    base_url=base_url,
                    model_id=model_id,
                    output_dir=condition_dir,
                    trace_generation_steps=64,
                    trace_profile="attribution",
                )
            case = json.loads(case_path.read_text())
            if not resume or not analysis_path.is_file():
                _render_contrast(case, trace_root, contrast_dir)
            analysis = json.loads(analysis_path.read_text())
            target_run = next(run for run in case["runs"] if run["role"] == "target")
            rows.append(
                {
                    "sample_id": sample.id,
                    "application": sample.application,
                    "ui_type": sample.ui_type,
                    "linear_scale": scale,
                    "input_image_size": list(condition_sample.image_size),
                    "bbox": list(condition_sample.bbox),
                    "strict_correct": bool(target_run.get("grounding_correct", target_run.get("correct", False))),
                    "format_valid": bool(target_run.get("format_valid", True)),
                    "click": target_run.get("click") or target_run.get("repaired_click"),
                    "metrics": {
                        name: analysis["metrics"][name][0]
                        for name in ("raw", "prompt_difference")
                    },
                    "stability": analysis["stability"],
                    "trace_ids": analysis["trace_ids"],
                    "control_count": len(analysis["controls"]),
                    "excluded_controls": analysis["excluded_controls"],
                    "analysis": str(analysis_path),
                }
            )
            _write_json(
                output_dir / "progress.json",
                {"completed_conditions": len(rows), "total_conditions": len(cases) * len(scales)},
            )
            print(json.dumps({"sample_id": sample.id, "scale": scale, "completed": len(rows)}), flush=True)
    payload = {
        **run_config,
        "runtime_seconds": time.time() - started,
        "results": rows,
        "summary": summarize_resolution_attribution(rows),
    }
    _write_json(output_dir / "summary.json", payload)
    return payload
