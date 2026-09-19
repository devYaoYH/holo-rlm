"""Generate a self-contained browser viewer and preview overlays for attribution maps."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .pipeline import AttentionAttribution, AttributionError, InputFrameAttribution, load_attribution

_COLOR_STOPS = np.asarray(
    [
        (13, 8, 47),
        (68, 15, 118),
        (156, 42, 99),
        (230, 92, 48),
        (252, 180, 43),
        (252, 253, 191),
    ],
    dtype=np.float32,
)


def write_attribution_viewer(attribution: AttentionAttribution, output_dir: Path) -> dict[str, object]:
    """Write a portable HTML viewer, machine-readable summary, and aggregate PNGs."""

    output_dir = output_dir.expanduser().resolve()
    if output_dir == attribution.trace_path or attribution.trace_path in output_dir.parents:
        raise AttributionError("viewer output must stay outside the immutable trace directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    step_summaries = []
    previews = []
    preview_steps = (
        set(attribution.steps)
        if len(attribution.steps) <= 8
        else {attribution.steps[0], attribution.steps[-1]}
    )
    if attribution.value_weighted_rollout_maps is not None:
        default_method = "value_norm_rollout"
    elif attribution.rollout_maps is not None:
        default_method = "rollout"
    elif attribution.value_weighted_maps is not None:
        default_method = "value_norm"
    else:
        default_method = "attention"
    for step in attribution.steps:
        aggregate = attribution.aggregate(step, method=default_method)
        if step in preview_steps:
            for frame in attribution.frames:
                frame_aggregate = attribution.aggregate(step, method=default_method, frame_index=frame.index)
                frame_prefix = f"frame-{frame.index:03d}-" if attribution.frame_count > 1 else ""
                preview_name = (
                    f"saliency-{default_method.replace('_', '-')}-{frame_prefix}step-{step:03d}.png"
                )
                _write_overlay(frame.image_path, frame_aggregate, output_dir / preview_name)
                previews.append(preview_name)
        methods = {
            "attention": _method_step_summary(attribution, step, "attention"),
        }
        if attribution.value_weighted_maps is not None:
            methods["value_norm"] = _method_step_summary(attribution, step, "value_norm")
        if attribution.rollout_maps is not None:
            methods["rollout"] = _method_step_summary(attribution, step, "rollout")
        if attribution.value_weighted_rollout_maps is not None:
            methods["value_norm_rollout"] = _method_step_summary(attribution, step, "value_norm_rollout")
        step_summaries.append(
            {
                "step": step,
                "generated_token_id": attribution.generated_token_ids[attribution.steps.index(step)],
                "generated_token_piece": attribution.generated_token_pieces[attribution.steps.index(step)],
                "methods": methods,
            }
        )
    for span in attribution.generated_spans:
        for frame in attribution.frames:
            span_maps = [
                attribution.aggregate(step, method=default_method, frame_index=frame.index)
                for step in span.steps
            ]
            if not span_maps:
                continue
            aggregate = np.mean(np.stack(span_maps), axis=0)
            frame_prefix = f"frame-{frame.index:03d}-" if attribution.frame_count > 1 else ""
            preview_name = f"saliency-{default_method.replace('_', '-')}-{frame_prefix}{span.id}.png"
            _write_overlay(frame.image_path, aggregate, output_dir / preview_name)
            previews.append(preview_name)

    summary = {
        "schema_version": 2,
        "methods": {
            "attention": "last-query direct attention over image-token keys",
            **(
                {
                    "value_norm": (
                        "per-head attention multiplied by the corresponding KV-head value-vector L2 norm, "
                        "then normalized across all attended keys"
                    )
                }
                if attribution.value_weighted_maps is not None
                else {}
            ),
            **(
                {"rollout": "residual-aware cross-layer rollout across every captured full-attention block"}
                if attribution.rollout_maps is not None
                else {}
            ),
            **(
                {
                    "value_norm_rollout": (
                        "value-norm-corrected, residual-aware cross-layer rollout across every captured "
                        "full-attention block"
                    )
                }
                if attribution.value_weighted_rollout_maps is not None
                else {}
            ),
        },
        "default_method": default_method,
        "trace_id": attribution.trace_path.name,
        "frame_count": attribution.frame_count,
        "image": {
            "source": attribution.image_path.name,
            "width": attribution.image_size[0],
            "height": attribution.image_size[1],
        },
        "frames": [
            {
                "index": frame.index,
                "source": frame.image_path.name,
                "width": frame.image_size[0],
                "height": frame.image_size[1],
                "patch_grid": {
                    "rows": frame.layout.rows,
                    "columns": frame.layout.columns,
                    "prompt_position_start": int(frame.layout.image_positions[0]),
                    "prompt_position_end": int(frame.layout.image_positions[-1]),
                },
                "role": "current" if frame.index == attribution.frame_count - 1 else "history",
            }
            for frame in attribution.frames
        ],
        "patch_grid": {
            "rows": attribution.layout.rows,
            "columns": attribution.layout.columns,
            "merge_size": attribution.layout.merge_size,
            "raw_grid_thw": attribution.layout.raw_grid_thw,
            "source": attribution.layout.source,
            "prompt_position_start": int(attribution.layout.image_positions[0]),
            "prompt_position_end": int(attribution.layout.image_positions[-1]),
        },
        "captured_steps": list(attribution.steps),
        "attention_layer_ordinals": list(attribution.layer_ordinals),
        "transformer_layer_indices": list(attribution.layer_indices),
        "head_count": attribution.head_count,
        "generated_spans": [
            {
                "id": span.id,
                "label": span.label,
                "parameter": span.parameter,
                "value": span.value,
                "steps": list(span.steps),
                "token_count": len(span.steps),
            }
            for span in attribution.generated_spans
        ],
        "steps": step_summaries,
        "warnings": list(attribution.warnings),
        "baseline_subtraction": {
            "available": len(attribution.steps) > 1,
            "reference": "mean of captured generated-token maps strictly before the target span",
            "matched_dimensions": ["input_frame", "attribution_method", "layer", "head"],
            "display": "positive residuals only",
        },
        "interpretation": (
            "These are routing-based diagnostics, not causal feature attributions. The value-norm correction "
            "suppresses high-attention paths whose value vectors carry little magnitude. Optional baseline "
            "subtraction compares a target with only earlier generated tokens in this completion; it is not a "
            "separately measured resting-state control and never uses future tokens."
        ),
    }
    (output_dir / "attribution.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    payload = _viewer_payload(attribution, summary)
    serialized = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = _HTML.replace("__ATTRIBUTION_PAYLOAD__", serialized)
    (output_dir / "viewer.html").write_text(html)
    return {
        "trace": str(attribution.trace_path),
        "viewer": str(output_dir / "viewer.html"),
        "summary": str(output_dir / "attribution.json"),
        "previews": [str(output_dir / name) for name in previews],
        "patch_grid": [attribution.layout.rows, attribution.layout.columns],
        "steps": len(attribution.steps),
        "layers": len(attribution.layer_ordinals),
        "heads": attribution.head_count,
        "frames": attribution.frame_count,
        "value_norm_correction": attribution.value_weighted_maps is not None,
        "warnings": list(attribution.warnings),
    }


def write_trajectory_viewer(
    bundle_path: Path,
    trace_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Write a multi-frame action viewer plus one detailed viewer per completed trace."""

    bundle_path = bundle_path.expanduser().resolve()
    trace_root = trace_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    action_paths = sorted((bundle_path / "actions").glob("*.json"))
    if not action_paths:
        raise AttributionError(f"trajectory has no completed actions: {bundle_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, object]] = []
    warnings: list[str] = []
    for action_path in action_paths:
        action_record = json.loads(action_path.read_text())
        raw_output = action_record.get("raw_output", {})
        trace_id = raw_output.get("instrumented_trace_id")
        if not isinstance(trace_id, str):
            warnings.append(f"{action_path.name} has no instrumented trace ID")
            continue
        trace_path = (trace_root / trace_id).resolve()
        if trace_path != trace_root and trace_root not in trace_path.parents:
            raise AttributionError(f"trace ID in {action_path.name} resolves outside the trace root")
        try:
            attribution = load_attribution(trace_path)
        except AttributionError as exc:
            warnings.append(f"{trace_id}: {exc}")
            continue
        detail_dir = output_dir / trace_id
        write_attribution_viewer(attribution, detail_dir)
        frames.append(
            _trajectory_frame_payload(
                len(frames),
                action_path.stem,
                action_record.get("normalized", {}),
                attribution,
                f"{trace_id}/viewer.html",
            )
        )
    if not frames:
        raise AttributionError("trajectory has no completed actions with usable attention traces")

    annotations_path = bundle_path / "annotations.json"
    annotations = json.loads(annotations_path.read_text()) if annotations_path.is_file() else {}
    summary = {
        "schema_version": 1,
        "trajectory_id": bundle_path.name,
        "frame_count": len(frames),
        "actions": [frame["action"] for frame in frames],
        "trace_ids": [frame["traceId"] for frame in frames],
        "captured_click": any(frame["action"].get("action") == "click" for frame in frames),
        "task_success": annotations.get("task_success"),
        "terminal_reason": annotations.get("terminal_reason"),
        "target": annotations.get("target"),
        "warnings": warnings,
    }
    (output_dir / "trajectory.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    serialized = json.dumps(
        {
            "trajectoryId": bundle_path.name,
            "frames": frames,
            "warnings": warnings,
            "taskSuccess": summary["task_success"],
            "terminalReason": summary["terminal_reason"],
            "target": summary["target"],
        },
        separators=(",", ":"),
    ).replace("</", "<\\/")
    (output_dir / "trajectory.html").write_text(_TRAJECTORY_HTML.replace("__TRAJECTORY_PAYLOAD__", serialized))
    return {
        "trajectory": str(bundle_path),
        "viewer": str(output_dir / "trajectory.html"),
        "summary": str(output_dir / "trajectory.json"),
        "frames": len(frames),
        "captured_click": summary["captured_click"],
        "trace_viewers": [str(output_dir / str(frame["viewerPath"])) for frame in frames],
        "warnings": warnings,
    }


def _trajectory_frame_payload(
    frame_index: int,
    action_index: str,
    action: object,
    attribution: AttentionAttribution,
    viewer_path: str,
) -> dict[str, object]:
    action = action if isinstance(action, dict) else {}
    if attribution.value_weighted_rollout_maps is not None:
        method = "value_norm_rollout"
        method_label = "Value-norm rollout"
    elif attribution.rollout_maps is not None:
        method = "rollout"
        method_label = "Cross-layer rollout"
    elif attribution.value_weighted_maps is not None:
        method = "value_norm"
        method_label = "Attention x ‖V‖₂"
    else:
        method = "attention"
        method_label = "Direct attention"

    targets: list[dict[str, object]] = []
    for span in attribution.generated_spans:
        maps = [attribution.aggregate(step, method=method) for step in span.steps]
        if not maps:
            continue
        targets.append(
            _trajectory_target_payload(
                span.id,
                span.label,
                np.mean(np.stack(maps), axis=0),
                len(span.steps),
            )
        )
    if not targets:
        step = attribution.steps[-1]
        targets.append(
            _trajectory_target_payload(
                "last-token",
                f"last generated token (step {step})",
                attribution.aggregate(step, method=method),
                1,
            )
        )
    image_bytes = attribution.image_path.read_bytes()
    suffix = attribution.image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return {
        "index": frame_index,
        "actionIndex": action_index,
        "traceId": attribution.trace_path.name,
        "action": action,
        "method": method,
        "methodLabel": method_label,
        "imageDataUrl": f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}",
        "width": attribution.image_size[0],
        "height": attribution.image_size[1],
        "rows": attribution.layout.rows,
        "columns": attribution.layout.columns,
        "targets": targets,
        "viewerPath": viewer_path,
    }


def _trajectory_target_payload(
    target_id: str,
    label: str,
    saliency: np.ndarray,
    token_count: int,
) -> dict[str, object]:
    values = np.nan_to_num(saliency.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    scale = float(np.max(values))
    quantized = np.zeros_like(values, dtype=np.uint8)
    if scale > 0:
        quantized = np.clip(np.rint(values / scale * 255), 0, 255).astype(np.uint8)
    return {
        "id": target_id,
        "label": label,
        "tokenCount": token_count,
        "scale": scale,
        "imageMass": float(np.sum(values)),
        "mapsU8": base64.b64encode(np.ascontiguousarray(quantized).tobytes()).decode("ascii"),
    }


def _viewer_payload(attribution: AttentionAttribution, summary: dict[str, object]) -> dict[str, object]:
    frames = [
        _encoded_frame(frame, current=frame.index == attribution.frame_count - 1)
        for frame in attribution.frames
    ]
    tokens = []
    for step, token_id, piece in zip(
        attribution.steps,
        attribution.generated_token_ids,
        attribution.generated_token_pieces,
        strict=True,
    ):
        tokens.append({"step": step, "id": token_id, "piece": piece})
    return {
        "traceId": attribution.trace_path.name,
        "frames": frames,
        "steps": list(attribution.steps),
        "layerOrdinals": list(attribution.layer_ordinals),
        "layerIndices": list(attribution.layer_indices),
        "heads": attribution.head_count,
        "tokens": tokens,
        "spans": summary["generated_spans"],
        "methods": [
            {key: method[key] for key in ("id", "label", "supportsSlice")}
            for method in frames[-1]["methods"]
        ],
        "defaultMethod": summary["default_method"],
        "summaries": summary["steps"],
        "warnings": list(attribution.warnings),
    }


def _encoded_frame(frame: InputFrameAttribution, *, current: bool) -> dict[str, object]:
    image_path = frame.image_path
    image_bytes = image_path.read_bytes()
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    methods = [
        _encoded_method("attention", "Direct attention", frame.maps, frame.layout.cells, supports_slice=True)
    ]
    if frame.value_weighted_maps is not None:
        methods.append(
            _encoded_method(
                "value_norm",
                "Attention x ‖V‖₂",
                frame.value_weighted_maps,
                frame.layout.cells,
                supports_slice=True,
            )
        )
    if frame.rollout_maps is not None:
        methods.append(
            _encoded_method(
                "rollout",
                "Cross-layer rollout",
                frame.rollout_maps,
                frame.layout.cells,
                supports_slice=False,
            )
        )
    if frame.value_weighted_rollout_maps is not None:
        methods.append(
            _encoded_method(
                "value_norm_rollout",
                "Value-norm rollout",
                frame.value_weighted_rollout_maps,
                frame.layout.cells,
                supports_slice=False,
            )
        )
    return {
        "index": frame.index,
        "label": f"Frame {frame.index} ({'current' if current else 'history'})",
        "source": image_path.name,
        "imageDataUrl": f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}",
        "width": frame.image_size[0],
        "height": frame.image_size[1],
        "rows": frame.layout.rows,
        "columns": frame.layout.columns,
        "methods": methods,
    }


def _encoded_method(
    method_id: str,
    label: str,
    maps: np.ndarray,
    cells: int,
    *,
    supports_slice: bool,
) -> dict[str, object]:
    blocks = maps.reshape(-1, cells)
    finite = np.nan_to_num(blocks, nan=0.0, posinf=0.0, neginf=0.0)
    scales = np.max(finite, axis=1).astype("<f4")
    divisors = np.where(scales > 0, scales, 1.0)[:, None]
    quantized = np.ascontiguousarray(np.clip(np.rint(finite / divisors * 255), 0, 255).astype(np.uint8))
    return {
        "id": method_id,
        "label": label,
        "supportsSlice": supports_slice,
        "mapsU8": base64.b64encode(quantized.tobytes()).decode("ascii"),
        "mapScalesF32": base64.b64encode(np.ascontiguousarray(scales).tobytes()).decode("ascii"),
    }


def _method_step_summary(
    attribution: AttentionAttribution,
    step: int,
    method: str,
) -> dict[str, object]:
    step_axis = attribution.steps.index(step)
    if method == "attention":
        source = attribution.maps
    elif method == "value_norm":
        source = attribution.value_weighted_maps
    elif method == "rollout":
        source = attribution.rollout_maps
    else:
        source = attribution.value_weighted_rollout_maps
    assert source is not None
    values = source[step_axis]
    if values.ndim == 2:
        image_attention_mass = float(np.nansum(values))
    else:
        image_attention_mass = float(np.nansum(values) / np.count_nonzero(~np.isnan(values[:, :, 0, 0])))
    aggregate = attribution.aggregate(step, method=method)
    flat = aggregate.reshape(-1)
    top = np.argsort(flat)[-5:][::-1]
    regions = []
    for index in top:
        row, column = divmod(int(index), attribution.layout.columns)
        regions.append(
            {
                "row": row,
                "column": column,
                "attention": float(flat[index]),
                "x_fraction": (column + 0.5) / attribution.layout.columns,
                "y_fraction": (row + 0.5) / attribution.layout.rows,
            }
        )
    return {
        "image_attention_mass": image_attention_mass,
        "top_patches": regions,
    }


def _write_overlay(image_path: Path, saliency: np.ndarray, output_path: Path) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGBA")
    normalized = _normalize_saliency(saliency)
    heatmap = _colorize(normalized)
    patch_image = Image.fromarray(heatmap, mode="RGBA")
    resized = patch_image.resize(image.size, Image.Resampling.BILINEAR)
    overlay = Image.alpha_composite(image, resized)
    overlay.save(output_path)


def _normalize_saliency(values: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite = np.maximum(finite, 0.0)
    ceiling = float(np.percentile(finite, 99))
    if not math.isfinite(ceiling) or ceiling <= 0:
        return np.zeros_like(finite)
    return np.power(np.clip(finite / ceiling, 0.0, 1.0), 0.62)


def _colorize(normalized: np.ndarray) -> np.ndarray:
    position = normalized * (_COLOR_STOPS.shape[0] - 1)
    lower = np.floor(position).astype(np.int32)
    upper = np.minimum(lower + 1, _COLOR_STOPS.shape[0] - 1)
    fraction = (position - lower)[..., None]
    rgb = _COLOR_STOPS[lower] * (1 - fraction) + _COLOR_STOPS[upper] * fraction
    alpha = np.clip(normalized * 210, 0, 210)[..., None]
    return np.concatenate([rgb, alpha], axis=-1).astype(np.uint8)


_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visual attention attribution</title>
<style>
:root{color-scheme:light dark;--bg:#f6f7f9;--panel:#fff;--fg:#17181b;--muted:#686c75;--line:#d9dce2;--accent:#6741d9;--soft:#eeebff;--shadow:0 18px 55px rgba(16,18,27,.12)}
@media(prefers-color-scheme:dark){:root{--bg:#111216;--panel:#1a1c21;--fg:#f3f3f5;--muted:#a7aab2;--line:#343740;--accent:#b9a5ff;--soft:#2d2744;--shadow:0 18px 55px rgba(0,0,0,.34)}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,input,select{font:inherit;color:inherit}main{max-width:1280px;margin:auto;padding:26px}.topbar{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:20px}.eyebrow{color:var(--accent);font-weight:600;letter-spacing:.09em;text-transform:uppercase;font-size:11px}.topbar h1{font-size:24px;line-height:1.15;margin:4px 0 5px;font-weight:600}.trace{color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}.method{max-width:560px;color:var(--muted);text-align:right}.stage{background:#08090b;border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:var(--shadow);position:relative;min-width:0}.frame-wrap{position:relative;line-height:0}.frame-wrap img{display:block;width:100%;height:auto}.frame-wrap canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}.hover-tip{position:absolute;display:none;pointer-events:none;background:rgba(12,12,15,.9);color:#fff;padding:7px 9px;border-radius:7px;font-size:12px;line-height:1.3;transform:translate(10px,-110%);white-space:nowrap}.stage-footer{display:flex;justify-content:space-between;gap:12px;align-items:center;background:#101115;color:#e7e7ea;padding:10px 13px;line-height:1.3}.token-label{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.patch-label{color:#a9acb4;font-size:12px}.below{display:grid;grid-template-columns:minmax(0,2fr) minmax(240px,1fr);gap:14px;margin-top:14px;align-items:start}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px}.panel h2{font-size:13px;margin:0 0 13px;font-weight:600}.control-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.control{min-width:0}.control-head{display:flex;justify-content:space-between;gap:10px;margin-bottom:6px}.control-head output{color:var(--muted);font-variant-numeric:tabular-nums}.control select,.control input[type=range]{width:100%}.control select{background:var(--bg);border:1px solid var(--line);border-radius:7px;padding:7px 9px}.control select:disabled,.control input:disabled{opacity:.55}.checks{display:flex;align-items:center;flex-wrap:wrap;gap:12px 22px;margin-top:14px}.check{display:flex;align-items:center;gap:8px}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px}.metric-label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.06em}.metric-value{font-size:20px;font-variant-numeric:tabular-nums;margin-top:2px}.token-strip{display:flex;gap:5px;overflow-x:auto;padding:12px 0 3px}.token-button{border:1px solid var(--line);background:transparent;border-radius:7px;padding:6px 8px;min-width:42px;cursor:pointer}.token-button[aria-pressed=true]{background:var(--soft);border-color:var(--accent);color:var(--accent)}.token-button small{display:block;color:var(--muted);font-size:9px}.note{font-size:12px;color:var(--muted);margin:0}.warning{margin-top:10px;color:#9a5b00}@media(prefers-color-scheme:dark){.warning{color:#f2ba62}}@media(max-width:880px){main{padding:16px}.below{grid-template-columns:1fr}.method{text-align:left}.topbar{display:block}.method{margin-top:10px}}@media(max-width:700px){.control-grid{grid-template-columns:1fr 1fr}}@media(max-width:480px){main{padding:10px}.control-grid{grid-template-columns:1fr}.stage-footer{align-items:flex-start;flex-direction:column}.topbar h1{font-size:20px}}
</style>
</head>
<body>
<main>
  <header class="topbar">
    <div><div class="eyebrow">Local model trace</div><h1>Visual attention attribution</h1><div class="trace" id="trace-id"></div></div>
    <div class="method" id="method-description"></div>
  </header>
  <section class="stage" aria-label="Input frame with attention saliency overlay">
    <div class="frame-wrap" id="frame-wrap"><img id="frame" alt="Captured model input frame"><canvas id="heatmap" role="img" aria-label="Attention saliency heatmap"></canvas><div class="hover-tip" id="hover-tip"></div></div>
    <div class="stage-footer"><span class="token-label" id="token-label"></span><span class="patch-label" id="patch-label"></span></div>
  </section>
  <div class="below">
    <section class="panel">
      <h2>Attribution controls</h2>
      <div class="control-grid">
        <div class="control"><div class="control-head"><label for="input-frame">Input frame</label></div><select id="input-frame"></select></div>
        <div class="control"><div class="control-head"><label for="focus">Generated target</label></div><select id="focus"><option value="token">Single output token</option></select></div>
        <div class="control"><div class="control-head"><label for="method">Attribution method</label></div><select id="method"></select></div>
        <div class="control"><div class="control-head"><label for="step">Generation step</label><output id="step-output"></output></div><input id="step" type="range" min="0" value="0" step="1"></div>
        <div class="control"><div class="control-head"><label for="layer">Transformer layer</label></div><select id="layer"><option value="all">All captured layers</option></select></div>
        <div class="control"><div class="control-head"><label for="head">Attention head</label></div><select id="head"><option value="all">All heads</option></select></div>
        <div class="control"><div class="control-head"><label for="opacity">Overlay opacity</label><output id="opacity-output">72%</output></div><input id="opacity" type="range" min="0" max="100" value="72"></div>
      </div>
      <div class="token-strip" id="token-strip" aria-label="Captured generated tokens"></div>
      <div class="checks">
        <label class="check"><input id="grid" type="checkbox"> Show patch boundaries</label>
        <label class="check"><input id="baseline" type="checkbox"> Subtract previous-token baseline</label>
      </div>
    </section>
    <div>
      <section class="panel metrics" aria-live="polite">
        <div><div class="metric-label" id="mass-label">Image allocation</div><div class="metric-value" id="mass">—</div></div>
        <div><div class="metric-label">Peak patch</div><div class="metric-value" id="peak">—</div></div>
      </section>
      <section class="panel"><h2>How to read this</h2><p class="note" id="reading-note"></p><p class="note warning" id="warning"></p><p class="note" style="margin-top:10px">Parameter views average their value-token maps, so long values do not receive extra weight merely for using more tokens. These remain routing diagnostics rather than causal proof.</p></section>
    </div>
  </div>
</main>
<script>
const data=__ATTRIBUTION_PAYLOAD__;
function decodeBase64(value){const binary=atob(value),result=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)result[i]=binary.charCodeAt(i);return result}
const frameMethodData=data.frames.map(item=>{const methods={};item.methods.forEach(method=>{const scaleBytes=decodeBase64(method.mapScalesF32);methods[method.id]={...method,maps:decodeBase64(method.mapsU8),scales:new Float32Array(scaleBytes.buffer)}});return methods});
const $=id=>document.getElementById(id);
const state={frameIndex:data.frames.length-1,stepAxis:0,focus:data.spans.length?data.spans[0].id:'token',layer:'all',head:'all',method:data.defaultMethod,opacity:.72,grid:false,baseline:false,baselineCount:null,current:null};
const frame=$('frame'),canvas=$('heatmap'),ctx=canvas.getContext('2d'),tip=$('hover-tip');
$('trace-id').textContent=data.traceId;
$('step').max=String(data.steps.length-1);
data.frames.forEach((item,index)=>{const option=document.createElement('option');option.value=String(index);option.textContent=item.label;$('input-frame').append(option)});$('input-frame').value=String(state.frameIndex);
data.methods.forEach(method=>{const option=document.createElement('option');option.value=method.id;option.textContent=method.label;$('method').append(option)});$('method').value=state.method;
data.spans.forEach(span=>{const option=document.createElement('option');option.value=span.id;option.textContent=`Parameter · ${span.label} · mean of ${span.token_count} token${span.token_count===1?'':'s'}`;$('focus').append(option)});$('focus').value=state.focus;
data.layerOrdinals.forEach((ordinal,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=`Layer ${data.layerIndices[i]} · capture ${ordinal}`;$('layer').append(option)});
for(let i=0;i<data.heads;i++){const option=document.createElement('option');option.value=String(i);option.textContent=`Head ${i}`;$('head').append(option)}
function tokenPiece(token){if(!token.piece)return `token ${token.id}`;return token.piece.replaceAll('Ġ','␠').replaceAll('Ċ','↵')}
data.tokens.forEach((token,i)=>{const button=document.createElement('button');button.type='button';button.className='token-button';button.dataset.index=String(i);button.innerHTML=`${escapeHtml(tokenPiece(token))}<small>${token.id}</small>`;button.addEventListener('click',()=>{$('focus').value='token';state.focus='token';$('step').value=String(i);state.stepAxis=i;render()});$('token-strip').append(button)});
function escapeHtml(value){return value.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function activeFrame(){return data.frames[state.frameIndex]}
function loadFrame(){const item=activeFrame();frame.src=item.imageDataUrl;$('patch-label').textContent=`${item.label} · ${item.columns} x ${item.rows} visual-token patches`;render()}
function selectedStepAxes(){if(state.focus==='token')return[state.stepAxis];const span=data.spans.find(item=>item.id===state.focus);return span?span.steps.map(step=>data.steps.indexOf(step)).filter(index=>index>=0):[state.stepAxis]}
function selectedMapAt(stepAxis){const item=activeFrame(),cells=item.rows*item.columns,out=new Float32Array(cells),source=frameMethodData[state.frameIndex][state.method];let count=0;if(!source.supportsSlice){const block=stepAxis,offset=block*cells,scale=source.scales[block];if(scale)for(let i=0;i<cells;i++)out[i]=source.maps[offset+i]/255*scale;return out}const layers=state.layer==='all'?[...data.layerOrdinals.keys()]:[Number(state.layer)];const heads=state.head==='all'?[...Array(data.heads).keys()]:[Number(state.head)];for(const l of layers)for(const h of heads){const block=(stepAxis*data.layerOrdinals.length+l)*data.heads+h,offset=block*cells,scale=source.scales[block];if(!scale)continue;for(let i=0;i<cells;i++)out[i]+=source.maps[offset+i]/255*scale;count++}if(count)for(let i=0;i<cells;i++)out[i]/=count;return out}
function selectedMap(){const axes=selectedStepAxes(),item=activeFrame(),cells=item.rows*item.columns,out=new Float32Array(cells);for(const axis of axes){const values=selectedMapAt(axis);for(let i=0;i<cells;i++)out[i]+=values[i]}if(axes.length>1)for(let i=0;i<cells;i++)out[i]/=axes.length;state.baselineCount=null;if(!state.baseline)return out;const startAxis=Math.min(...axes);state.baselineCount=startAxis;if(startAxis===0)return new Float32Array(cells);const baseline=new Float32Array(cells);for(let axis=0;axis<startAxis;axis++){const values=selectedMapAt(axis);for(let i=0;i<cells;i++)baseline[i]+=values[i]}for(let i=0;i<cells;i++)out[i]-=baseline[i]/startAxis;return out}
const stops=[[13,8,47],[68,15,118],[156,42,99],[230,92,48],[252,180,43],[252,253,191]];
function color(value){const p=value*(stops.length-1),lo=Math.floor(p),hi=Math.min(lo+1,stops.length-1),f=p-lo;return[0,1,2].map(i=>Math.round(stops[lo][i]*(1-f)+stops[hi][i]*f))}
function render(){
  const item=activeFrame(),values=selectedMap(),source=frameMethodData[state.frameIndex][state.method];
  const corrected=state.method.includes('value_norm'),rollout=state.method.includes('rollout');
  const span=data.spans.find(item=>item.id===state.focus),baselineSuffix=state.baseline?' above baseline':'';
  state.current=values;$('layer').disabled=!source.supportsSlice;$('head').disabled=!source.supportsSlice;$('step').disabled=Boolean(span);
  const positive=[...values].map(value=>Math.max(0,Number.isFinite(value)?value:0)).sort((a,b)=>a-b);
  const ceiling=positive[Math.min(positive.length-1,Math.floor(positive.length*.99))]||1;
  const small=document.createElement('canvas');small.width=item.columns;small.height=item.rows;
  const smallCtx=small.getContext('2d'),pixels=smallCtx.createImageData(item.columns,item.rows);
  let total=0,peak=-1,peakIndex=0;
  for(let i=0;i<values.length;i++){const raw=Math.max(0,values[i]||0);total+=raw;if(raw>peak){peak=raw;peakIndex=i}const value=Math.pow(Math.min(1,raw/ceiling),.62),rgb=color(value),o=i*4;pixels.data[o]=rgb[0];pixels.data[o+1]=rgb[1];pixels.data[o+2]=rgb[2];pixels.data[o+3]=Math.round(value*255)}
  smallCtx.putImageData(pixels,0,0);canvas.width=item.width;canvas.height=item.height;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.imageSmoothingEnabled=true;ctx.globalAlpha=state.opacity;ctx.drawImage(small,0,0,canvas.width,canvas.height);ctx.globalAlpha=1;
  if(state.grid){ctx.strokeStyle='rgba(255,255,255,.28)';ctx.lineWidth=Math.max(1,canvas.width/1600);for(let c=1;c<item.columns;c++){const x=c*canvas.width/item.columns;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke()}for(let r=1;r<item.rows;r++){const y=r*canvas.height/item.rows;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke()}}
  const token=data.tokens[state.stepAxis],methodLabel=source.label;
  $('step-output').textContent=span?`${span.token_count} tokens averaged`:`${state.stepAxis+1} / ${data.steps.length}`;
  $('token-label').textContent=(span?`${item.label} · parameter ${span.label} · token-length normalized mean`:`${item.label} · predicting ${tokenPiece(token)} · ID ${token.id}`)+baselineSuffix;
  $('mass-label').textContent=state.baseline?'Positive differential':(rollout?'Frame influence':'Frame allocation');$('mass').textContent=`${(total*100).toFixed(2)}%`;$('peak').textContent=`${peakIndex%item.columns}, ${Math.floor(peakIndex/item.columns)}`;
  const baseMethod=rollout?(corrected?'Value-norm-corrected attention is composed across all captured full-attention blocks with residual paths.':'Head-mean attention is composed across all captured full-attention blocks with residual paths.'):(corrected?'Attention is scaled by each key value vector\'s L₂ norm and renormalized across all keys before image patches are selected.':'Direct last-query attention over image-token keys; no value magnitude correction.');
  $('method-description').textContent=baseMethod+(state.baseline?' The mean map of strictly previous generated tokens is subtracted; future tokens are never used and only positive residuals are shown.':'');
  $('reading-note').textContent=state.baseline?(state.baselineCount?`Baseline is the mean of the ${state.baselineCount} generated-token map${state.baselineCount===1?'':'s'} strictly before this target begins, matched by frame, method, layer, and head. Below-baseline residuals are transparent.`:'This target begins at the first captured token, so no causal previous-token baseline exists and the differential overlay is empty.'):(span?`This map is the arithmetic mean of the ${span.token_count} token maps that spell the parameter value, preventing longer values from dominating.`:(rollout?'Brighter regions accumulated more attention flow through the full-attention blocks.':'Brighter regions received more direct attention while the selected output token was predicted.'));
  $('heatmap').setAttribute('aria-label',`${item.label}; ${methodLabel} saliency${baselineSuffix}; positive allocation ${(total*100).toFixed(2)} percent; peak patch ${peakIndex%item.columns}, ${Math.floor(peakIndex/item.columns)}`);
  const activeAxes=new Set(selectedStepAxes());document.querySelectorAll('.token-button').forEach((button,i)=>button.setAttribute('aria-pressed',String(activeAxes.has(i))))
}
$('input-frame').addEventListener('change',e=>{state.frameIndex=Number(e.target.value);loadFrame()});$('step').addEventListener('input',e=>{state.stepAxis=Number(e.target.value);render()});$('focus').addEventListener('change',e=>{state.focus=e.target.value;render()});$('method').addEventListener('change',e=>{state.method=e.target.value;render()});$('layer').addEventListener('change',e=>{state.layer=e.target.value;render()});$('head').addEventListener('change',e=>{state.head=e.target.value;render()});$('opacity').addEventListener('input',e=>{state.opacity=Number(e.target.value)/100;$('opacity-output').textContent=`${e.target.value}%`;render()});$('grid').addEventListener('change',e=>{state.grid=e.target.checked;render()});$('baseline').addEventListener('change',e=>{state.baseline=e.target.checked;render()});
canvas.addEventListener('pointermove',event=>{if(!state.current)return;const item=activeFrame(),rect=canvas.getBoundingClientRect(),column=Math.min(item.columns-1,Math.floor((event.clientX-rect.left)/rect.width*item.columns)),row=Math.min(item.rows-1,Math.floor((event.clientY-rect.top)/rect.height*item.rows)),value=state.current[row*item.columns+column];tip.style.display='block';tip.style.left=`${event.clientX-rect.left}px`;tip.style.top=`${event.clientY-rect.top}px`;tip.textContent=`${item.label} · patch ${column}, ${row} · ${(value*100).toFixed(4)}% ${state.baseline?'differential':'allocation'}`});canvas.addEventListener('pointerleave',()=>tip.style.display='none');
$('warning').textContent=data.warnings.join(' · ');frame.addEventListener('load',render);loadFrame();
</script>
</body>
</html>
'''


_TRAJECTORY_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trajectory attention attribution</title>
<style>
:root{color-scheme:light dark;--bg:#f4f5f7;--panel:#fff;--fg:#17181b;--muted:#686c75;--line:#d9dce2;--accent:#6741d9;--soft:#eeebff;--shadow:0 18px 55px rgba(16,18,27,.12)}
@media(prefers-color-scheme:dark){:root{--bg:#111216;--panel:#1a1c21;--fg:#f3f3f5;--muted:#a7aab2;--line:#343740;--accent:#b9a5ff;--soft:#2d2744;--shadow:0 18px 55px rgba(0,0,0,.34)}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,input,select{font:inherit;color:inherit}main{max-width:1280px;margin:auto;padding:26px}.eyebrow{color:var(--accent);font-weight:650;letter-spacing:.09em;text-transform:uppercase;font-size:11px}h1{font-size:25px;line-height:1.15;margin:4px 0 6px}.subtitle{color:var(--muted);max-width:780px}.stage{margin-top:20px;background:#08090b;border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:var(--shadow)}.frame-wrap{position:relative;line-height:0}.frame-wrap img{display:block;width:100%;height:auto}.frame-wrap canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.stage-footer{display:flex;justify-content:space-between;gap:16px;background:#101115;color:#e7e7ea;padding:11px 14px;line-height:1.35}.stage-footer span:last-child{color:#aaadb5;text-align:right}.controls{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px;margin-top:14px}.control-grid{display:grid;grid-template-columns:1.2fr 1.2fr 1fr;gap:14px}.control label{display:block;font-size:12px;color:var(--muted);margin-bottom:6px}.control select,.control input{width:100%}.control select{background:var(--bg);border:1px solid var(--line);border-radius:7px;padding:8px 9px}.filmstrip{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin-top:14px}.card{border:1px solid var(--line);background:var(--panel);border-radius:11px;padding:0;overflow:hidden;text-align:left;cursor:pointer}.card[aria-current=true]{border-color:var(--accent);box-shadow:0 0 0 2px var(--soft)}.card img{display:block;width:100%;height:auto}.card-copy{padding:10px 11px}.card-title{font-weight:650}.card-meta{color:var(--muted);font-size:11px;margin-top:3px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.detail-row{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-top:14px;color:var(--muted)}.detail-row a{color:var(--accent);text-decoration:none;font-weight:600}.warning{color:#9a5b00}.method-note{margin:8px 0 0;color:var(--muted);font-size:12px}@media(prefers-color-scheme:dark){.warning{color:#f2ba62}}@media(max-width:720px){main{padding:14px}.control-grid{grid-template-columns:1fr}.stage-footer,.detail-row{align-items:flex-start;flex-direction:column}.stage-footer span:last-child{text-align:left}}
</style>
</head>
<body>
<main>
  <header><div class="eyebrow">Multi-frame local trajectory</div><h1>Action attention across frames</h1><div class="subtitle" id="subtitle"></div></header>
  <section class="stage" aria-label="Selected pre-action frame with parameter saliency">
    <div class="frame-wrap"><img id="frame" alt="Captured pre-action model input"><canvas id="heatmap" role="img" aria-label="Attention saliency heatmap"></canvas></div>
    <div class="stage-footer"><span id="action-label"></span><span id="trace-label"></span></div>
  </section>
  <section class="controls">
    <div class="control-grid">
      <div class="control"><label for="frame-select">Pre-action frame</label><select id="frame-select"></select></div>
      <div class="control"><label for="target-select">Generated parameter target</label><select id="target-select"></select></div>
      <div class="control"><label for="opacity">Overlay opacity · <output id="opacity-output">72%</output></label><input id="opacity" type="range" min="0" max="100" value="72"></div>
    </div>
    <p class="method-note" id="method-note"></p>
  </section>
  <section class="filmstrip" id="filmstrip" aria-label="Trajectory frames"></section>
  <div class="detail-row"><span id="allocation"></span><a id="detail-link">Inspect every generated token in this frame →</a></div>
  <p class="warning" id="warning"></p>
</main>
<script>
const data=__TRAJECTORY_PAYLOAD__;
const $=id=>document.getElementById(id),frame=$('frame'),canvas=$('heatmap'),ctx=canvas.getContext('2d');
const state={frame:0,target:0,opacity:.72};
function decode(value){const raw=atob(value),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);return bytes}
function actionText(action){if(action.action==='scroll')return`scroll · delta_y ${action.delta_y}`;if(action.action==='click')return`click · (${action.x}, ${action.y})`;return Object.entries(action).map(([key,value])=>`${key} ${value}`).join(' · ')}
function escapeHtml(value){return String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
data.frames.forEach((item,index)=>{const option=document.createElement('option');option.value=String(index);option.textContent=`${index+1}. ${actionText(item.action)}`;$('frame-select').append(option);const card=document.createElement('button');card.type='button';card.className='card';card.innerHTML=`<img src="${item.imageDataUrl}" alt="Pre-action frame ${index+1}"><div class="card-copy"><div class="card-title">${escapeHtml(actionText(item.action))}</div><div class="card-meta">${escapeHtml(item.traceId)}</div></div>`;card.addEventListener('click',()=>{state.frame=index;$('frame-select').value=String(index);loadFrame()});$('filmstrip').append(card)});
function loadFrame(){const item=data.frames[state.frame];state.target=0;frame.src=item.imageDataUrl;$('target-select').replaceChildren();item.targets.forEach((target,index)=>{const option=document.createElement('option');option.value=String(index);option.textContent=`${target.label} · mean of ${target.tokenCount} token${target.tokenCount===1?'':'s'}`;$('target-select').append(option)});$('action-label').textContent=`Frame ${state.frame+1} · ${actionText(item.action)}`;$('trace-label').textContent=`${item.methodLabel} · ${item.columns} x ${item.rows} patches`;$('detail-link').href=item.viewerPath;document.querySelectorAll('.card').forEach((card,index)=>card.setAttribute('aria-current',String(index===state.frame)));render()}
const stops=[[13,8,47],[68,15,118],[156,42,99],[230,92,48],[252,180,43],[252,253,191]];
function color(value){const p=value*(stops.length-1),lo=Math.floor(p),hi=Math.min(lo+1,stops.length-1),f=p-lo;return[0,1,2].map(i=>Math.round(stops[lo][i]*(1-f)+stops[hi][i]*f))}
function render(){const item=data.frames[state.frame],target=item.targets[state.target],values=decode(target.mapsU8),small=document.createElement('canvas');small.width=item.columns;small.height=item.rows;const smallCtx=small.getContext('2d'),pixels=smallCtx.createImageData(item.columns,item.rows);for(let i=0;i<values.length;i++){const value=Math.pow(values[i]/255,.62),rgb=color(value),offset=i*4;pixels.data[offset]=rgb[0];pixels.data[offset+1]=rgb[1];pixels.data[offset+2]=rgb[2];pixels.data[offset+3]=Math.round(value*255)}smallCtx.putImageData(pixels,0,0);canvas.width=item.width;canvas.height=item.height;ctx.clearRect(0,0,item.width,item.height);ctx.globalAlpha=state.opacity;ctx.drawImage(small,0,0,item.width,item.height);ctx.globalAlpha=1;if(item.action.action==='click'){ctx.strokeStyle='#fff';ctx.lineWidth=3;const x=item.action.x/1000*item.width,y=item.action.y/1000*item.height;ctx.beginPath();ctx.moveTo(x-14,y);ctx.lineTo(x+14,y);ctx.moveTo(x,y-14);ctx.lineTo(x,y+14);ctx.stroke()}$('allocation').textContent=`${target.label} · image influence ${(target.imageMass*100).toFixed(2)}%`;$('method-note').textContent=`${item.methodLabel}; the selected parameter map is averaged over its value tokens. Each frame is the exact image seen immediately before that action.`;canvas.setAttribute('aria-label',`${target.label} saliency for ${actionText(item.action)}`)}
$('frame-select').addEventListener('change',event=>{state.frame=Number(event.target.value);loadFrame()});$('target-select').addEventListener('change',event=>{state.target=Number(event.target.value);render()});$('opacity').addEventListener('input',event=>{state.opacity=Number(event.target.value)/100;$('opacity-output').textContent=`${event.target.value}%`;render()});
const outcome=data.taskSuccess===true?'task succeeded':data.taskSuccess===false?`task failed · ${data.terminalReason}`:'capture in progress';$('subtitle').textContent=`${data.trajectoryId} · ${data.frames.length} completed action frame${data.frames.length===1?'':'s'} · ${outcome}. Select a frame and generated parameter, then open its detailed viewer to scrub every captured output token.`;$('warning').textContent=data.warnings.join(' · ');frame.addEventListener('load',render);loadFrame();
</script>
</body>
</html>
'''
