"""Cross-request attribution baselines and spatial diagnostics."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import AttentionAttribution, AttributionError


@dataclass(frozen=True)
class PromptContrast:
    """One target trace contrasted with matched same-image instruction traces."""

    target: AttentionAttribution
    controls: tuple[AttentionAttribution, ...]
    method: str
    parameters: tuple[str, ...]
    target_steps: tuple[int, ...]
    control_steps: tuple[tuple[int, ...], ...]
    raw_maps: tuple[np.ndarray, ...]
    causal_maps: tuple[np.ndarray, ...]
    prompt_baseline_maps: tuple[np.ndarray, ...]
    prompt_difference_maps: tuple[np.ndarray, ...]
    control_instructions: tuple[str, ...]
    stability: dict[str, float | None]


def parameter_steps(attribution: AttentionAttribution, parameters: tuple[str, ...]) -> tuple[int, ...]:
    """Return ordered token steps for one or more structured-output values."""

    selected = [
        step
        for span in attribution.generated_spans
        if span.parameter in parameters
        for step in span.steps
    ]
    result = tuple(sorted(set(selected)))
    missing = [name for name in parameters if not any(span.parameter == name for span in attribution.generated_spans)]
    if missing:
        raise AttributionError(f"generated output has no parameter span(s): {', '.join(missing)}")
    if not result:
        raise AttributionError("selected generated parameters contain no captured value tokens")
    return result


def normalized_frame_maps(
    attribution: AttentionAttribution,
    steps: tuple[int, ...],
    *,
    method: str,
) -> tuple[np.ndarray, ...]:
    """L1-normalize one request over every input-image patch as one allocation."""

    maps = tuple(
        np.maximum(
            np.nan_to_num(
                attribution.aggregate_steps(steps, method=method, frame_index=frame.index),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            0.0,
        ).astype(np.float32)
        for frame in attribution.frames
    )
    total = float(sum(np.sum(values) for values in maps))
    if total <= 0:
        return tuple(np.zeros_like(values) for values in maps)
    return tuple(values / total for values in maps)


def build_prompt_contrast(
    target: AttentionAttribution,
    controls: tuple[AttentionAttribution, ...],
    *,
    method: str = "value_norm_rollout",
    parameters: tuple[str, ...] = ("x", "y"),
    control_instructions: tuple[str, ...] | None = None,
) -> PromptContrast:
    """Subtract the mean normalized attribution from matched control prompts."""

    if not controls:
        raise AttributionError("prompt-ensemble attribution requires at least one control trace")
    for control in controls:
        _validate_compatible_trace(target, control)
    if control_instructions is None:
        control_instructions = tuple(_instruction_from_trace(control.trace_path) for control in controls)
    if len(control_instructions) != len(controls):
        raise AttributionError("control instruction count does not match control trace count")

    target_steps = parameter_steps(target, parameters)
    control_steps = tuple(parameter_steps(control, parameters) for control in controls)
    raw_maps = normalized_frame_maps(target, target_steps, method=method)
    control_maps = tuple(
        normalized_frame_maps(control, steps, method=method)
        for control, steps in zip(controls, control_steps, strict=True)
    )
    prompt_baseline_maps = tuple(
        np.mean(np.stack([maps[index] for maps in control_maps]), axis=0, dtype=np.float32)
        for index in range(target.frame_count)
    )
    prompt_difference_maps = tuple(
        raw - baseline for raw, baseline in zip(raw_maps, prompt_baseline_maps, strict=True)
    )
    causal_maps = tuple(
        target.causal_difference(target_steps, method=method, frame_index=frame.index).astype(np.float32)
        for frame in target.frames
    )
    return PromptContrast(
        target=target,
        controls=controls,
        method=method,
        parameters=parameters,
        target_steps=target_steps,
        control_steps=control_steps,
        raw_maps=raw_maps,
        causal_maps=causal_maps,
        prompt_baseline_maps=prompt_baseline_maps,
        prompt_difference_maps=prompt_difference_maps,
        control_instructions=control_instructions,
        stability=_ensemble_stability(raw_maps, control_maps),
    )


def spatial_metrics(
    values: np.ndarray,
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> dict[str, float | bool | list[int]]:
    """Measure positive attribution alignment with a pixel-space target box."""

    positive = np.maximum(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0).astype(np.float64)
    total = float(positive.sum())
    distribution = positive / total if total > 0 else np.zeros_like(positive)
    overlap = _bbox_patch_overlap(values.shape, bbox, image_size)
    target_mass = float(np.sum(distribution * overlap))
    target_patches = overlap > 0
    if np.any(target_patches):
        best_target_score = float(np.max(distribution[target_patches]))
        best_target_patch_rank: int | None = 1 + int(np.sum(distribution > best_target_score))
    else:
        best_target_patch_rank = None
    width, height = image_size
    x1, y1, x2, y2 = bbox
    area_fraction = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)
    lift = target_mass / area_fraction if area_fraction > 0 else 0.0
    peak_index = int(np.argmax(positive)) if positive.size else 0
    peak_row, peak_column = divmod(peak_index, values.shape[1])
    peak_x = (peak_column + 0.5) * width / values.shape[1]
    peak_y = (peak_row + 0.5) * height / values.shape[0]
    target_x = (x1 + x2) / 2
    target_y = (y1 + y2) / 2
    peak_distance = math.hypot(peak_x - target_x, peak_y - target_y) / math.hypot(width, height)
    nonzero = distribution[distribution > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / math.log(distribution.size)) if nonzero.size else 0.0
    return {
        "positive_mass": total,
        "target_mass": target_mass,
        "target_area_fraction": area_fraction,
        "target_lift": lift,
        "best_target_patch_rank": best_target_patch_rank,
        "target_patch_count": int(np.sum(target_patches)),
        "peak_patch": [peak_column, peak_row],
        "peak_inside_target": bool(overlap[peak_row, peak_column] > 0),
        "peak_distance_diagonal": peak_distance,
        "normalized_entropy": entropy,
    }


def layer_head_statistics(
    contrast: PromptContrast,
    bbox: tuple[float, float, float, float],
    *,
    frame_index: int = 0,
) -> dict[str, Any]:
    """Score every full-attention layer and head against the annotated target."""

    target = contrast.target
    image_size = target.frames[frame_index].image_size
    rows: list[dict[str, Any]] = []
    for layer_ordinal, layer_index in zip(target.layer_ordinals, target.layer_indices, strict=True):
        for head in range(target.head_count):
            raw = target.aggregate_steps(
                contrast.target_steps,
                layer_ordinal=layer_ordinal,
                head=head,
                method="value_norm",
                frame_index=frame_index,
            )
            causal = target.causal_difference(
                contrast.target_steps,
                layer_ordinal=layer_ordinal,
                head=head,
                method="value_norm",
                frame_index=frame_index,
            )
            control_values = []
            for control, steps in zip(contrast.controls, contrast.control_steps, strict=True):
                value = control.aggregate_steps(
                    steps,
                    layer_ordinal=layer_ordinal,
                    head=head,
                    method="value_norm",
                    frame_index=frame_index,
                )
                control_values.append(_l1_normalize(value))
            prompt_difference = _l1_normalize(raw) - np.mean(np.stack(control_values), axis=0)
            rows.append(
                {
                    "layer_ordinal": layer_ordinal,
                    "transformer_layer": layer_index,
                    "head": head,
                    "raw": spatial_metrics(raw, bbox, image_size),
                    "causal": spatial_metrics(causal, bbox, image_size),
                    "prompt_difference": spatial_metrics(prompt_difference, bbox, image_size),
                }
            )
    layer_rows = []
    for ordinal, index in zip(target.layer_ordinals, target.layer_indices, strict=True):
        members = [row for row in rows if row["layer_ordinal"] == ordinal]
        layer_rows.append(
            {
                "layer_ordinal": ordinal,
                "transformer_layer": index,
                "raw_mean_target_lift": float(np.mean([row["raw"]["target_lift"] for row in members])),
                "causal_mean_target_lift": float(np.mean([row["causal"]["target_lift"] for row in members])),
                "prompt_mean_target_lift": float(
                    np.mean([row["prompt_difference"]["target_lift"] for row in members])
                ),
                "best_prompt_head": max(members, key=lambda row: row["prompt_difference"]["target_lift"])["head"],
                "best_prompt_target_lift": max(
                    row["prompt_difference"]["target_lift"] for row in members
                ),
            }
        )
    return {"heads": rows, "layers": layer_rows}


def _validate_compatible_trace(target: AttentionAttribution, control: AttentionAttribution) -> None:
    if target.frame_count != control.frame_count:
        raise AttributionError("target and control traces have different input-frame counts")
    if target.layer_indices != control.layer_indices or target.head_count != control.head_count:
        raise AttributionError("target and control traces have different attention layer/head layouts")
    for target_frame, control_frame in zip(target.frames, control.frames, strict=True):
        if target_frame.image_size != control_frame.image_size:
            raise AttributionError("target and control traces have different image dimensions")
        if (target_frame.layout.rows, target_frame.layout.columns) != (
            control_frame.layout.rows,
            control_frame.layout.columns,
        ):
            raise AttributionError("target and control traces have different patch grids")
        if _sha256(target_frame.image_path) != _sha256(control_frame.image_path):
            raise AttributionError("target and control traces do not contain the exact same input image")
    for name in ("model.json", "processor.json"):
        if _canonical_json(target.trace_path / name) != _canonical_json(control.trace_path / name):
            raise AttributionError(f"target and control traces use different {name.removesuffix('.json')} metadata")


def _bbox_patch_overlap(
    shape: tuple[int, ...],
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> np.ndarray:
    rows, columns = shape
    width, height = image_size
    x1, y1, x2, y2 = bbox
    patch_width = width / columns
    patch_height = height / rows
    result = np.zeros((rows, columns), dtype=np.float64)
    for row in range(rows):
        py1, py2 = row * patch_height, (row + 1) * patch_height
        for column in range(columns):
            px1, px2 = column * patch_width, (column + 1) * patch_width
            intersection = max(0.0, min(x2, px2) - max(x1, px1)) * max(
                0.0, min(y2, py2) - max(y1, py1)
            )
            result[row, column] = intersection / (patch_width * patch_height)
    return result


def _l1_normalize(values: np.ndarray) -> np.ndarray:
    finite = np.maximum(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0).astype(np.float32)
    total = float(finite.sum())
    return finite / total if total > 0 else np.zeros_like(finite)


def _ensemble_stability(
    target_maps: tuple[np.ndarray, ...],
    control_maps: tuple[tuple[np.ndarray, ...], ...],
) -> dict[str, float | None]:
    target = np.concatenate([value.reshape(-1) for value in target_maps])
    controls = [np.concatenate([value.reshape(-1) for value in maps]) for maps in control_maps]
    if len(controls) < 2:
        return {"mean_leave_one_out_cosine": None, "minimum_leave_one_out_cosine": None}
    full = target - np.mean(np.stack(controls), axis=0)
    scores = []
    for index in range(len(controls)):
        subset = controls[:index] + controls[index + 1 :]
        candidate = target - np.mean(np.stack(subset), axis=0)
        denominator = float(np.linalg.norm(full) * np.linalg.norm(candidate))
        scores.append(float(np.dot(full, candidate) / denominator) if denominator else 0.0)
    return {
        "mean_leave_one_out_cosine": float(np.mean(scores)),
        "minimum_leave_one_out_cosine": float(np.min(scores)),
    }


def _instruction_from_trace(trace_path: Path) -> str:
    request_path = trace_path / "request.json"
    if not request_path.is_file():
        return trace_path.name
    request = json.loads(request_path.read_text())
    for message in request.get("messages", []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return str(part.get("text", "")).removeprefix("Instruction: ")
    return trace_path.name


def _canonical_json(path: Path) -> str:
    if not path.is_file():
        return ""
    return json.dumps(json.loads(path.read_text()), sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_signed_map(values: np.ndarray) -> dict[str, Any]:
    """Quantize a signed map symmetrically for a portable browser payload."""

    finite = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.max(np.abs(finite)))
    codes = np.zeros_like(finite, dtype=np.int8)
    if scale > 0:
        codes = np.clip(np.rint(finite / scale * 127), -127, 127).astype(np.int8)
    return {
        "scale": scale,
        "valuesI8": base64.b64encode(np.ascontiguousarray(codes).tobytes()).decode("ascii"),
    }
