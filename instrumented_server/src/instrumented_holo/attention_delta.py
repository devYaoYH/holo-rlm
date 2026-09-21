"""Memory-bounded teacher-forced attention comparison for Qwen3.5 and Holo.

The model forward uses its normal memory-efficient attention implementation.
Only the query rows that predict the declared oracle action tokens are
reconstructed afterward from captured full-attention-layer inputs. This keeps
native-resolution visual inputs practical on unified-memory Macs without
materializing every sequence-by-sequence attention matrix.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

from .activation_patching import ActivationPatchingRunner, PreparedCondition, overlapping_patch_indices
from .delta_lens import (
    DeltaLensCase,
    _input_signature,
    _release_engine,
    load_delta_lens_plan,
    override_plan_paths,
    require_plan_paths,
    select_plan_cases,
)
from .model import InstrumentedHolo, decode_data_image
from .settings import Settings


def _rotate_half(torch: Any, value: Any) -> Any:
    midpoint = value.shape[-1] // 2
    return torch.cat((-value[..., midpoint:], value[..., :midpoint]), dim=-1)


def _apply_rotary(torch: Any, value: Any, cos: Any, sin: Any) -> Any:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    rotated, passthrough = value[..., :rotary_dim], value[..., rotary_dim:]
    rotated = (rotated * cos) + (_rotate_half(torch, rotated) * sin)
    return torch.cat((rotated, passthrough), dim=-1)


def _slice_position_ids(position_ids: Any) -> Any:
    if position_ids.ndim == 3:
        return position_ids[:, :1, :]
    return position_ids[:1]


def _case_image_sizes(case: DeltaLensCase) -> tuple[tuple[int, int], ...]:
    sizes: list[tuple[int, int]] = []
    for message in case.request["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image_url":
                continue
            value = part.get("image_url", {})
            url = value if isinstance(value, str) else value.get("url")
            if isinstance(url, str):
                sizes.append(decode_data_image(url).size)
    return tuple(sizes)


def _normalized(values: list[list[float]]) -> list[list[float]]:
    total = sum(sum(frame) for frame in values)
    if total <= 0:
        return [[0.0 for _ in frame] for frame in values]
    return [[value / total for value in frame] for frame in values]


def _target_summary(
    case: DeltaLensCase,
    image_sizes: tuple[tuple[int, int], ...],
    grids: tuple[tuple[int, int], ...],
    maps: dict[str, list[list[float]]],
) -> dict[str, Any] | None:
    bbox = case.metadata.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4 or not grids:
        return None
    box = tuple(int(round(value)) for value in bbox)
    indices = overlapping_patch_indices(box, image_sizes[0], grids[0])
    total_patches = sum(math.prod(grid) for grid in grids)
    area_fraction = len(indices) / total_patches
    result: dict[str, Any] = {
        "image_index": 0,
        "bbox_pixels": list(bbox),
        "patch_indices": list(indices),
        "patch_area_fraction": area_fraction,
    }
    for method, frames in maps.items():
        mass = sum(frames[0][index] for index in indices)
        peak = max(range(len(frames[0])), key=frames[0].__getitem__)
        result[method] = {
            "target_mass": mass,
            "target_lift": mass / area_fraction if area_fraction else None,
            "peak_patch_index": peak,
            "peak_inside_target": peak in set(indices),
        }
    return result


def reconstruct_oracle_attention(
    engine: InstrumentedHolo,
    condition: PreparedCondition,
) -> dict[str, Any]:
    """Capture exact oracle-token attention rows without retaining full matrices."""

    torch = engine.torch
    model = engine.model
    assert torch is not None and model is not None
    language_model = model.model.language_model
    layer_types = list(language_model.config.layer_types)
    layer_indices = [index for index, kind in enumerate(layer_types) if kind == "full_attention"]
    captured: dict[int, Any] = {}

    with ExitStack() as stack:
        for layer_index in layer_indices:
            def capture_input(_module: Any, args: tuple[Any, ...], index: int = layer_index) -> None:
                captured[index] = args[0][0].detach().to("cpu")

            handle = language_model.layers[layer_index].register_forward_pre_hook(capture_input)
            stack.callback(handle.remove)
        with torch.inference_mode():
            outputs = language_model(
                input_ids=None,
                inputs_embeds=condition.inputs_embeds[:1],
                attention_mask=condition.attention_mask[:1],
                position_ids=_slice_position_ids(condition.position_ids),
                use_cache=False,
                return_dict=True,
            )
        del outputs

    if set(captured) != set(layer_indices):
        raise RuntimeError("did not capture every full-attention layer input")

    query_positions = tuple(int(value) for value in condition.prediction_positions[0])
    sequence_length = int(condition.inputs_embeds.shape[1])
    valid_keys = condition.attention_mask[0].to(dtype=torch.bool)
    position_ids = _slice_position_ids(condition.position_ids)
    with torch.inference_mode():
        cos, sin = language_model.rotary_emb(condition.inputs_embeds[:1], position_ids)

    direct_sums = [torch.zeros(len(span), dtype=torch.float64) for span in condition.image_positions]
    value_sums = [torch.zeros(len(span), dtype=torch.float64) for span in condition.image_positions]
    layer_rows: list[dict[str, Any]] = []

    for layer_index in layer_indices:
        attention = language_model.layers[layer_index].self_attn
        hidden = captured.pop(layer_index).to(engine.device).unsqueeze(0)
        selected = hidden[:, list(query_positions), :]
        input_shape = selected.shape[:-1]
        head_dim = int(attention.head_dim)
        with torch.inference_mode():
            query, _gate = torch.chunk(
                attention.q_proj(selected).view(*input_shape, -1, head_dim * 2),
                2,
                dim=-1,
            )
            query = attention.q_norm(query).transpose(1, 2)
            key = attention.k_norm(
                attention.k_proj(hidden).view(1, sequence_length, -1, head_dim)
            ).transpose(1, 2)
            value = attention.v_proj(hidden).view(1, sequence_length, -1, head_dim).transpose(1, 2)
            query = _apply_rotary(torch, query, cos[:, list(query_positions), :], sin[:, list(query_positions), :])
            key = _apply_rotary(torch, key, cos, sin)
            key = key.repeat_interleave(attention.num_key_value_groups, dim=1)
            value_norm = torch.linalg.vector_norm(value, ord=2, dim=-1).repeat_interleave(
                attention.num_key_value_groups, dim=1
            )
            scores = torch.matmul(query, key.transpose(2, 3)) * float(attention.scaling)
            key_index = torch.arange(sequence_length, device=scores.device)
            causal = key_index[None, :] <= torch.tensor(query_positions, device=scores.device)[:, None]
            allowed = causal & valid_keys.to(scores.device)[None, :]
            scores = scores.masked_fill(~allowed[None, None, :, :], torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores.float(), dim=-1)
            weighted = weights * value_norm[:, :, None, :].float()

            direct_image_mass = 0.0
            value_image_mass = 0.0
            for frame_index, span in enumerate(condition.image_positions):
                positions = torch.tensor(span, device=scores.device, dtype=torch.long)
                # MPS does not implement float64. Cast to float32 on-device,
                # transfer synchronously, then promote on CPU for accumulation.
                direct = (
                    weights.index_select(-1, positions)
                    .mean(dim=(0, 1, 2))
                    .float()
                    .cpu()
                    .to(torch.float64)
                )
                value_map = (
                    weighted.index_select(-1, positions)
                    .mean(dim=(0, 1, 2))
                    .float()
                    .cpu()
                    .to(torch.float64)
                )
                direct_sums[frame_index] += direct
                value_sums[frame_index] += value_map
                direct_image_mass += float(direct.sum())
                value_image_mass += float(value_map.sum())
            layer_rows.append(
                {
                    "layer": layer_index,
                    "direct_image_mass": direct_image_mass,
                    "value_norm_image_mass": value_image_mass,
                }
            )
        del hidden, selected, query, key, value, value_norm, scores, weights, weighted
        gc.collect()
        if engine.device == "mps":
            torch.mps.empty_cache()

    layer_count = len(layer_indices)
    raw_maps = {
        "direct_attention": [(values / layer_count).tolist() for values in direct_sums],
        "value_norm_attention": [(values / layer_count).tolist() for values in value_sums],
    }
    normalized_maps = {name: _normalized(values) for name, values in raw_maps.items()}
    return {
        "definition": (
            "mean over scored oracle-token query rows, heads, and all full-attention layers; "
            "maps are normalized across all image patches after aggregation"
        ),
        "candidate_label": condition.candidates[0].label,
        "query_positions": list(query_positions),
        "scored_token_ids": list(condition.scored_token_ids[0]),
        "full_attention_layers": layer_indices,
        "image_grids": [list(grid) for grid in condition.image_grids],
        "layer_image_mass": layer_rows,
        "maps": normalized_maps,
    }


def _model_case_result(engine: InstrumentedHolo, case: DeltaLensCase) -> dict[str, Any]:
    assert engine.processor is not None
    engine.processor.image_processor.size = {
        "shortest_edge": case.image_min_pixels,
        "longest_edge": case.image_max_pixels,
    }
    runner = ActivationPatchingRunner(engine, reduction=case.reduction)  # type: ignore[arg-type]
    condition = runner.prepare(
        name=case.id,
        messages=case.request["messages"],
        tools=case.request.get("tools", []),
        candidates=case.candidates,
        chat_template_kwargs=case.request.get("chat_template_kwargs"),
    )
    result = reconstruct_oracle_attention(engine, condition)
    image_sizes = _case_image_sizes(case)
    if len(image_sizes) != len(condition.image_grids):
        raise RuntimeError("source image count does not match the processor grids")
    target = _target_summary(case, image_sizes, condition.image_grids, result["maps"])
    return {
        "id": case.id,
        "protocol": case.protocol,
        "image_min_pixels": case.image_min_pixels,
        "image_max_pixels": case.image_max_pixels,
        "image_sizes": [list(size) for size in image_sizes],
        "alignment": _input_signature(condition),
        "target": target,
        **result,
    }


def compare_attention_results(base: dict[str, Any], tuned: dict[str, Any]) -> dict[str, Any]:
    base_cases = {case["id"]: case for case in base["cases"]}
    tuned_cases = {case["id"]: case for case in tuned["cases"]}
    if base_cases.keys() != tuned_cases.keys():
        raise ValueError("base and tuned attention runs contain different cases")
    cases = []
    for case_id, base_case in base_cases.items():
        tuned_case = tuned_cases[case_id]
        if base_case["alignment"] != tuned_case["alignment"]:
            raise ValueError(f"case {case_id!r} token/image alignment differs between checkpoints")
        methods: dict[str, Any] = {}
        for method, base_frames in base_case["maps"].items():
            tuned_frames = tuned_case["maps"][method]
            if [len(frame) for frame in base_frames] != [len(frame) for frame in tuned_frames]:
                raise ValueError(f"case {case_id!r} attention map shapes differ")
            methods[method] = {
                "base": base_frames,
                "tuned": tuned_frames,
                "delta": [
                    [tuned_value - base_value for base_value, tuned_value in zip(base_frame, tuned_frame, strict=True)]
                    for base_frame, tuned_frame in zip(base_frames, tuned_frames, strict=True)
                ],
            }
        target_delta: dict[str, Any] | None = None
        if base_case.get("target") and tuned_case.get("target"):
            target_delta = {}
            for method in methods:
                base_metric = base_case["target"][method]
                tuned_metric = tuned_case["target"][method]
                target_delta[method] = {
                    "base_target_mass": base_metric["target_mass"],
                    "tuned_target_mass": tuned_metric["target_mass"],
                    "delta_target_mass": tuned_metric["target_mass"] - base_metric["target_mass"],
                    "base_target_lift": base_metric["target_lift"],
                    "tuned_target_lift": tuned_metric["target_lift"],
                    "delta_target_lift": tuned_metric["target_lift"] - base_metric["target_lift"],
                    "base_peak_inside_target": base_metric["peak_inside_target"],
                    "tuned_peak_inside_target": tuned_metric["peak_inside_target"],
                }
        cases.append(
            {
                "id": case_id,
                "protocol": tuned_case["protocol"],
                "image_sizes": tuned_case["image_sizes"],
                "image_grids": tuned_case["image_grids"],
                "methods": methods,
                "target_delta": target_delta,
            }
        )
    return {
        "definition": "Holo3.1-4B minus Qwen3.5-4B on identical teacher-forced oracle tokens",
        "cases": cases,
    }


def run_manifest(
    manifest_path: Path,
    *,
    output_dir: Path,
    base_model: Path | None = None,
    tuned_model: Path | None = None,
    processor_path: Path | None = None,
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    plan = select_plan_cases(
        override_plan_paths(
            load_delta_lens_plan(manifest_path, require_processor=False),
            base_model=base_model,
            tuned_model=tuned_model,
            processor_path=processor_path,
        ),
        case_ids,
    )
    require_plan_paths(plan)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_outputs: dict[str, dict[str, Any]] = {}
    started = time.time()
    for model_spec in plan.models:
        final_path = output_dir / f"{model_spec.role}.json"
        if final_path.is_file():
            payload = json.loads(final_path.read_text())
            expected_cases = [case.id for case in plan.cases]
            if payload.get("role") != model_spec.role or [case["id"] for case in payload.get("cases", [])] != expected_cases:
                raise ValueError(f"existing {final_path.name} does not match the selected run")
            metadata_engine = InstrumentedHolo(
                replace(
                    Settings.from_environment(),
                    model_path=model_spec.path,
                    processor_path=plan.processor_path,
                    eager_attention=False,
                )
            )
            processor_metadata = metadata_engine.processor_metadata()
            runtime_size = payload.get("processor", {}).get("runtime_image_processor_size")
            if isinstance(runtime_size, dict):
                processor_metadata["runtime_image_processor_size"] = runtime_size
            if payload.get("processor") != processor_metadata:
                payload["processor"] = processor_metadata
                final_path.write_text(json.dumps(payload, indent=2) + "\n")
            model_outputs[model_spec.role] = payload
            print(json.dumps({"model": model_spec.label, "resumed": True, "complete": True}), flush=True)
            continue
        settings = replace(
            Settings.from_environment(),
            model_path=model_spec.path,
            processor_path=plan.processor_path,
            eager_attention=False,
        )
        engine = InstrumentedHolo(settings)
        try:
            engine.load()
            partial_path = output_dir / f"{model_spec.role}.partial.json"
            cases: list[dict[str, Any]] = []
            if partial_path.is_file():
                partial = json.loads(partial_path.read_text())
                if partial.get("role") != model_spec.role:
                    raise ValueError(f"existing {partial_path.name} belongs to a different model role")
                cases = list(partial.get("cases", []))
            completed = {case["id"] for case in cases}
            for case in plan.cases:
                if case.id in completed:
                    print(json.dumps({"model": model_spec.label, "case": case.id, "resumed": True}), flush=True)
                    continue
                case_started = time.time()
                result = _model_case_result(engine, case)
                result["runtime_seconds"] = time.time() - case_started
                cases.append(result)
                partial_payload = {
                    "schema_version": 1,
                    "label": model_spec.label,
                    "role": model_spec.role,
                    "model": engine.model_metadata(),
                    "processor": engine.processor_metadata(),
                    "cases": cases,
                }
                partial_path.write_text(json.dumps(partial_payload, indent=2) + "\n")
                print(json.dumps({"model": model_spec.label, "case": case.id, "complete": True}), flush=True)
            payload = {
                "schema_version": 1,
                "label": model_spec.label,
                "role": model_spec.role,
                "model": engine.model_metadata(),
                "processor": engine.processor_metadata(),
                "cases": cases,
            }
            model_outputs[model_spec.role] = payload
            temporary_path = output_dir / f".{model_spec.role}.json.tmp"
            temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
            temporary_path.replace(final_path)
            partial_path.unlink(missing_ok=True)
        finally:
            _release_engine(engine)
    comparison = compare_attention_results(model_outputs["base"], model_outputs["tuned"])
    result = {
        "schema_version": 1,
        "experiment": f"{plan.raw.get('name', plan.path.stem)}_attention_delta",
        "manifest": str(plan.path),
        "models": {role: payload["model"] for role, payload in model_outputs.items()},
        "comparison": comparison,
        "runtime_seconds": time.time() - started,
    }
    (output_dir / "attention-delta.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_dir / "manifest.snapshot.json").write_text(
        json.dumps(plan.raw, indent=2, sort_keys=True) + "\n"
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="holo-attention-delta")
    result.add_argument("manifest", type=Path)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--base-model", type=Path)
    result.add_argument("--tuned-model", type=Path)
    result.add_argument("--processor", type=Path)
    result.add_argument("--case-id", action="append")
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    result = run_manifest(
        args.manifest,
        output_dir=args.output,
        base_model=args.base_model,
        tuned_model=args.tuned_model,
        processor_path=args.processor,
        case_ids=args.case_id,
    )
    print(json.dumps({"output": str(args.output.resolve()), "cases": len(result["comparison"]["cases"])}, indent=2))


if __name__ == "__main__":
    main()
