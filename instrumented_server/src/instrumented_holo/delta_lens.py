"""Layerwise teacher-forced comparison between a base Qwen checkpoint and Holo.

The comparison deliberately holds the Holo processor, chat template, prompt,
images, tool schema, and candidate suffixes fixed.  Each checkpoint is loaded
sequentially so a local Metal machine never has to retain both 4B models.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .activation_patching import ActivationPatchingRunner, CandidateSequence, PreparedCondition
from .model import InstrumentedHolo
from .patching_cli import _candidates, _load_json_or_inline
from .settings import Settings


@dataclass(frozen=True)
class DeltaLensCase:
    id: str
    protocol: str
    request: dict[str, Any]
    candidates: tuple[CandidateSequence, CandidateSequence]
    reduction: str
    image_min_pixels: int
    image_max_pixels: int


@dataclass(frozen=True)
class DeltaLensModel:
    label: str
    role: str
    path: Path


@dataclass(frozen=True)
class DeltaLensPlan:
    path: Path
    raw: dict[str, Any]
    processor_path: Path
    models: tuple[DeltaLensModel, DeltaLensModel]
    cases: tuple[DeltaLensCase, ...]


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_delta_lens_plan(
    manifest_path: Path,
    *,
    require_models: bool = False,
    require_processor: bool = True,
) -> DeltaLensPlan:
    manifest_path = manifest_path.expanduser().resolve()
    raw = json.loads(manifest_path.read_text())
    if raw.get("schema_version") != 1:
        raise ValueError("delta-lens manifest must use schema_version 1")
    base = manifest_path.parent
    processor_path = _resolve(base, str(raw["processor_path"]))
    if require_processor and not processor_path.is_dir():
        raise FileNotFoundError(f"processor checkpoint is missing: {processor_path}")

    raw_models = raw.get("models")
    if not isinstance(raw_models, list) or len(raw_models) != 2:
        raise ValueError("delta-lens manifest must declare exactly two models")
    models = tuple(
        DeltaLensModel(
            label=str(item["label"]),
            role=str(item["role"]),
            path=_resolve(base, str(item["path"])),
        )
        for item in raw_models
    )
    if {model.role for model in models} != {"base", "tuned"}:
        raise ValueError("delta-lens model roles must be exactly base and tuned")
    if len({model.label for model in models}) != 2:
        raise ValueError("delta-lens model labels must be unique")
    if require_models:
        for model in models:
            if not model.path.is_dir():
                raise FileNotFoundError(f"{model.role} checkpoint is missing: {model.path}")

    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("delta-lens manifest must declare at least one case")
    cases: list[DeltaLensCase] = []
    case_ids: set[str] = set()
    for raw_case in raw_cases:
        case_id = str(raw_case["id"])
        if case_id in case_ids:
            raise ValueError(f"duplicate delta-lens case id: {case_id}")
        case_ids.add(case_id)
        reduction = str(raw_case.get("score_reduction", raw.get("score_reduction", "sum")))
        if reduction not in {"sum", "mean"}:
            raise ValueError(f"case {case_id!r} has an invalid score reduction")
        request = _load_json_or_inline(raw_case["request"], base)
        if not isinstance(request.get("messages"), list) or not request["messages"]:
            raise ValueError(f"case {case_id!r} request has no messages")
        cases.append(
            DeltaLensCase(
                id=case_id,
                protocol=str(raw_case["protocol"]),
                request=request,
                candidates=_candidates(raw_case["candidates"]),
                reduction=reduction,
                image_min_pixels=int(raw_case.get("image_min_pixels", raw.get("image_min_pixels", 65_536))),
                image_max_pixels=int(
                    raw_case.get("image_max_pixels", raw.get("image_max_pixels", 16_777_216))
                ),
            )
        )
        if cases[-1].image_max_pixels < cases[-1].image_min_pixels:
            raise ValueError(f"case {case_id!r} image_max_pixels is below image_min_pixels")
    return DeltaLensPlan(
        path=manifest_path,
        raw=raw,
        processor_path=processor_path,
        models=models,  # type: ignore[arg-type]
        cases=tuple(cases),
    )


def manifest_summary(plan: DeltaLensPlan) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "valid": True,
        "manifest": str(plan.path),
        "processor_path": str(plan.processor_path),
        "processor_present": plan.processor_path.is_dir(),
        "models": [
            {"label": model.label, "role": model.role, "path": str(model.path), "present": model.path.is_dir()}
            for model in plan.models
        ],
        "cases": [
            {
                "id": case.id,
                "protocol": case.protocol,
                "score_reduction": case.reduction,
                "image_min_pixels": case.image_min_pixels,
                "image_max_pixels": case.image_max_pixels,
                "candidate_labels": [candidate.label for candidate in case.candidates],
                "image_count": sum(
                    1
                    for message in case.request["messages"]
                    for part in (message.get("content") if isinstance(message.get("content"), list) else [])
                    if part.get("type") == "image_url"
                ),
            }
            for case in plan.cases
        ],
    }


def override_plan_paths(
    plan: DeltaLensPlan,
    *,
    base_model: Path | None = None,
    tuned_model: Path | None = None,
    processor_path: Path | None = None,
) -> DeltaLensPlan:
    """Apply explicit deployment paths without changing the frozen scientific manifest."""

    overrides = {"base": base_model, "tuned": tuned_model}
    models = tuple(
        replace(model, path=overrides[model.role].expanduser().resolve())
        if overrides[model.role] is not None
        else model
        for model in plan.models
    )
    processor = processor_path.expanduser().resolve() if processor_path is not None else plan.processor_path
    return replace(plan, processor_path=processor, models=models)  # type: ignore[arg-type]


def select_plan_cases(plan: DeltaLensPlan, case_ids: list[str] | None) -> DeltaLensPlan:
    """Select an explicit subset while retaining the manifest's stable case order."""

    if not case_ids:
        return plan
    requested = set(case_ids)
    if len(requested) != len(case_ids):
        raise ValueError("delta-lens case ids must not be repeated")
    available = {case.id for case in plan.cases}
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"unknown delta-lens case ids: {', '.join(missing)}")
    return replace(plan, cases=tuple(case for case in plan.cases if case.id in requested))


def require_plan_paths(plan: DeltaLensPlan) -> None:
    if not plan.processor_path.is_dir():
        raise FileNotFoundError(f"processor checkpoint is missing: {plan.processor_path}")
    for model in plan.models:
        if not model.path.is_dir():
            raise FileNotFoundError(f"{model.role} checkpoint is missing: {model.path}")


def _selected_hidden(output: Any, condition: PreparedCondition) -> tuple[Any, ...]:
    hidden = output[0] if isinstance(output, tuple) else output
    return tuple(
        hidden[batch_index, list(positions), :].detach().to("cpu")
        for batch_index, positions in enumerate(condition.prediction_positions)
    )


def _input_signature(condition: PreparedCondition) -> dict[str, Any]:
    input_bytes = condition.input_ids.detach().to("cpu").contiguous().numpy().tobytes()
    return {
        "input_ids_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "input_shape": list(condition.input_ids.shape),
        "prompt_length": condition.prompt_length,
        "image_grids": [list(grid) for grid in condition.image_grids],
        "candidate_token_ids": [list(candidate.token_ids) for candidate in condition.candidates],
        "scored_token_offsets": [list(candidate.scored_token_offsets) for candidate in condition.candidates],
        "scored_token_ids": [list(values) for values in condition.scored_token_ids],
    }


def _score_hidden(
    engine: InstrumentedHolo,
    condition: PreparedCondition,
    hidden_by_candidate: tuple[Any, ...],
    *,
    reduction: str,
) -> dict[str, Any]:
    torch = engine.torch
    model = engine.model
    processor = engine.processor
    assert torch is not None and model is not None and processor is not None
    normalizer = model.model.language_model.norm
    candidates: list[dict[str, Any]] = []
    for batch_index, (hidden_cpu, token_ids) in enumerate(
        zip(hidden_by_candidate, condition.scored_token_ids, strict=True)
    ):
        hidden = hidden_cpu.to(engine.device)
        with torch.inference_mode():
            logits = model.lm_head(normalizer(hidden)).float()
            targets = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
            log_probs = torch.log_softmax(logits, dim=-1)
            forced = log_probs.gather(1, targets[:, None]).squeeze(1)
            top_log_probs, top_ids = log_probs.max(dim=-1)
            ranks = 1 + (logits > logits.gather(1, targets[:, None])).sum(dim=-1)
        token_rows = []
        for token_index, token_id in enumerate(token_ids):
            top_id = int(top_ids[token_index].item())
            token_rows.append(
                {
                    "scored_index": token_index,
                    "token_id": int(token_id),
                    "token": processor.tokenizer.convert_ids_to_tokens(int(token_id)),
                    "decoded": processor.tokenizer.decode([int(token_id)], skip_special_tokens=False),
                    "log_prob_nats": float(forced[token_index].item()),
                    "rank": int(ranks[token_index].item()),
                    "top_token_id": top_id,
                    "top_token": processor.tokenizer.convert_ids_to_tokens(top_id),
                    "top_log_prob_nats": float(top_log_probs[token_index].item()),
                }
            )
        aggregate = forced.sum() if reduction == "sum" else forced.mean()
        candidates.append(
            {
                "label": condition.candidates[batch_index].label,
                "aggregate_log_prob_nats": float(aggregate.item()),
                "tokens": token_rows,
            }
        )
        del hidden, logits, log_probs, forced, top_log_probs, top_ids, ranks
    return {
        "candidates": candidates,
        "margin_nats": candidates[0]["aggregate_log_prob_nats"]
        - candidates[1]["aggregate_log_prob_nats"],
    }


def layerwise_score(
    engine: InstrumentedHolo,
    condition: PreparedCondition,
    *,
    reduction: str,
) -> dict[str, Any]:
    """Apply the checkpoint's final RMSNorm and LM head to every residual layer."""

    torch = engine.torch
    model = engine.model
    assert torch is not None and model is not None
    layers = model.model.language_model.layers
    captured: dict[int, tuple[Any, ...]] = {}
    with ExitStack() as stack:
        for layer_index, layer in enumerate(layers):
            def hook(_module: Any, _args: Any, output: Any, index: int = layer_index) -> None:
                captured[index] = _selected_hidden(output, condition)

            handle = layer.register_forward_hook(hook)
            stack.callback(handle.remove)
        with torch.inference_mode():
            outputs = model.model.language_model(
                input_ids=None,
                inputs_embeds=condition.inputs_embeds,
                attention_mask=condition.attention_mask,
                position_ids=condition.position_ids,
                use_cache=False,
                return_dict=True,
            )
        del outputs

    expected = set(range(len(layers)))
    if set(captured) != expected:
        raise RuntimeError(f"captured layers {sorted(captured)} do not match expected {sorted(expected)}")
    input_hidden = _selected_hidden(condition.inputs_embeds, condition)
    rows = [
        {
            "layer": -1,
            "label": "multimodal_input_embeddings",
            **_score_hidden(engine, condition, input_hidden, reduction=reduction),
        }
    ]
    for layer_index in range(len(layers)):
        rows.append(
            {
                "layer": layer_index,
                "label": f"residual_after_layer_{layer_index}",
                **_score_hidden(engine, condition, captured.pop(layer_index), reduction=reduction),
            }
        )
    return {"alignment": _input_signature(condition), "layers": rows}


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
    result = layerwise_score(engine, condition, reduction=case.reduction)
    return {
        "id": case.id,
        "protocol": case.protocol,
        "score_reduction": case.reduction,
        "image_min_pixels": case.image_min_pixels,
        "image_max_pixels": case.image_max_pixels,
        "candidate_text": [candidate.text for candidate in case.candidates],
        **result,
    }


def _release_engine(engine: InstrumentedHolo) -> None:
    torch = engine.torch
    engine.model = None
    engine.processor = None
    gc.collect()
    if torch is not None and engine.device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif torch is not None and engine.device == "cuda":
        torch.cuda.empty_cache()


def _token_delta(tuned: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    keys = ("token_id", "token", "decoded")
    if any(tuned[key] != base[key] for key in keys):
        raise ValueError("scored token identity differs between base and tuned checkpoints")
    return {
        **{key: tuned[key] for key in keys},
        "tuned_log_prob_nats": tuned["log_prob_nats"],
        "base_log_prob_nats": base["log_prob_nats"],
        "delta_log_prob_nats": tuned["log_prob_nats"] - base["log_prob_nats"],
        "tuned_rank": tuned["rank"],
        "base_rank": base["rank"],
    }


def compare_model_results(base: dict[str, Any], tuned: dict[str, Any]) -> dict[str, Any]:
    base_cases = {case["id"]: case for case in base["cases"]}
    tuned_cases = {case["id"]: case for case in tuned["cases"]}
    if base_cases.keys() != tuned_cases.keys():
        raise ValueError("base and tuned runs contain different case IDs")
    comparisons = []
    for case_id in base_cases:
        base_case, tuned_case = base_cases[case_id], tuned_cases[case_id]
        if base_case["alignment"] != tuned_case["alignment"]:
            raise ValueError(f"case {case_id!r} token/image alignment differs between checkpoints")
        if len(base_case["layers"]) != len(tuned_case["layers"]):
            raise ValueError(f"case {case_id!r} has a different layer count between checkpoints")
        layers = []
        for base_layer, tuned_layer in zip(base_case["layers"], tuned_case["layers"], strict=True):
            if base_layer["layer"] != tuned_layer["layer"]:
                raise ValueError(f"case {case_id!r} layer indices do not align")
            candidates = []
            for base_candidate, tuned_candidate in zip(
                base_layer["candidates"], tuned_layer["candidates"], strict=True
            ):
                if base_candidate["label"] != tuned_candidate["label"]:
                    raise ValueError(f"case {case_id!r} candidate labels do not align")
                if len(base_candidate["tokens"]) != len(tuned_candidate["tokens"]):
                    raise ValueError(f"case {case_id!r} scored token counts do not align")
                candidates.append(
                    {
                        "label": tuned_candidate["label"],
                        "base_log_prob_nats": base_candidate["aggregate_log_prob_nats"],
                        "tuned_log_prob_nats": tuned_candidate["aggregate_log_prob_nats"],
                        "delta_log_prob_nats": tuned_candidate["aggregate_log_prob_nats"]
                        - base_candidate["aggregate_log_prob_nats"],
                        "tokens": [
                            _token_delta(tuned_token, base_token)
                            for base_token, tuned_token in zip(
                                base_candidate["tokens"], tuned_candidate["tokens"], strict=True
                            )
                        ],
                    }
                )
            layers.append(
                {
                    "layer": tuned_layer["layer"],
                    "label": tuned_layer["label"],
                    "base_margin_nats": base_layer["margin_nats"],
                    "tuned_margin_nats": tuned_layer["margin_nats"],
                    "delta_margin_nats": tuned_layer["margin_nats"] - base_layer["margin_nats"],
                    "candidates": candidates,
                }
            )
        peak = max(layers, key=lambda item: abs(item["delta_margin_nats"]))
        comparisons.append(
            {
                "id": case_id,
                "protocol": tuned_case["protocol"],
                "score_reduction": tuned_case["score_reduction"],
                "alignment": tuned_case["alignment"],
                "summary": {
                    "peak_absolute_delta_margin_layer": peak["layer"],
                    "peak_absolute_delta_margin_nats": peak["delta_margin_nats"],
                    "final_delta_margin_nats": layers[-1]["delta_margin_nats"],
                },
                "layers": layers,
            }
        )
    return {
        "definition": "tuned minus base at identical teacher-forced tokens",
        "cases": comparisons,
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
        settings = replace(
            Settings.from_environment(),
            model_path=model_spec.path,
            processor_path=plan.processor_path,
        )
        engine = InstrumentedHolo(settings)
        try:
            engine.load()
            cases = []
            for case in plan.cases:
                case_started = time.time()
                result = _model_case_result(engine, case)
                result["runtime_seconds"] = time.time() - case_started
                cases.append(result)
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
            (output_dir / f"{model_spec.role}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        finally:
            _release_engine(engine)
    comparison = compare_model_results(model_outputs["base"], model_outputs["tuned"])
    result = {
        "schema_version": 1,
        "experiment": str(plan.raw.get("name", plan.path.stem)),
        "manifest": str(plan.path),
        "git_commit": plan.raw.get("git_commit"),
        "processor_path": str(plan.processor_path),
        "metric": {
            "name": "teacher_forced_layerwise_logit_delta",
            "formula": "log p_tuned(forced tokens) - log p_base(forced tokens)",
            "candidate_margin_delta": "(tuned candidate-0 minus candidate-1) - (base candidate-0 minus candidate-1)",
            "units": "nats",
            "lens": "each residual stream is projected through that checkpoint's final RMSNorm and tied LM head",
        },
        "models": {role: payload["model"] for role, payload in model_outputs.items()},
        "comparison": comparison,
        "runtime_seconds": time.time() - started,
    }
    (output_dir / "delta-lens.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output_dir / "manifest.snapshot.json").write_text(
        json.dumps(plan.raw, indent=2, sort_keys=True) + "\n"
    )
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="holo-delta-lens")
    result.add_argument("manifest", type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--base-model", type=Path, help="override the base-checkpoint path")
    result.add_argument("--tuned-model", type=Path, help="override the Holo-checkpoint path")
    result.add_argument("--processor", type=Path, help="override the shared Holo processor path")
    result.add_argument(
        "--case-id",
        action="append",
        help="run only this manifest case (repeat to select more than one)",
    )
    result.add_argument(
        "--validate-only",
        action="store_true",
        help="validate prompts, images, candidate suffixes, and model declarations without loading weights",
    )
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    plan = select_plan_cases(
        override_plan_paths(
            load_delta_lens_plan(args.manifest, require_processor=args.processor is None),
            base_model=args.base_model,
            tuned_model=args.tuned_model,
            processor_path=args.processor,
        ),
        args.case_id,
    )
    if args.validate_only:
        print(json.dumps(manifest_summary(plan), indent=2))
        return
    if args.output is None:
        raise SystemExit("--output is required unless --validate-only is used")
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
