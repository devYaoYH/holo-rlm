#!/usr/bin/env python3
"""Sample low-resolution Holo coordinates with a confidence-width guardrail."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import LogitsProcessor

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
from scripts.run_coordinate_beam_pilot import (  # noqa: E402
    parse_coordinate_text,
    source_point,
    summarize_candidates,
)


class ConfidenceWidthGuardrail(LogitsProcessor):
    """Force top-1 when fewer than ``min_width`` tokens lie near the best token.

    Width is the number of tokens whose raw log probability is within
    ``logprob_delta`` nats of the top token. High-certainty structural positions
    should have width one and therefore decode greedily. Wider distributions
    remain open to the standard temperature, top-p, and top-k samplers.
    """

    def __init__(self, *, logprob_delta: float, min_width: int) -> None:
        self.logprob_delta = logprob_delta
        self.min_width = min_width
        self.step_records: list[dict[str, Any]] = []

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        log_probs = scores.float().log_softmax(dim=-1)
        top_values, top_indices = log_probs.max(dim=-1)
        widths = (log_probs >= (top_values[:, None] - self.logprob_delta)).sum(dim=-1)
        force_greedy = widths < self.min_width
        self.step_records.append(
            {
                "batch_rows": int(scores.shape[0]),
                "forced_greedy_rows": int(force_greedy.sum().item()),
                "open_sampling_rows": int((~force_greedy).sum().item()),
                "minimum_width": int(widths.min().item()),
                "maximum_width": int(widths.max().item()),
                "mean_width": float(widths.float().mean().item()),
            }
        )
        if not bool(force_greedy.any()):
            return scores
        guarded = scores.clone()
        row_indices = force_greedy.nonzero(as_tuple=False).squeeze(-1)
        guarded[row_indices] = -math.inf
        guarded[row_indices, top_indices[row_indices]] = scores[row_indices, top_indices[row_indices]]
        return guarded

    def summary(self) -> dict[str, Any]:
        total_rows = sum(record["batch_rows"] for record in self.step_records)
        forced_rows = sum(record["forced_greedy_rows"] for record in self.step_records)
        open_rows = sum(record["open_sampling_rows"] for record in self.step_records)
        return {
            "generation_steps": len(self.step_records),
            "evaluated_sequence_steps": total_rows,
            "forced_greedy_sequence_steps": forced_rows,
            "open_sampling_sequence_steps": open_rows,
            "forced_greedy_fraction": forced_rows / total_rows if total_rows else None,
            "minimum_observed_width": min((record["minimum_width"] for record in self.step_records), default=None),
            "maximum_observed_width": max((record["maximum_width"] for record in self.step_records), default=None),
            "mean_step_mean_width": (
                sum(record["mean_width"] for record in self.step_records) / len(self.step_records)
                if self.step_records
                else None
            ),
        }


def strict_coordinate_json(text: str) -> dict[str, int] | None:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"x", "y"}:
        return None
    if type(payload["x"]) is not int or type(payload["y"]) is not int:
        return None
    if not 0 <= payload["x"] <= 1000 or not 0 <= payload["y"] <= 1000:
        return None
    return {"x": payload["x"], "y": payload["y"]}


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
    for sample_index, sample_id in enumerate(args.sample_ids):
        sample = load_screenspot_sample(args.annotations, args.images, sample_id)
        with Image.open(sample.image_path) as source:
            source_image = source.convert("RGB")
        low_size = (
            max(1, round(source_image.width * args.scale)),
            max(1, round(source_image.height * args.scale)),
        )
        low_image = source_image.resize(low_size, Image.Resampling.LANCZOS)
        request = build_screenspot_request(low_image, sample.instruction, args.model_id)
        inputs = engine._prepare_inputs(  # noqa: SLF001 - reuse the official engine path.
            request["messages"],
            request.get("tools"),
            enable_thinking=False,
        )
        stop_strings = generation_stop_strings(request.get("tools"), request)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        candidates: list[dict[str, Any]] = []
        guardrail_batches: list[dict[str, Any]] = []
        sample_started = time.perf_counter()

        for batch_start in range(0, args.sample_count, args.batch_size):
            batch_count = min(args.batch_size, args.sample_count - batch_start)
            batch_seed = args.seed + sample_index * 10_000 + batch_start
            engine.torch.manual_seed(batch_seed)
            guardrail = ConfidenceWidthGuardrail(
                logprob_delta=args.width_logprob_delta,
                min_width=args.min_sampling_width,
            )
            generate_kwargs: dict[str, Any] = {
                **inputs,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": True,
                "num_beams": 1,
                "num_return_sequences": batch_count,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "logits_processor": [guardrail],
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            if stop_strings:
                generate_kwargs["stop_strings"] = list(stop_strings)
                generate_kwargs["tokenizer"] = engine.processor.tokenizer
            with engine.torch.inference_mode():
                generated = engine.model.generate(**generate_kwargs)
            generated_tokens = generated.sequences[:, prompt_tokens:]
            texts = engine.processor.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            guardrail_batches.append({"seed": batch_seed, "sample_count": batch_count, **guardrail.summary()})
            for text in texts:
                strict_coordinate = strict_coordinate_json(text)
                coordinate = strict_coordinate or parse_coordinate_text(text)
                candidate: dict[str, Any] = {
                    "sample_index": len(candidates),
                    "seed_batch": batch_seed,
                    "text": text,
                    "coordinate": coordinate,
                    "strict_format_valid": strict_coordinate is not None,
                    "recoverable_coordinate": coordinate is not None,
                }
                if coordinate is not None:
                    point = source_point(coordinate, sample.image_size)
                    candidate["source_pixel"] = point
                    candidate["hits_oracle"] = point_hits_bbox(point, sample.bbox)
                candidates.append(candidate)

        sample_runtime = time.perf_counter() - sample_started
        spread = summarize_candidates(candidates)
        strict_valid_count = sum(candidate["strict_format_valid"] for candidate in candidates)
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
            "strict_format_valid_count": strict_valid_count,
            "strict_format_valid_fraction": strict_valid_count / len(candidates),
            "candidates": candidates,
            "spread": spread,
            "guardrail_batches": guardrail_batches,
        }
        results.append(result)
        print(
            json.dumps(
                {
                    "sample_id": sample.id,
                    "runtime_seconds": round(sample_runtime, 2),
                    "strict_valid": strict_valid_count,
                    "sample_count": len(candidates),
                    "unique": spread["unique_count"],
                    "rms_radius": spread["rms_radius"],
                    "hit_count": spread["hit_count"],
                }
            ),
            flush=True,
        )

    valid_spreads = [row["spread"]["rms_radius"] for row in results if row["spread"]["rms_radius"] is not None]
    output = {
        "schema_version": 1,
        "experiment": "coordinate_stochastic_width_guardrail_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "inference_protocol": SCREENSPOT_PROTOCOL,
        "method": "independent ancestral coordinate sampling with confidence-width greedy guardrail",
        "model": engine.model_metadata(),
        "configuration": {
            "sample_ids": args.sample_ids,
            "sample_count_per_case": args.sample_count,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "linear_scale": args.scale,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "width_logprob_delta_nats": args.width_logprob_delta,
            "minimum_sampling_width": args.min_sampling_width,
            "guardrail": (
                "force top-1 whenever fewer than minimum_sampling_width tokens lie within "
                "width_logprob_delta_nats of the best raw next-token log probability"
            ),
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "image_min_pixels": args.image_min_pixels,
            "image_max_pixels": args.image_max_pixels,
            "eager_attention": False,
            "resize": "PIL Lanczos before the checkpoint processor",
            "coordinate_contract": "strict JSON object with integer x/y in normalized 0-1000 space",
        },
        "aggregate": {
            "case_count": len(results),
            "candidate_count": sum(len(row["candidates"]) for row in results),
            "strict_format_valid_count": sum(row["strict_format_valid_count"] for row in results),
            "hit_count": sum(row["spread"]["hit_count"] for row in results),
            "mean_unique_coordinate_count": sum(row["spread"]["unique_count"] for row in results) / len(results),
            "mean_rms_radius": sum(valid_spreads) / len(valid_spreads) if valid_spreads else None,
            "runtime_seconds": time.perf_counter() - started,
        },
        "results": results,
        "claim_boundary": (
            "Selected two-case feasibility pilot. This tests stochastic decoding behavior and JSON-format preservation, "
            "not benchmark accuracy, calibrated uncertainty, or adaptive-crop efficacy."
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
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "local-results" / "coordinate-stochastic-widthguard-v1")
    parser.add_argument("--sample-ids", nargs="+", default=["powerpoint_windows_59", "vscode_macos_0"])
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--width-logprob-delta", type=float, default=5.0)
    parser.add_argument("--min-sampling-width", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--image-min-pixels", type=int, default=65_536)
    parser.add_argument("--image-max-pixels", type=int, default=16_777_216)
    args = parser.parse_args()
    if not 0 < args.scale <= 1:
        parser.error("--scale must be in (0, 1]")
    if args.sample_count < 2:
        parser.error("--sample-count must be at least 2")
    if not 1 <= args.batch_size <= args.sample_count:
        parser.error("--batch-size must be between 1 and sample-count")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.width_logprob_delta <= 0:
        parser.error("--width-logprob-delta must be positive")
    if args.min_sampling_width < 2:
        parser.error("--min-sampling-width must be at least 2")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    payload = run(cli_args)
    print(json.dumps({"output": str((cli_args.output / "summary.json").resolve()), "aggregate": payload["aggregate"]}, indent=2))
