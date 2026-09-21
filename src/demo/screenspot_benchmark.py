"""Resumable and shardable ScreenSpot-Pro benchmark execution."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .screenspot import (
    SCREENSPOT_PROTOCOL,
    ScreenSpotSample,
    list_screenspot_samples,
    run_screenspot_case,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _item_summary(case: dict[str, Any], trace_root: Path) -> dict[str, Any]:
    target = next(run for run in case["runs"] if run["role"] == "target")
    trace_id = target.get("trace_id")
    token_logprobs = None
    if isinstance(trace_id, str):
        logprob_path = trace_root / trace_id / "token_logprobs.json"
        if logprob_path.is_file():
            payload = json.loads(logprob_path.read_text())
            token_logprobs = {
                "sequence_logprob_nats": payload["sequence_logprob_nats"],
                "generated_token_count": payload["generated_token_count"],
                "parameters": payload["parameters"],
            }
    return {
        "sample_id": case["sample"]["id"],
        "application": case["sample"]["application"],
        "platform": case["sample"]["platform"],
        "ui_type": case["sample"]["ui_type"],
        "instruction": case["sample"]["instruction"],
        "format_valid": bool(target.get("format_valid", False)),
        "strict_correct": bool(target.get("correct", False)),
        "grounding_correct_after_repair": bool(target.get("grounding_correct", False)),
        "trace_id": trace_id,
        "token_logprobs": token_logprobs,
        "error": target.get("error"),
        "repair_error": target.get("repair_error"),
    }


def select_samples(
    samples: tuple[ScreenSpotSample, ...],
    *,
    shard_index: int,
    num_shards: int,
    offset: int,
    count: int | None,
) -> tuple[ScreenSpotSample, ...]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard index must be within [0, num_shards)")
    if offset < 0 or (count is not None and count < 1):
        raise ValueError("offset must be non-negative and count must be positive")
    sharded = tuple(sample for index, sample in enumerate(samples) if index % num_shards == shard_index)
    return sharded[offset:] if count is None else sharded[offset : offset + count]


def run_screenspot_benchmark(
    *,
    annotation_root: Path,
    image_root: Path,
    base_url: str,
    model_id: str,
    trace_root: Path,
    output_dir: Path,
    trace_profile: str,
    trace_generation_steps: int,
    shard_index: int = 0,
    num_shards: int = 1,
    offset: int = 0,
    count: int | None = None,
    resume: bool = True,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Run a stable ScreenSpot shard and persist one atomic result per item."""

    output_dir = output_dir.expanduser().resolve()
    trace_root = trace_root.expanduser().resolve()
    items_dir = output_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    selected = select_samples(
        list_screenspot_samples(annotation_root, image_root),
        shard_index=shard_index,
        num_shards=num_shards,
        offset=offset,
        count=count,
    )
    run_config = {
        "schema_version": 1,
        "dataset": "likaixin/ScreenSpot-Pro",
        "inference_protocol": SCREENSPOT_PROTOCOL,
        "base_url": base_url,
        "model_id": model_id,
        "trace_profile": trace_profile,
        "trace_generation_steps": trace_generation_steps,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "offset": offset,
        "count": count,
        "selected_count": len(selected),
        "annotation_root": str(annotation_root.expanduser().resolve()),
        "image_root": str(image_root.expanduser().resolve()),
        "trace_root": str(trace_root),
    }
    _write_json(output_dir / "run.json", run_config)
    started = time.time()
    summaries: list[dict[str, Any]] = []
    for sample in selected:
        item_dir = items_dir / sample.id
        summary_path = item_dir / "result.json"
        if resume and summary_path.is_file():
            summaries.append(json.loads(summary_path.read_text()))
            continue
        try:
            case = run_screenspot_case(
                sample=sample,
                controls=(),
                base_url=base_url,
                model_id=model_id,
                output_dir=item_dir,
                trace_generation_steps=trace_generation_steps,
                trace_profile=trace_profile,
            )
            summary = _item_summary(case, trace_root)
        except Exception as exc:
            summary = {
                "sample_id": sample.id,
                "application": sample.application,
                "platform": sample.platform,
                "ui_type": sample.ui_type,
                "instruction": sample.instruction,
                "format_valid": False,
                "strict_correct": False,
                "grounding_correct_after_repair": False,
                "trace_id": None,
                "token_logprobs": None,
                "error": repr(exc),
            }
            if fail_fast:
                _write_json(summary_path, summary)
                raise
        _write_json(summary_path, summary)
        summaries.append(summary)
        print(
            json.dumps(
                {
                    "sample_id": sample.id,
                    "completed": len(summaries),
                    "selected": len(selected),
                    "strict_correct": summary["strict_correct"],
                    "error": summary.get("error"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    failures = sum(bool(item.get("error")) for item in summaries)
    completed = len(summaries)
    result = {
        **run_config,
        "completed_count": completed,
        "strict_correct": sum(bool(item["strict_correct"]) for item in summaries),
        "format_valid": sum(bool(item["format_valid"]) for item in summaries),
        "grounding_correct_after_repair": sum(
            bool(item["grounding_correct_after_repair"]) for item in summaries
        ),
        "failed_count": failures,
        "strict_accuracy": (
            sum(bool(item["strict_correct"]) for item in summaries) / completed if completed else None
        ),
        "runtime_seconds": time.time() - started,
        "items": summaries,
    }
    _write_json(output_dir / "summary.json", result)
    (output_dir / "items.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in summaries)
    )
    return result
