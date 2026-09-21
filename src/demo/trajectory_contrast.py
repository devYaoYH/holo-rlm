"""Matched instruction probes over an immutable multi-frame trajectory context."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from .screenspot import SCREENSPOT_CLICK_SCHEMA, parse_screenspot_click, point_hits_bbox

MULTIFRAME_SYSTEM_PROMPT = """You are evaluating one GUI-grounding instruction on a deterministic booking fixture.
The screenshots are exact observations in chronological order; the final screenshot is the current frame. Use all
screenshots as visual memory, then return exactly one desktop_action tool call that clicks the requested visible
target in the current frame. Coordinates are normalized integers from 0 to 1000. Do not scroll, type, wait, navigate,
or explain. Each coordinate value must contain digits only."""


def build_multiframe_prompt_request(
    source_request: dict[str, Any],
    instruction: str,
    *,
    trace_generation_steps: int,
    max_frame_width: int | None = None,
) -> dict[str, Any]:
    """Clone one captured trajectory decision while changing only its instruction.

    The screenshots and assistant action history remain byte-for-byte identical
    across the target and control requests. Task-specific continuation text is
    replaced by a shared neutral continuation so the requested target is the
    sole varying semantic input.
    """

    request = copy.deepcopy(source_request)
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("source trajectory request has no messages")
    if messages[0].get("role") != "system":
        raise ValueError("source trajectory request must begin with a system message")
    messages[0]["content"] = MULTIFRAME_SYSTEM_PROMPT

    if max_frame_width is not None:
        if max_frame_width < 64:
            raise ValueError("max frame width must be at least 64 pixels")
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image_record = part.get("image_url")
                if isinstance(image_record, dict) and isinstance(image_record.get("url"), str):
                    image_record["url"] = _resample_data_url(image_record["url"], max_frame_width)

    task_replaced = False
    string_user_indices = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    ]
    if not string_user_indices:
        raise ValueError("source trajectory request has no textual user instruction")
    for ordinal, index in enumerate(string_user_indices):
        if ordinal == 0:
            messages[index]["content"] = f"Instruction: {instruction}"
            task_replaced = True
        elif ordinal == len(string_user_indices) - 1:
            messages[index]["content"] = (
                "The previous action was applied. All results have appeared. "
                "Use every retained screenshot and click the requested visible target in the current frame."
            )
        else:
            messages[index]["content"] = (
                "The previous action was applied. Continue the instruction using every screenshot seen so far."
            )
    if not task_replaced:
        raise ValueError("source trajectory instruction could not be replaced")

    request["temperature"] = 0
    request["max_tokens"] = max(64, trace_generation_steps)
    request["chat_template_kwargs"] = {"enable_thinking": False}
    request["tools"] = [
        {
            "type": "function",
            "function": {"name": "desktop_action", "parameters": SCREENSPOT_CLICK_SCHEMA},
        }
    ]
    request["tool_choice"] = {"type": "function", "function": {"name": "desktop_action"}}
    request["trace"] = {
        "capture_attentions": True,
        "capture_hidden_states": False,
        "capture_kv": False,
        "capture_value_norms": True,
        "capture_rollout": True,
        "capture_logprobs": True,
        "max_generation_steps": trace_generation_steps,
    }
    return request


def run_multiframe_prompt_case(
    *,
    source_trace: Path,
    target_instruction: str,
    controls: Iterable[str],
    target_bbox: tuple[float, float, float, float],
    frame_bboxes: tuple[tuple[float, float, float, float] | None, ...],
    base_url: str,
    output_dir: Path,
    trace_generation_steps: int,
    max_frame_width: int | None = None,
) -> dict[str, Any]:
    """Capture a target and diverse instructions over the same frame history."""

    source_trace = source_trace.expanduser().resolve()
    source_request_path = source_trace / "request.json"
    if not source_request_path.is_file():
        raise FileNotFoundError(f"source trace request is missing: {source_request_path}")
    source_request = json.loads(source_request_path.read_text())
    source_images = sorted(source_trace.glob("model-input-*.*"))
    if not source_images:
        raise ValueError("source trajectory trace contains no captured frames")
    if len(frame_bboxes) != len(source_images):
        raise ValueError("frame bbox count must match the source trajectory frame count")
    with Image.open(source_images[-1]) as image:
        source_image_size = image.size
    if max_frame_width is not None and source_image_size[0] > max_frame_width:
        scale = max_frame_width / source_image_size[0]
    else:
        scale = 1.0
    image_size = (
        round(source_image_size[0] * scale),
        round(source_image_size[1] * scale),
    )
    scaled_target_bbox = _scale_bbox(target_bbox, scale)
    scaled_frame_bboxes = tuple(
        _scale_bbox(value, scale) if value is not None else None for value in frame_bboxes
    )

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = [
        ("target", target_instruction),
        *[(f"control-{index:02d}", value) for index, value in enumerate(controls)],
    ]
    runs: list[dict[str, Any]] = []
    with httpx.Client(timeout=3600, trust_env=False) as client:
        for role, instruction in prompts:
            request = build_multiframe_prompt_request(
                source_request,
                instruction,
                trace_generation_steps=trace_generation_steps,
                max_frame_width=max_frame_width,
            )
            response = client.post(f"{base_url.rstrip('/')}/chat/completions", json=request)
            response.raise_for_status()
            payload = response.json()
            run: dict[str, Any] = {
                "role": role,
                "instruction": instruction,
                "trace_id": payload.get("instrumented_trace_id"),
                "response": payload,
            }
            try:
                click = parse_screenspot_click(payload, image_size)
                run["click"] = click
                run["format_valid"] = True
                if role == "target":
                    run["grounding_correct"] = point_hits_bbox(click["pixel"], scaled_target_bbox)
                    run["correct"] = run["grounding_correct"]
            except (KeyError, TypeError, ValueError) as exc:
                run["error"] = str(exc)
                run["format_valid"] = False
                if role == "target":
                    run["correct"] = False
            runs.append(run)

    result = {
        "schema_version": 1,
        "dataset": "booking-fixture-v1",
        "source_trace": str(source_trace),
        "source_frames": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in source_images
        ],
        "model": {"id": source_request.get("model"), "base_url": base_url},
        "target_instruction": target_instruction,
        "target_bbox": list(scaled_target_bbox),
        "frame_bboxes": [list(value) if value is not None else None for value in scaled_frame_bboxes],
        "source_image_size": list(source_image_size),
        "image_size": list(image_size),
        "frame_scale": scale,
        "max_frame_width": max_frame_width,
        "trace_generation_steps": trace_generation_steps,
        "runs": runs,
    }
    manifest = output_dir / "case.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {"manifest": str(manifest), **result}


def _scale_bbox(
    bbox: tuple[float, float, float, float],
    scale: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return x1 * scale, y1 * scale, x2 * scale, y2 * scale


def _resample_data_url(url: str, max_width: int) -> str:
    if not url.startswith("data:image/") or "," not in url:
        raise ValueError("multi-frame probes require embedded data-image URLs")
    _header, encoded = url.split(",", 1)
    with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as source:
        if source.width <= max_width:
            return url
        scale = max_width / source.width
        resized = source.convert("RGB").resize(
            (max_width, round(source.height * scale)),
            resample=Image.Resampling.LANCZOS,
        )
    buffer = BytesIO()
    resized.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
