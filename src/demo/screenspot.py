"""Reproducible ScreenSpot-Pro point-grounding probes for the local Holo endpoint."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from .backends import png_data_url

SCREENSPOT_SYSTEM_PROMPT = """You are evaluating one GUI-grounding instruction on one static screenshot.
Return exactly one desktop_action tool call that clicks the requested visible target. Coordinates are normalized
integers from 0 to 1000, with (0, 0) at the top-left and (1000, 1000) at the bottom-right. Do not scroll, type,
wait, navigate, or explain. Each coordinate value must contain digits only: no quotes, commas, units, or prose.
Base the click only on the screenshot and instruction."""

SCREENSPOT_CLICK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "x", "y"],
    "properties": {
        "action": {"const": "click"},
        "x": {"type": "integer", "minimum": 0, "maximum": 1000},
        "y": {"type": "integer", "minimum": 0, "maximum": 1000},
    },
}
TRACE_PROFILES = ("none", "logprobs", "attribution", "full")


@dataclass(frozen=True)
class ScreenSpotSample:
    """One positive ScreenSpot-Pro instruction and its point-grounding target."""

    id: str
    instruction: str
    image_path: Path
    bbox: tuple[float, float, float, float]
    image_size: tuple[int, int]
    application: str
    platform: str
    ui_type: str
    annotation_path: Path


def load_screenspot_sample(annotation_root: Path, image_root: Path, sample_id: str) -> ScreenSpotSample:
    """Resolve a sample ID from the official per-application annotation files."""

    annotation_root = annotation_root.expanduser().resolve()
    image_root = image_root.expanduser().resolve()
    for annotation_path in sorted(annotation_root.glob("*.json")):
        records = json.loads(annotation_path.read_text())
        for record in records:
            if record.get("id") != sample_id:
                continue
            bbox = tuple(float(value) for value in record["bbox"])
            image_size = tuple(int(value) for value in record["img_size"])
            image_path = image_root / str(record["img_filename"])
            if not image_path.is_file():
                # Candidate subsets may store the image under its stable sample ID.
                image_path = image_root / f"{sample_id}.png"
            if not image_path.is_file():
                raise FileNotFoundError(f"ScreenSpot-Pro image is missing for {sample_id}: {image_path}")
            return ScreenSpotSample(
                id=sample_id,
                instruction=str(record["instruction"]),
                image_path=image_path,
                bbox=bbox,  # type: ignore[arg-type]
                image_size=image_size,  # type: ignore[arg-type]
                application=str(record["application"]),
                platform=str(record["platform"]),
                ui_type=str(record["ui_type"]),
                annotation_path=annotation_path,
            )
    raise KeyError(f"unknown ScreenSpot-Pro sample ID: {sample_id}")


def list_screenspot_samples(annotation_root: Path, image_root: Path) -> tuple[ScreenSpotSample, ...]:
    """Load every official positive sample in stable ID order."""

    annotation_root = annotation_root.expanduser().resolve()
    image_root = image_root.expanduser().resolve()
    samples: list[ScreenSpotSample] = []
    seen: set[str] = set()
    for annotation_path in sorted(annotation_root.glob("*.json")):
        records = json.loads(annotation_path.read_text())
        if not isinstance(records, list):
            raise ValueError(f"ScreenSpot annotation must contain a list: {annotation_path}")
        for record in records:
            sample_id = str(record["id"])
            if sample_id in seen:
                raise ValueError(f"duplicate ScreenSpot sample ID: {sample_id}")
            seen.add(sample_id)
            image_path = image_root / str(record["img_filename"])
            if not image_path.is_file():
                image_path = image_root / f"{sample_id}.png"
            if not image_path.is_file():
                raise FileNotFoundError(f"ScreenSpot-Pro image is missing for {sample_id}: {image_path}")
            samples.append(
                ScreenSpotSample(
                    id=sample_id,
                    instruction=str(record["instruction"]),
                    image_path=image_path,
                    bbox=tuple(float(value) for value in record["bbox"]),  # type: ignore[arg-type]
                    image_size=tuple(int(value) for value in record["img_size"]),  # type: ignore[arg-type]
                    application=str(record["application"]),
                    platform=str(record["platform"]),
                    ui_type=str(record["ui_type"]),
                    annotation_path=annotation_path,
                )
            )
    return tuple(sorted(samples, key=lambda sample: sample.id))


def build_screenspot_request(
    image: Image.Image,
    instruction: str,
    model_id: str,
    *,
    trace_generation_steps: int = 0,
    trace_profile: str | None = None,
) -> dict[str, Any]:
    """Build the deterministic, single-click request used for target and controls."""

    image_url, _ = png_data_url(image)
    request: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SCREENSPOT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Instruction: {instruction}"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "temperature": 0,
        # A complete native Holo click call is about 50 tokens. Keeping the
        # deterministic probe bounded at 64 avoids paying for a long malformed
        # completion while retaining every coordinate token.
        "max_tokens": max(64, trace_generation_steps),
        "chat_template_kwargs": {"enable_thinking": False},
        "tools": [
            {
                "type": "function",
                "function": {"name": "desktop_action", "parameters": SCREENSPOT_CLICK_SCHEMA},
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "desktop_action"}},
    }
    profile = trace_profile or ("attribution" if trace_generation_steps else "none")
    if profile not in TRACE_PROFILES:
        raise ValueError(f"unknown trace profile: {profile}")
    if profile != "none":
        attribution = profile in {"attribution", "full"}
        request["trace"] = {
            "capture_attentions": attribution,
            "capture_hidden_states": profile == "full",
            "capture_kv": False,
            "capture_value_norms": attribution,
            "capture_rollout": attribution,
            "capture_logprobs": True,
            "max_generation_steps": trace_generation_steps or 64,
        }
    return request


def parse_screenspot_click(response: dict[str, Any], image_size: tuple[int, int]) -> dict[str, Any]:
    """Extract the normalized and pixel-space click from one completion."""

    message = response["choices"][0]["message"]
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise ValueError("ScreenSpot completion must contain exactly one tool call")
    call = tool_calls[0]
    if call.get("function", {}).get("name") != "desktop_action":
        raise ValueError("ScreenSpot completion called the wrong tool")
    arguments = call["function"]["arguments"]
    action = json.loads(arguments) if isinstance(arguments, str) else arguments
    if not isinstance(action, dict) or set(action) != {"action", "x", "y"} or action.get("action") != "click":
        raise ValueError("ScreenSpot completion did not return one click")
    if type(action["x"]) is not int or type(action["y"]) is not int:
        raise ValueError("ScreenSpot click coordinates must be integers")
    x = action["x"]
    y = action["y"]
    if not 0 <= x <= 1000 or not 0 <= y <= 1000:
        raise ValueError("ScreenSpot click is outside normalized coordinate bounds")
    width, height = image_size
    return {
        "normalized": {"x": x, "y": y},
        "pixel": {"x": x * width / 1000, "y": y * height / 1000},
    }


def repair_screenspot_click(response: dict[str, Any], image_size: tuple[int, int]) -> dict[str, Any]:
    """Recover the first unsigned integer from malformed coordinate fields.

    This deliberately narrow repair is reported separately from the official
    end-to-end score. It lets the case study distinguish visual grounding from
    Holo's occasional punctuation inside an otherwise unambiguous x/y value.
    """

    message = response["choices"][0]["message"]
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise ValueError("ScreenSpot completion must contain exactly one tool call")
    call = tool_calls[0]
    if call.get("function", {}).get("name") != "desktop_action":
        raise ValueError("ScreenSpot completion called the wrong tool")
    arguments = call["function"]["arguments"]
    action = json.loads(arguments) if isinstance(arguments, str) else arguments
    if not isinstance(action, dict) or action.get("action") != "click":
        raise ValueError("ScreenSpot completion did not return one click")

    repaired: dict[str, int] = {}
    source: dict[str, Any] = {}
    for key in ("x", "y"):
        raw = action.get(key)
        source[key] = raw
        if type(raw) is int:
            value = raw
        elif isinstance(raw, str):
            match = re.search(r"(?<![-\d])\d+", raw)
            if match is None:
                raise ValueError(f"ScreenSpot {key} field has no integer")
            value = int(match.group())
        else:
            raise ValueError(f"ScreenSpot {key} field has no integer")
        if not 0 <= value <= 1000:
            raise ValueError("ScreenSpot repaired click is outside normalized coordinate bounds")
        repaired[key] = value

    width, height = image_size
    return {
        "normalized": repaired,
        "pixel": {"x": repaired["x"] * width / 1000, "y": repaired["y"] * height / 1000},
        "source": source,
        "repair": "first_unsigned_integer_per_coordinate_field",
    }


def point_hits_bbox(point: dict[str, float], bbox: tuple[float, float, float, float]) -> bool:
    """Match the official ScreenSpot positive-sample point-in-box metric."""

    x1, y1, x2, y2 = bbox
    return x1 <= point["x"] <= x2 and y1 <= point["y"] <= y2


def run_screenspot_case(
    *,
    sample: ScreenSpotSample,
    controls: Iterable[str],
    base_url: str,
    model_id: str,
    output_dir: Path,
    trace_generation_steps: int,
    trace_profile: str | None = None,
) -> dict[str, Any]:
    """Run one target instruction and a same-image diverse prompt ensemble."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(sample.image_path) as source:
        image = source.convert("RGB")
    if image.size != sample.image_size:
        raise ValueError(
            f"annotation says {sample.image_size} but {sample.image_path.name} is {image.size}"
        )
    prompts = [("target", sample.instruction), *[(f"control-{index:02d}", value) for index, value in enumerate(controls)]]
    runs: list[dict[str, Any]] = []
    with httpx.Client(timeout=3600, trust_env=False) as client:
        for role, instruction in prompts:
            request = build_screenspot_request(
                image,
                instruction,
                model_id,
                trace_generation_steps=trace_generation_steps,
                trace_profile=trace_profile,
            )
            response = client.post(f"{base_url.rstrip('/')}/chat/completions", json=request)
            response.raise_for_status()
            payload = response.json()
            run = {
                "role": role,
                "instruction": instruction,
                "trace_id": payload.get("instrumented_trace_id"),
                "response": payload,
            }
            try:
                click = parse_screenspot_click(payload, image.size)
                run["click"] = click
                run["format_valid"] = True
                if role == "target":
                    run["grounding_correct"] = point_hits_bbox(click["pixel"], sample.bbox)
                    run["correct"] = point_hits_bbox(click["pixel"], sample.bbox)
            except (KeyError, TypeError, ValueError) as exc:
                run["error"] = str(exc)
                run["format_valid"] = False
                try:
                    repaired_click = repair_screenspot_click(payload, image.size)
                    run["repaired_click"] = repaired_click
                    run["grounding_correct"] = point_hits_bbox(repaired_click["pixel"], sample.bbox)
                except (KeyError, TypeError, ValueError) as repair_exc:
                    run["repair_error"] = str(repair_exc)
                if role == "target":
                    run["correct"] = False
            runs.append(run)
    image_bytes = sample.image_path.read_bytes()
    result = {
        "schema_version": 2,
        "dataset": "likaixin/ScreenSpot-Pro",
        "dataset_url": "https://huggingface.co/datasets/likaixin/ScreenSpot-Pro",
        "sample": {
            "id": sample.id,
            "instruction": sample.instruction,
            "bbox": list(sample.bbox),
            "image_size": list(sample.image_size),
            "image_path": str(sample.image_path),
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "application": sample.application,
            "platform": sample.platform,
            "ui_type": sample.ui_type,
            "annotation_path": str(sample.annotation_path),
        },
        "model": {"id": model_id, "base_url": base_url},
        "trace_generation_steps": trace_generation_steps,
        "trace_profile": trace_profile or ("attribution" if trace_generation_steps else "none"),
        "runs": runs,
    }
    manifest = output_dir / "case.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {"manifest": str(manifest), **result}
