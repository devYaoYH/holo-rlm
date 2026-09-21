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

SCREENSPOT_LOCALIZATION_SCHEMA: dict[str, Any] = {
    "properties": {
        "x": {
            "description": "X coordinate as integer in [0, 1000]",
            "maximum": 1000,
            "minimum": 0,
            "title": "X",
            "type": "integer",
        },
        "y": {
            "description": "Y coordinate as integer in [0, 1000]",
            "maximum": 1000,
            "minimum": 0,
            "title": "Y",
            "type": "integer",
        },
    },
    "required": ["x", "y"],
    "title": "VisualLocalizerOutput",
    "type": "object",
}
# Kept only for replaying the older multi-frame diagnostic protocol. New
# ScreenSpot benchmark requests use SCREENSPOT_LOCALIZATION_SCHEMA above.
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
SCREENSPOT_PROTOCOL = "hcompany_element_localization_v1"


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
    """Build H Company's official deterministic element-localization request."""

    image_url, _ = png_data_url(image)
    prompt = (
        "Localize an element on the GUI image according to the provided target "
        "and output a click position.\n"
        f" * You must output a valid JSON following the format: {SCREENSPOT_LOCALIZATION_SCHEMA}\n"
        f" Your target is:\n{instruction}"
    )
    request: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": 0,
        # The official output is a two-field JSON object. Keep enough headroom
        # to retain a malformed completion for diagnosis without an unbounded run.
        "max_tokens": max(64, trace_generation_steps),
        "chat_template_kwargs": {"enable_thinking": False},
        "structured_outputs": {"json": SCREENSPOT_LOCALIZATION_SCHEMA},
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


def _screenspot_coordinates(response: dict[str, Any]) -> dict[str, Any]:
    """Read the official JSON localization response, with legacy tool-call compatibility."""

    message = response["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        coordinates = json.loads(content)
        if not isinstance(coordinates, dict):
            raise ValueError("ScreenSpot completion content must be a JSON object")
        return coordinates

    # Older captured runs used a custom desktop_action tool. Retaining this
    # reader keeps their artifacts inspectable without using that protocol for
    # new benchmark requests.
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or len(tool_calls) != 1:
        raise ValueError("ScreenSpot completion must contain one localization result")
    call = tool_calls[0]
    if call.get("function", {}).get("name") != "desktop_action":
        raise ValueError("ScreenSpot completion called the wrong tool")
    arguments = call["function"]["arguments"]
    action = json.loads(arguments) if isinstance(arguments, str) else arguments
    if not isinstance(action, dict) or action.get("action") != "click":
        raise ValueError("ScreenSpot completion did not return one click")
    return {"x": action.get("x"), "y": action.get("y")}


def parse_screenspot_click(response: dict[str, Any], image_size: tuple[int, int]) -> dict[str, Any]:
    """Extract the normalized and pixel-space click from one completion."""

    coordinates = _screenspot_coordinates(response)
    if set(coordinates) != {"x", "y"}:
        raise ValueError("ScreenSpot completion must contain exactly x and y")
    if type(coordinates["x"]) is not int or type(coordinates["y"]) is not int:
        raise ValueError("ScreenSpot click coordinates must be integers")
    x = coordinates["x"]
    y = coordinates["y"]
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

    coordinates = _screenspot_coordinates(response)

    repaired: dict[str, int] = {}
    source: dict[str, Any] = {}
    for key in ("x", "y"):
        raw = coordinates.get(key)
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
        "inference_protocol": SCREENSPOT_PROTOCOL,
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
