"""Strict validation for research-quality v0 trajectory bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from .normalizer import read_jsonl
from .schema import EVENT_TYPES, SCHEMA_VERSION, validate_action
from .security import UnsafeCapture, scan_payload, scan_text


class ValidationError(ValueError):
    pass


def _fail(message: str) -> None:
    raise ValidationError(message)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read {path}: {exc}") from exc


def validate_bundle(path: Path, *, require_replay_ready: bool = True) -> dict[str, Any]:
    path = path.resolve()
    manifest_path = path / "manifest.json"
    manifest = _load_json(manifest_path)
    required_manifest = {
        "schema_version",
        "trajectory_id",
        "fixture",
        "task_id",
        "timestamps",
        "software",
        "model",
        "backend",
        "viewport",
        "capture_consent",
        "redaction",
        "files",
    }
    missing = required_manifest - set(manifest)
    if missing:
        _fail(f"manifest missing fields: {sorted(missing)}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        _fail(f"unsupported schema_version: {manifest['schema_version']!r}")
    if manifest["capture_consent"] is not True:
        _fail("capture consent is not true")
    viewport = manifest["viewport"]
    expected_dimensions = (int(viewport["width"]), int(viewport["height"]))

    files = manifest["files"]
    for relative, metadata in files.items():
        artifact = (path / relative).resolve()
        if path not in artifact.parents or not artifact.is_file():
            _fail(f"missing or unsafe artifact: {relative}")
        data = artifact.read_bytes()
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != metadata.get("sha256"):
            _fail(f"hash mismatch: {relative}")
        if len(data) != metadata.get("bytes"):
            _fail(f"size mismatch: {relative}")
        try:
            if artifact.suffix.lower() in {".json", ".jsonl", ".html", ".txt"}:
                scan_text(data.decode(), source=relative)
        except (UnicodeDecodeError, UnsafeCapture) as exc:
            raise ValidationError(str(exc)) from exc

    required_files = {"events.jsonl", "raw/events.jsonl", "annotations.json"}
    if not required_files <= set(files):
        _fail(f"bundle missing required hashed files: {sorted(required_files - set(files))}")
    actual_files = {
        file.relative_to(path).as_posix()
        for file in path.rglob("*")
        if file.is_file() and file.name != "manifest.json"
    }
    unhashed = actual_files - set(files)
    if unhashed:
        _fail(f"unhashed files present: {sorted(unhashed)}")

    raw_events = list(read_jsonl(path / "raw/events.jsonl"))
    raw_ids = [str(event.get("raw_event_id")) for event in raw_events]
    if len(raw_ids) != len(set(raw_ids)):
        _fail("raw event IDs are not unique")
    events = list(read_jsonl(path / "events.jsonl"))
    if not events:
        _fail("event stream is empty")
    last_mono = -1
    seen_types: set[str] = set()
    steps: dict[int, list[dict[str, Any]]] = {}
    model_inputs: set[str] = set()
    for sequence, event in enumerate(events):
        if event.get("sequence") != sequence:
            _fail(f"non-contiguous sequence at event {sequence}")
        monotonic_ns = event.get("monotonic_ns")
        if not isinstance(monotonic_ns, int) or monotonic_ns < last_mono:
            _fail(f"non-monotonic timestamp at event {sequence}")
        last_mono = monotonic_ns
        event_type = event.get("event_type")
        if event_type not in EVENT_TYPES:
            _fail(f"unknown event type at {sequence}: {event_type!r}")
        seen_types.add(event_type)
        if not isinstance(event.get("wall_time"), str):
            _fail(f"event {sequence} lacks wall timestamp")
        for reserved in ("safety", "interpretability", "rlm"):
            if reserved not in event:
                _fail(f"event {sequence} lacks reserved field {reserved}")
        refs = event.get("raw_event_refs")
        if not isinstance(refs, list) or not refs:
            _fail(f"event {sequence} is not traceable to a raw event")
        unknown_refs = set(map(str, refs)) - set(raw_ids)
        if unknown_refs:
            _fail(f"event {sequence} has unknown raw refs: {sorted(unknown_refs)}")
        for artifact in event.get("artifacts", []):
            relative = artifact.get("path")
            if relative not in files:
                _fail(f"event {sequence} references unhashed artifact: {relative}")
            if artifact.get("sha256") != files[relative]["sha256"]:
                _fail(f"event {sequence} artifact hash disagrees with manifest: {relative}")
            if artifact.get("role") == "model_input":
                model_inputs.add(relative)
                with Image.open(path / relative) as image:
                    if image.size != expected_dimensions:
                        _fail(f"model input has dimensions {image.size}, expected {expected_dimensions}: {relative}")
        step = event.get("parent_step_ref")
        if isinstance(step, int):
            steps.setdefault(step, []).append(event)

    if require_replay_ready:
        missing_types = EVENT_TYPES - seen_types
        if missing_types:
            _fail(f"replay-ready capture lacks event types: {sorted(missing_types)}")
        if not model_inputs:
            _fail("replay-ready capture has no exact model-input image")
        _validate_step_linkage(steps)
        if not any(event["interpretability"].get("replay_ready") is True for event in events if event["event_type"] == "model_request"):
            _fail("no replay-ready model request")
    try:
        scan_payload(manifest, source="manifest.json")
    except UnsafeCapture as exc:
        raise ValidationError(str(exc)) from exc
    return {
        "trajectory_id": manifest["trajectory_id"],
        "events": len(events),
        "raw_events": len(raw_events),
        "files": len(files),
        "replay_ready": require_replay_ready,
    }


def _validate_step_linkage(steps: dict[int, list[dict[str, Any]]]) -> None:
    if not steps or sorted(steps) != list(range(min(steps), max(steps) + 1)):
        _fail("step IDs are not contiguous")
    required = ["observation", "model_request", "model_response", "proposed_action", "action_result"]
    for step, events in steps.items():
        event_types = [event["event_type"] for event in events]
        if event_types != required:
            _fail(f"step {step} linkage is {event_types}, expected {required}")
        observation, request, response, action, result = events
        request_paths = {item["path"] for item in request["artifacts"]}
        observation_inputs = {item["path"] for item in observation["artifacts"] if item.get("role") == "model_input"}
        if not observation_inputs or not observation_inputs <= request_paths:
            _fail(f"step {step} request does not link its exact pre-action model input")
        response_paths = {item["path"] for item in response["artifacts"]}
        action_paths = {item["path"] for item in action["artifacts"]}
        if not response_paths or not action_paths:
            _fail(f"step {step} response/action artifact missing")
        normalized = action.get("data", {}).get("normalized")
        try:
            validate_action(normalized)
        except ValueError as exc:
            _fail(f"step {step} action invalid: {exc}")
        if result.get("data", {}).get("before_state") is None or result.get("data", {}).get("after_state") is None:
            _fail(f"step {step} result lacks before/after state")
