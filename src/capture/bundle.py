"""Atomic-ish trajectory bundle construction with exact-byte hashing."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Iterable
from copy import deepcopy
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .schema import RESERVED_FIELDS, SCHEMA_VERSION
from .security import scan_payload


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n").encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BundleWriter:
    """Write one bundle, scanning every textual payload before persistence."""

    def __init__(
        self,
        root: Path,
        *,
        backend: str,
        fixture: dict[str, Any],
        task_id: str,
        software: dict[str, Any],
        model: dict[str, Any],
        capture_consent: bool,
        redactions: Iterable[tuple[int, int, int, int]] = (),
        trajectory_id: str | None = None,
    ) -> None:
        if not capture_consent:
            raise ValueError("capture_consent must be explicit")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.trajectory_id = trajectory_id or f"traj-{timestamp}-{uuid.uuid4().hex[:10]}"
        self.path = root / self.trajectory_id
        self.path.mkdir(parents=True, exist_ok=False)
        for directory in ("frames", "requests", "responses", "actions", "raw"):
            (self.path / directory).mkdir()
        self._started_wall = datetime.now(UTC)
        self._started_mono = time.monotonic_ns()
        self._events: list[dict[str, Any]] = []
        self._raw_events: list[dict[str, Any]] = []
        self._redactions = tuple(tuple(int(v) for v in rect) for rect in redactions)
        self.manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "trajectory_id": self.trajectory_id,
            "fixture": fixture,
            "task_id": task_id,
            "timestamps": {"started_at": self._started_wall.isoformat(), "completed_at": None},
            "software": software,
            "model": model,
            "backend": backend,
            "research_split": "local-4b" if backend == "local" else None,
            "research_eligible": backend == "local",
            "viewport": fixture["viewport"],
            "capture_consent": True,
            "redaction": {
                "applied": bool(self._redactions),
                "rectangles": [list(rect) for rect in self._redactions],
                "method": "solid-black-before-persist" if self._redactions else None,
            },
            "files": {},
        }

    def write_json(self, relative: str, payload: Any) -> str:
        scan_payload(payload, source=relative)
        return self.write_bytes(relative, canonical_json(payload))

    def write_bytes(self, relative: str, data: bytes) -> str:
        path = self._safe_path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return relative

    def write_image(self, relative: str, image: Image.Image) -> dict[str, Any]:
        image = image.convert("RGB")
        if self._redactions:
            image = image.copy()
            draw = ImageDraw.Draw(image)
            for rectangle in self._redactions:
                draw.rectangle(rectangle, fill=(0, 0, 0))
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=False, compress_level=9)
        data = buffer.getvalue()
        self.write_bytes(relative, data)
        return {"path": relative, "sha256": sha256_bytes(data), "width": image.width, "height": image.height}

    def add_raw_event(self, event: dict[str, Any]) -> str:
        scan_payload(event, source="raw/events.jsonl")
        raw_id = str(event.get("raw_event_id") or f"raw-{len(self._raw_events):06d}")
        record = {"raw_event_id": raw_id, **event}
        self._raw_events.append(record)
        return raw_id

    def add_event(
        self,
        event_type: str,
        *,
        step: int | None,
        data: dict[str, Any],
        artifacts: list[dict[str, Any]] | None = None,
        raw_event_refs: list[str] | None = None,
        interpretability_replay_ready: bool | None = None,
        interpretability: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sequence = len(self._events)
        now = datetime.now(UTC)
        reserved = deepcopy(RESERVED_FIELDS)
        reserved["interpretability"]["replay_ready"] = interpretability_replay_ready
        if interpretability:
            reserved["interpretability"].update(interpretability)
        record = {
            "sequence": sequence,
            "monotonic_ns": time.monotonic_ns() - self._started_mono,
            "wall_time": now.isoformat(),
            "event_type": event_type,
            "parent_step_ref": step,
            "data": data,
            "artifacts": artifacts or [],
            "raw_event_refs": raw_event_refs or [],
            **reserved,
        }
        scan_payload(record, source="events.jsonl")
        self._events.append(record)
        return record

    def finalize(self, annotations: dict[str, Any], *, extra_files: dict[str, bytes] | None = None) -> Path:
        self.write_bytes("events.jsonl", b"".join(canonical_json_line(event) for event in self._events))
        self.write_bytes("raw/events.jsonl", b"".join(canonical_json_line(event) for event in self._raw_events))
        self.write_json("annotations.json", annotations)
        for relative, data in (extra_files or {}).items():
            self.write_bytes(relative, data)
        self.manifest["timestamps"]["completed_at"] = datetime.now(UTC).isoformat()
        files: dict[str, Any] = {}
        for path in sorted(self.path.rglob("*")):
            if not path.is_file() or path.name == "manifest.json":
                continue
            relative = path.relative_to(self.path).as_posix()
            data = path.read_bytes()
            files[relative] = {"sha256": sha256_bytes(data), "bytes": len(data)}
        self.manifest["files"] = files
        self.write_bytes("manifest.json", canonical_json(self.manifest))
        return self.path

    def _safe_path(self, relative: str) -> Path:
        candidate = (self.path / relative).resolve()
        if self.path.resolve() not in candidate.parents:
            raise ValueError(f"artifact path escapes bundle: {relative}")
        return candidate


def canonical_json_line(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str) + "\n").encode()
