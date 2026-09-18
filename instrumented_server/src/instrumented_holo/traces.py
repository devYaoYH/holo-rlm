"""Trace persistence with explicit opt-in for large attention/KV artifacts."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any) -> str:
    return str(value)


@dataclass(frozen=True)
class TraceOptions:
    enabled: bool = False
    capture_attentions: bool = True
    capture_hidden_states: bool = True
    capture_kv: bool = False
    max_traced_generation_steps: int = 4

    @classmethod
    def from_request(cls, raw: dict[str, Any]) -> TraceOptions:
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        trace = raw.get("trace", metadata.get("trace", False))
        if isinstance(trace, dict):
            return cls(
                enabled=True,
                capture_attentions=bool(trace.get("capture_attentions", True)),
                capture_hidden_states=bool(trace.get("capture_hidden_states", True)),
                capture_kv=bool(trace.get("capture_kv", False)),
                max_traced_generation_steps=max(0, int(trace.get("max_generation_steps", 4))),
            )
        return cls(enabled=bool(trace))


class TraceWriter:
    """Writes one self-contained trace bundle per traced completion."""

    def __init__(self, trace_root: Path, *, request_id: str) -> None:
        self.path = trace_root / request_id
        self.path.mkdir(parents=True, exist_ok=False)

    @classmethod
    def create(cls, trace_root: Path) -> tuple[str, TraceWriter]:
        request_id = f"trace-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
        return request_id, cls(trace_root, request_id=request_id)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.path / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
        return path

    def write_bytes(self, name: str, payload: bytes) -> Path:
        path = self.path / name
        path.write_bytes(payload)
        return path

    def write_array(self, name: str, array: Any) -> Path:
        path = self.path / name
        np.save(path, array)
        return path.with_suffix(path.suffix + ".npy") if not path.suffix else path

    def write_npz(self, name: str, arrays: dict[str, Any]) -> Path:
        path = self.path / name
        np.savez_compressed(path, **arrays)
        return path

    def manifest(self) -> dict[str, Any]:
        records = []
        for path in sorted(self.path.iterdir()):
            if path.is_file():
                records.append(
                    {
                        "path": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size,
                    }
                )
        return {"schema_version": 1, "files": records}


def append_trace_response(trace_root: Path, trace_id: str, response: dict[str, Any]) -> None:
    """Add the final OpenAI response and refresh hashes after response shaping."""

    path = (trace_root / trace_id).resolve()
    if trace_root.resolve() not in path.parents or not path.is_dir():
        raise ValueError(f"unknown trace id: {trace_id}")
    response_path = path / "response.json"
    response_path.write_text(json.dumps(response, indent=2, sort_keys=True, default=_json_default) + "\n")
    records = []
    for artifact in sorted(path.iterdir()):
        if artifact.is_file() and artifact.name != "manifest.json":
            records.append(
                {
                    "path": artifact.name,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "bytes": artifact.stat().st_size,
                }
            )
    (path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": records}, indent=2, sort_keys=True) + "\n"
    )
