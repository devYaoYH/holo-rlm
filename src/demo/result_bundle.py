"""Create a portable archive containing an experiment run and its referenced traces."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any


def _walk_trace_ids(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"trace_id", "instrumented_trace_id"} and isinstance(item, str) and item.startswith("trace-"):
                result.add(item)
            elif key in {"trace_ids", "instrumented_trace_ids"} and isinstance(item, list):
                result.update(
                    trace_id for trace_id in item if isinstance(trace_id, str) and trace_id.startswith("trace-")
                )
            else:
                result.update(_walk_trace_ids(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_walk_trace_ids(item))
    return result


def referenced_trace_ids(run_path: Path) -> tuple[str, ...]:
    trace_ids: set[str] = set()
    for path in sorted(run_path.rglob("*.json")):
        try:
            trace_ids.update(_walk_trace_ids(json.loads(path.read_text())))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    return tuple(sorted(trace_ids))


def package_results(run_path: Path, trace_root: Path, output: Path) -> dict[str, Any]:
    """Archive one run plus only the activation traces it references."""

    run_path = run_path.expanduser().resolve()
    trace_root = trace_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if not run_path.exists():
        raise FileNotFoundError(run_path)
    trace_ids = referenced_trace_ids(run_path)
    missing = [trace_id for trace_id in trace_ids if not (trace_root / trace_id).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} referenced traces: {', '.join(missing[:5])}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "run_name": run_path.name,
        "trace_count": len(trace_ids),
        "trace_ids": list(trace_ids),
    }
    with tarfile.open(output, "w:gz") as archive:
        archive.add(run_path, arcname=f"run/{run_path.name}", recursive=True)
        for trace_id in trace_ids:
            archive.add(trace_root / trace_id, arcname=f"traces/{trace_id}", recursive=True)
        encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo("bundle-manifest.json")
        info.size = len(encoded)
        archive.addfile(info, io.BytesIO(encoded))
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{sha256}  {output.name}\n")
    return {
        "archive": str(output),
        "sha256": sha256,
        "sha256_file": str(sidecar),
        "bytes": output.stat().st_size,
        **manifest,
    }
