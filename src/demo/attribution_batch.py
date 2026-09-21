"""Batch post-processing for traced requests and multi-frame trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from attribution import (
    load_attribution,
    resolve_trace_path,
    write_attribution_viewer,
    write_trajectory_viewer,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def run_attribution_batch(
    manifest_path: Path,
    *,
    trace_root: Path,
    output_root: Path,
    resume: bool = True,
) -> dict[str, Any]:
    """Render every trace or trajectory declared in a portable JSON manifest."""

    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("items"), list):
        raise ValueError("attribution batch manifest must use schema_version 1 and contain items")
    base = manifest_path.parent
    trace_root = trace_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for index, item in enumerate(manifest["items"]):
        item_id = str(item.get("id") or f"item-{index:04d}")
        mode = str(item.get("mode", "trajectory"))
        source = _resolve(base, item["path"])
        output = output_root / item_id
        result_path = output / "batch-result.json"
        if resume and result_path.is_file():
            results.append(json.loads(result_path.read_text()))
            continue
        if mode == "trajectory":
            payload = write_trajectory_viewer(source, trace_root, output)
        elif mode == "trace":
            resolved = resolve_trace_path(
                source,
                trace_root=trace_root,
                trace_index=int(item.get("trace_index", 0)),
            )
            payload = write_attribution_viewer(load_attribution(resolved), output)
        else:
            raise ValueError(f"unknown attribution batch mode {mode!r} for {item_id}")
        result = {"id": item_id, "mode": mode, "source": str(source), **payload}
        _write_json(result_path, result)
        results.append(result)
        print(json.dumps({"id": item_id, "completed": len(results)}, sort_keys=True), flush=True)
    summary = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "trace_root": str(trace_root),
        "output_root": str(output_root),
        "completed_count": len(results),
        "items": results,
    }
    _write_json(output_root / "summary.json", summary)
    return summary
