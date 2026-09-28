"""Verify every staged trace against its manifest and index.

Usage: python3 scripts/verify_hf_stage.py data/hf-stage/screenspot
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    root = args.stage.resolve()
    index = [json.loads(line) for line in (root / "index.jsonl").read_text().splitlines()]
    trace_ids = [record["trace_id"] for record in index]
    if len(set(trace_ids)) != len(trace_ids):
        raise ValueError("duplicate trace IDs in index")
    staged = {path.name for path in (root / "traces").iterdir() if path.is_dir()}
    if set(trace_ids) != staged:
        raise ValueError("index and trace directories differ")
    file_count = 0
    for trace_id in trace_ids:
        path = root / "traces" / trace_id
        manifest = json.loads((path / "manifest.json").read_text())
        records = manifest["files"]
        actual = {item.name for item in path.iterdir() if item.is_file()} - {"manifest.json"}
        if {record["path"] for record in records} != actual:
            raise ValueError(f"manifest file set differs: {trace_id}")
        for record in records:
            name = record["path"]
            if name != Path(name).name:
                raise ValueError(f"unsafe manifest path: {name}")
            item = path / name
            if item.stat().st_size != record["bytes"] or digest(item) != record["sha256"]:
                raise ValueError(f"checksum mismatch: {item}")
            file_count += 1
    print(f"Verified {len(index)} traces and {file_count} files in {root}")


if __name__ == "__main__":
    main()
