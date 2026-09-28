"""Inventory trace bundles and write a value-free release-readiness report.

Usage: python scripts/audit_trace_release.py [--verify-hashes] [--output PATH]
The report contains filenames, counts, hashes and warnings, never request text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
KEY = re.compile(r"step_(\d+)_layer_(\d+)$")
PRIVATE = re.compile(r"(?i)(?:-----BEGIN .*PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{12,}|\bbearer\s+[A-Za-z0-9._~+/=-]{8,}|\b(?:authorization|cookie|api[_-]?key|access[_-]?token)\s*[:=])")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def npz_layout(path: Path) -> tuple[list[int], list[int], int, list[str]]:
    """Read member names only; no giant tensor decompression."""
    steps, layers, bad = set(), set(), []
    try:
        with ZipFile(path) as archive:
            for name in archive.namelist():
                match = KEY.fullmatch(name.removesuffix(".npy"))
                if match:
                    steps.add(int(match[1]))
                    layers.add(int(match[2]))
                else:
                    bad.append(name)
            count = len(archive.namelist())
    except (OSError, ValueError):
        return [], [], 0, ["unreadable_npz"]
    return sorted(steps), sorted(layers), count, bad


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_trace(path: Path, verify_hashes: bool) -> dict:
    files = {item.name: item for item in path.iterdir() if item.is_file()}
    model = read_json(path / "model.json")
    positions = read_json(path / "positions.json")
    manifest = read_json(path / "manifest.json")
    tokens = read_json(path / "generated_tokens.json")
    logprobs = read_json(path / "token_logprobs.json")
    warnings = []
    records = manifest.get("files", [])
    if not isinstance(records, list):
        records = []
    listed = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            warnings.append("invalid_manifest_record")
            continue
        name = record["path"]
        listed.add(name)
        item = files.get(name)
        if item is None or name != Path(name).name:
            warnings.append("missing_or_unsafe_manifest_file")
            continue
        if item.stat().st_size != record.get("bytes"):
            warnings.append("size_mismatch")
        if verify_hashes and sha256(item) != record.get("sha256"):
            warnings.append("sha256_mismatch")
    if set(files) - {"manifest.json"} != listed:
        warnings.append("manifest_file_set_mismatch")
    for name in ("input_ids.npy", "generated_ids.npy", "model.json", "positions.json", "completion.json", "attention_last_query_rows.npz"):
        if name not in files:
            warnings.append(f"missing_{name}")
    prompt = positions.get("prompt_token_count")
    generated = positions.get("generated_token_count")
    try:
        if int(np.load(path / "input_ids.npy", mmap_mode="r", allow_pickle=False).shape[-1]) != prompt:
            warnings.append("prompt_count_mismatch")
        if int(np.load(path / "generated_ids.npy", mmap_mode="r", allow_pickle=False).shape[-1]) != generated:
            warnings.append("generated_count_mismatch")
    except (OSError, ValueError, KeyError):
        warnings.append("unreadable_token_ids")
    if tokens and len(tokens.get("token_ids", [])) != generated:
        warnings.append("generated_tokens_count_mismatch")
    if logprobs and (logprobs.get("generated_token_count") != generated or len(logprobs.get("tokens", [])) != generated):
        warnings.append("logprobs_count_mismatch")
    attention_steps, attention_layers, attention_count, malformed = npz_layout(path / "attention_last_query_rows.npz") if "attention_last_query_rows.npz" in files else ([], [], 0, [])
    if malformed:
        warnings.append("malformed_attention_members")
    expected_layers = model.get("attention_layer_indices", [])
    if attention_layers and attention_layers not in (list(range(len(expected_layers))), expected_layers):
        warnings.append("attention_layer_mapping_unknown")
    if attention_steps and attention_steps != list(range(len(attention_steps))):
        warnings.append("attention_steps_noncontiguous")
    hidden_steps, hidden_layers, hidden_count, malformed = npz_layout(path / "hidden_state_last_query_rows.npz") if "hidden_state_last_query_rows.npz" in files else ([], [], 0, [])
    if malformed:
        warnings.append("malformed_hidden_members")
    if hidden_steps and hidden_steps != list(range(len(hidden_steps))):
        warnings.append("hidden_steps_noncontiguous")
    if hidden_steps and hidden_count != len(hidden_steps) * len(hidden_layers):
        warnings.append("hidden_grid_incomplete")
    for name in ("request.json", "response.json", "completion.json"):
        if name in files:
            content = files[name].read_text(errors="replace")
            if PRIVATE.search(content):
                warnings.append(f"possible_secret_{name}")
            if EMAIL.search(content):
                warnings.append(f"email_{name}")
    return {
        "id": path.name,
        "source": str(path.parent.relative_to(DATA)),
        "bytes": sum(item.stat().st_size for item in files.values()),
        "model_revision": model.get("model_revision"),
        "model_type": model.get("model_type"),
        "device": model.get("device"),
        "prompt_tokens": prompt,
        "generated_tokens": generated,
        "captured_attention_steps": len(attention_steps),
        "captured_attention_layers": len(attention_layers),
        "attention_members": attention_count,
        "captured_hidden_steps": len(hidden_steps),
        "captured_hidden_layers": len(hidden_layers),
        "hidden_members": hidden_count,
        "has_logprobs": "token_logprobs.json" in files,
        "has_prompt_matrices": "attention_prompt_mean.npz" in files,
        "has_input_image": any(name.startswith("model-input-") for name in files),
        "has_response": "response.json" in files,
        "warnings": sorted(set(warnings)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--output", type=Path, default=DATA / "release-audit" / "audit.json")
    args = parser.parse_args()
    trace_paths = sorted(
        p for p in DATA.rglob("manifest.json")
        if p.parent.name.startswith("trace-") and "hf-stage" not in p.relative_to(DATA).parts
    )
    rows = [audit_trace(path.parent, args.verify_hashes) for path in trace_paths]
    sources = defaultdict(list)
    for row in rows:
        sources[row["source"]].append(row)
    groups = {
        source: {
            "traces": len(items),
            "bytes": sum(item["bytes"] for item in items),
            "with_hidden_states": sum(item["captured_hidden_steps"] > 0 for item in items),
            "with_logprobs": sum(item["has_logprobs"] for item in items),
            "with_prompt_matrices": sum(item["has_prompt_matrices"] for item in items),
            "warnings": dict(Counter(warning for item in items for warning in item["warnings"])),
        }
        for source, items in sorted(sources.items())
    }
    incomplete_dirs = sorted(
        str(p.relative_to(DATA)) for p in DATA.rglob("trace-*")
        if p.is_dir() and "hf-stage" not in p.relative_to(DATA).parts
        and (p / "request.json").is_file() and not (p / "manifest.json").is_file()
    )
    report = {
        "schema_version": 1,
        "hashes_verified": args.verify_hashes,
        "trace_count": len(rows),
        "trace_bytes": sum(row["bytes"] for row in rows),
        "incomplete_trace_directories": incomplete_dirs,
        "groups": groups,
        "model_revisions": dict(Counter(row["model_revision"] or "unknown" for row in rows)),
        "warning_counts": dict(Counter(warning for row in rows for warning in row["warnings"])),
        "traces": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("trace_count", "trace_bytes", "groups", "model_revisions", "warning_counts", "incomplete_trace_directories")}, indent=2))


if __name__ == "__main__":
    main()
