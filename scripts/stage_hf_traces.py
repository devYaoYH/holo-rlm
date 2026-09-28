"""Stage a compact Hugging Face dataset from audited trace bundles.

No network calls. Raw requests, machine paths, and giant prompt matrices are
excluded. Staged binary files are hard-linked when possible; never edit them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
AUDIT = DATA / "release-audit" / "audit.json"
COMMON = {
    "input_ids.npy", "generated_ids.npy", "generated_tokens.json", "positions.json",
    "vision_inputs.json", "completion.json", "attention_last_query_rows.npz",
    "hidden_state_last_query_rows.npz", "value_norms.npz", "value_norms.json",
    "token_logprobs.json",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def stage_file(source: Path, target: Path, *, copy: bool) -> None:
    if target.exists():
        if source.samefile(target):
            return
        if target.stat().st_size == source.stat().st_size and digest(target) == digest(source):
            return
        raise ValueError(f"existing staged file differs: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copyfile(source, target)
    else:
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)


def compact_model(raw: dict) -> dict:
    return {key: raw[key] for key in (
        "architectures", "attention_layer_indices", "dtype", "model_revision", "model_type",
        "torch_version", "transformers_version", "image_token_id",
    ) if key in raw} | {"vision_config": {key: raw.get("vision_config", {}).get(key)
                                      for key in ("spatial_merge_size", "patch_size", "hidden_size")}}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("screenspot", "residual-review"), default="screenspot")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--copy", action="store_true", help="Copy instead of local hard links")
    args = parser.parse_args()
    audit = read(AUDIT)
    if not audit.get("hashes_verified"):
        parser.error("run audit_trace_release.py --verify-hashes first")
    benchmark_provenance = {}
    for cohort in ("screenspot-resolution-saliency-pilot-v1", "screenspot-resolution-saliency-expansion12-v1"):
        summary_path = DATA / "remote-results" / cohort / "run" / "summary.json"
        summary = read(summary_path)
        for result in summary.get("results", []):
            base = {"cohort": cohort, "sample_id": result.get("sample_id"),
                    "application": result.get("application"), "linear_scale": result.get("linear_scale"),
                    "ui_type": result.get("ui_type"), "run_manifest_sha256": summary.get("manifest_sha256"),
                    "target_bbox": result.get("bbox"), "input_image_size": result.get("input_image_size"),
                    "strict_correct": result.get("strict_correct"), "format_valid": result.get("format_valid"),
                    "predicted_click": result.get("click")}
            trace_ids = result.get("trace_ids", {})
            if trace_ids.get("target"):
                benchmark_provenance[trace_ids["target"]] = {**base, "role": "target", "control_index": None}
            for control_index, trace_id in enumerate(trace_ids.get("controls", [])):
                benchmark_provenance[trace_id] = {**base, "role": "control", "control_index": control_index}
    target = (args.output or (DATA / "hf-stage" / args.profile)).resolve()
    target.mkdir(parents=True, exist_ok=True)
    candidates = []
    for row in audit["traces"]:
        if row["warnings"]:
            continue
        if args.profile == "screenspot":
            if row["source"] != "remote-results/screenspot-resolution-saliency-pilot-v1/traces" or not row["has_logprobs"] or row["id"] not in benchmark_provenance:
                continue
        elif row["source"] != "traces" or row["captured_hidden_steps"] == 0 or not row["model_revision"]:
            continue
        candidates.append(row)
    index = []
    for row in candidates:
        source = DATA / row["source"] / row["id"]
        out = target / "traces" / row["id"]
        out.mkdir(parents=True, exist_ok=True)
        names = set(COMMON)
        if args.profile == "screenspot":
            names.update(item.name for item in source.iterdir() if item.name.startswith("model-input-") and item.suffix.lower() in (".png", ".jpg", ".jpeg"))
        for name in sorted(names):
            if (source / name).is_file():
                stage_file(source / name, out / name, copy=args.copy)
        model_record = compact_model(read(source / "model.json"))
        model_record["model_id"] = read(source / "request.json").get("model")
        write_json(out / "model.json", model_record)
        processor = read(source / "processor.json") if (source / "processor.json").exists() else {}
        write_json(out / "processor.json", {key: processor[key] for key in ("revision", "runtime_image_processor_size") if key in processor})
        files = [{"path": item.name, "bytes": item.stat().st_size, "sha256": digest(item)}
                 for item in sorted(out.iterdir()) if item.is_file() and item.name != "manifest.json"]
        write_json(out / "manifest.json", {"schema_version": 1, "files": files})
        index.append({"trace_id": row["id"], "source": row["source"], "model_revision": row["model_revision"],
                      "prompt_tokens": row["prompt_tokens"], "generated_tokens": row["generated_tokens"],
                      "captured_attention_steps": row["captured_attention_steps"],
                      "captured_hidden_steps": row["captured_hidden_steps"],
                      "has_logprobs": row["has_logprobs"], "has_input_image": args.profile == "screenspot" and row["has_input_image"],
                      **benchmark_provenance.get(row["id"], {})})
    (target / "index.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in index))
    card = f"""---
pretty_name: Holo 3.1 4B generation traces ({args.profile})
language:
- en
tags:
- interpretability
- multimodal
- gui-agent
---

# Holo 3.1 4B generation traces — {args.profile}

This is a **local release candidate**, not a published dataset. It contains {len(index)} completed trace bundles.
Each `traces/<trace-id>/` contains generated token IDs and pieces, capture-position metadata, last-query attention rows,
and optional last-query residual vectors and chosen-token log probabilities. The `manifest.json` records SHA-256 for every file.
Use `index.jsonl` to enumerate the bundles. Run the local viewer with
`python -m apps.trace_viewer.server --root <this-directory>/traces` from the source repository.
{'The index joins 120 pilot and 240 expansion requests to their ScreenSpot sample, image scale, target/control role, and run-manifest hash.' if args.profile == 'screenspot' else 'This review profile includes only local residual traces with an explicit model revision.'}

Raw model requests and responses, machine-specific paths, and quadratic prompt attention matrices are omitted.
The {args.profile} profile {'includes benchmark screenshots' if args.profile == 'screenspot' else 'omits all input screenshots'}.
No model weights are included. The eight conventional attention blocks are mapped to physical layer indices in `model.json`.
Other blocks use linear attention and have no softmax row here. Residual vectors are post-block query states, not causal source attributions.

## Provenance and release review

Model: [Hcompany/Holo-3.1-4B](https://huggingface.co/Hcompany/Holo-3.1-4B).
{'Benchmark: [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro). The screenshot and derived-data redistribution terms still need a final review.' if args.profile == 'screenspot' else 'These captures were made locally. Review prompt and screenshot provenance before making any residual vectors public; latent vectors can retain information about input content.'}
Check benchmark/image rights, privacy, and dataset license before upload. This card deliberately does not assert a new license.
"""
    (target / "README.md").write_text(card)
    summary = {"profile": args.profile, "trace_count": len(index), "staged_bytes": sum(item.stat().st_size for item in (target / "traces").rglob("*") if item.is_file()),
               "linked_files": not args.copy, "source_audit": str(AUDIT.relative_to(ROOT))}
    write_json(target / "stage-summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
