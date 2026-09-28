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
HOTEL_BUNDLE = DATA / "remote-results/hotel-freegen-paired-v1-20260922/hotel-freegen-paired-v1/holo-traced/bundles/test-0010"
HOTEL_TRACE_SOURCE = "remote-results/hotel-freegen-paired-v1-20260922/hotel-freegen-paired-v1/holo-traced/server-traces"
HOTEL_ACTION_BUNDLE = DATA / "remote-results/hotel-official-tools-native-20260922-rerun/extracted/run/traj-20260922T055823Z-397ab3c2af"
HOTEL_ACTION_TRACE_SOURCE = "remote-results/hotel-official-tools-native-20260922-rerun/extracted/traces"


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
    parser.add_argument("--profile", choices=("screenspot", "residual-review", "hotel-trajectory-review", "hotel-action-review"), default="screenspot")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--copy", action="store_true", help="Copy instead of local hard links")
    parser.add_argument("--public-card", action="store_true", help="Use the reviewed MIT public ScreenSpot card")
    args = parser.parse_args()
    if args.public_card and args.profile != "screenspot":
        parser.error("--public-card currently applies only to screenspot")
    audit = read(AUDIT)
    if not audit.get("hashes_verified"):
        parser.error("run audit_trace_release.py --verify-hashes first")
    hotel_profile = args.profile in ("hotel-trajectory-review", "hotel-action-review")
    hotel_bundle = HOTEL_ACTION_BUNDLE if args.profile == "hotel-action-review" else HOTEL_BUNDLE
    hotel_source = HOTEL_ACTION_TRACE_SOURCE if args.profile == "hotel-action-review" else HOTEL_TRACE_SOURCE
    hotel_annotations = read(hotel_bundle / "annotations.json") if hotel_profile else {}
    hotel_trace_ids = hotel_annotations.get("instrumented_trace_ids", [])
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
        elif hotel_profile:
            if row["source"] != hotel_source or row["id"] not in hotel_trace_ids:
                continue
        else:
            if row["source"] != "traces" or row["captured_hidden_steps"] == 0 or not row["model_revision"]:
                continue
        candidates.append(row)
    if hotel_profile and len(candidates) != len(hotel_trace_ids):
        raise ValueError("hotel trajectory and audited traces differ")
    index = []
    for row in candidates:
        source = DATA / row["source"] / row["id"]
        out = target / "traces" / row["id"]
        out.mkdir(parents=True, exist_ok=True)
        names = set(COMMON)
        if args.profile == "screenspot" or hotel_profile:
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
                      "has_logprobs": row["has_logprobs"], "has_input_image": (args.profile == "screenspot" or hotel_profile) and row["has_input_image"],
                      **({"trajectory_step": hotel_trace_ids.index(row["id"])} if hotel_profile else {}),
                      **benchmark_provenance.get(row["id"], {})})
    (target / "index.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in index))
    if hotel_profile:
        frames = target / "frames"
        steps = []
        for step, trace_id in enumerate(hotel_trace_ids):
            action = read(hotel_bundle / "actions" / f"{step:04d}.json")["normalized"]
            frame_names = (f"{step:04d}-model-input.png", f"{step:04d}-post-action.png")
            for name in frame_names:
                stage_file(hotel_bundle / "frames" / name, frames / name, copy=args.copy)
            steps.append({"step": step, "trace_id": trace_id, "action": action,
                          "input_frame": f"frames/{frame_names[0]}", "post_action_frame": f"frames/{frame_names[1]}"})
        write_json(target / "trajectory.json", {
            "trajectory_id": read(hotel_bundle / "manifest.json")["trajectory_id"],
            "fixture": {key: read(hotel_bundle / "manifest.json")["fixture"][key]
                        for key in ("generator_version", "dataset_split", "item_id")},
            "task": "Inspect all hotel prices, then open the cheapest hotel",
            "success": hotel_annotations["task_success"], "terminal_reason": hotel_annotations["terminal_reason"],
            "manual_privacy_review": hotel_annotations["manual_privacy_review"], "steps": steps,
        })
        top_files = [target / "trajectory.json", *sorted(frames.iterdir())]
        write_json(target / "trajectory-manifest.json", {"schema_version": 1, "files": [
            {"path": str(item.relative_to(target)), "bytes": item.stat().st_size, "sha256": digest(item)}
            for item in top_files]})
    if args.profile == "hotel-trajectory-review":
        detail = """This candidate contains one successful three-action trajectory on a synthetic hotel page: scroll, scroll, then click the cheapest result. `trajectory.json` links ordered actions and before/after frames to the three generation traces. It omits raw requests, responses, and event logs. All three traces have 8 conventional attention layers and 64 captured generation steps, while their complete responses contain 302, 282, and 363 tokens respectively. Tokens after the captured prefix have no saved attention rows. There are no residual vectors or chosen-token log probabilities. The source bundle validates as replay ready, but its `manual_privacy_review` flag is false; review frames and generated text before any upload. This is one replicate, not a representative performance sample."""
    elif args.profile == "hotel-action-review":
        detail = """This candidate contains one successful four-action Holo trajectory on a synthetic large-UI hotel diagnostic: three scrolls, then a click. `trajectory.json` links ordered actions and before/after frames to four generation traces. The first two traces capture attention for all 215 and 213 generated tokens. The third saves 256 of 355 token steps. The final click saves 256 of 262; its click x value is at captured steps 246-248 and y at 252-254. These are observational attention rows from Holo's eight conventional full-attention layers. The model's linear-attention blocks have no matching softmax rows. There are no residual vectors or chosen-token log probabilities. This is a single diagnostic trajectory, not a paired Holo/Qwen comparison or a representative benchmark result. The source bundle's `manual_privacy_review` flag is false; review frames and generated text before any upload."""
    elif args.profile == "screenspot":
        detail = "The index joins 120 pilot and 240 expansion requests to their ScreenSpot sample, image scale, target/control role, and run-manifest hash."
    else:
        detail = "This review profile includes only local residual traces with an explicit model revision."
    screenspot_notes = """
## ScreenSpot-Pro subset and capture

This selected research subset has 18 ScreenSpot-Pro items from Photoshop, PowerPoint, and VS Code. For each item, Holo was run at four linear image scales (100%, 75%, 50%, 25%) with one target instruction and four same-image control instructions: 18 x 4 x 5 = 360 traces. It is not a benchmark-wide accuracy estimate. All 360 traces use Holo model revision `8c88265a5a159bfd1492db9243733dd2e6e04a6e` and save chosen-token log probabilities and last-query attention rows for every generated token. They do not save hidden-state vectors.

Each line of `index.jsonl` identifies a trace, benchmark item, application, image scale, target/control role, bounding box, predicted click, and strict correctness label. Each trace directory has `generated_tokens.json`, `token_logprobs.json`, image inputs, attention rows, value norms, and a SHA-256 manifest. The `attention_last_query_rows.npz` file stores query-to-key rows for the eight conventional full-attention layers. Holo's other language blocks use linear attention and do not yield equivalent softmax rows in this capture. Attention visualizations describe routing; they do not establish causal influence.

## Use

From the [source repository](https://github.com/devYaoYH/holo-rlm), run `python3 -m apps.trace_viewer.server --root /path/to/this/dataset/traces` and open the address printed by the server. The viewer steps through generated tokens, layers, heads, and image patches. `scripts/verify_hf_stage.py` checks the staged files against their manifests. These traces are intended for evaluation and interpretability research; using benchmark examples for training may compromise later benchmark evaluation.

## Attribution

The screenshots and benchmark annotations originate from [likaixin/ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro), whose Hub card labels the dataset MIT. The benchmark authors provide the [dataset citation](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro#citation). Holo model information is at [Hcompany/Holo-3.1-4B](https://huggingface.co/Hcompany/Holo-3.1-4B). This dataset does not claim ownership of the benchmark screenshots or model weights.
""" if args.profile == "screenspot" else ""
    license_line = "license: mit\n" if args.public_card else ""
    release_status = (
        "This dataset is released under MIT. Its benchmark screenshots and annotations come from the upstream ScreenSpot-Pro dataset, whose Hub card also lists MIT; the original authors are credited below."
        if args.public_card else
        "This repository is private while release terms and privacy are reviewed. No separate license has been asserted for these traces yet."
    )
    card = f"""---
pretty_name: Holo 3.1 4B generation traces ({args.profile})
{license_line}language:
- en
tags:
- interpretability
- multimodal
- gui-agent
---

# Holo 3.1 4B generation traces — {args.profile}

This dataset contains {len(index)} completed trace bundles.
Each `traces/<trace-id>/` contains generated token IDs and pieces, capture-position metadata, last-query attention rows,
and optional last-query residual vectors and chosen-token log probabilities. The `manifest.json` records SHA-256 for every file.
Use `index.jsonl` to enumerate the bundles. Run the local viewer with
`python -m apps.trace_viewer.server --root <this-directory>/traces` from the source repository.
{detail}

Raw model requests and responses, machine-specific paths, and quadratic prompt attention matrices are omitted.
The {args.profile} profile {'omits all input screenshots' if args.profile == 'residual-review' else 'includes input screenshots'}.
No model weights are included. The eight conventional attention blocks are mapped to physical layer indices in `model.json`.
Other blocks use linear attention and have no softmax row here. Residual vectors are post-block query states, not causal source attributions.

## Provenance and release status

Model: [Hcompany/Holo-3.1-4B](https://huggingface.co/Hcompany/Holo-3.1-4B).
{'Benchmark: [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro). Its Hub card labels the source dataset MIT.' if args.profile == 'screenspot' else 'These captures were made locally. Review prompt and screenshot provenance before public release.'}
{release_status}
{screenspot_notes}
"""
    (target / "README.md").write_text(card)
    summary = {"profile": args.profile, "trace_count": len(index), "staged_bytes": sum(item.stat().st_size for item in target.rglob("*") if item.is_file()),
               "linked_files": not args.copy, "source_audit": str(AUDIT.relative_to(ROOT))}
    write_json(target / "stage-summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
