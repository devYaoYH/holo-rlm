"""Command-line entry point for doctor, fixture, capture, validation, and Holo plumbing."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from apps.booking_fixture.generator import (
    SUPPORTED_SPLITS,
    large_ui_diagnostic,
    load_manifest,
    manifest_item,
    write_manifest,
)
from attribution import (
    AttributionError,
    build_prompt_contrast,
    load_attribution,
    resolve_trace_path,
    write_attribution_viewer,
    write_prompt_contrast_viewer,
    write_trajectory_viewer,
)
from attribution.contrast import parameter_steps
from capture.validator import ValidationError, validate_bundle

from .attribution_batch import run_attribution_batch
from .backends import OpenAIBackend, ScriptedBackend
from .fixture import FixtureClient, running_fixture
from .holo_cli import import_runtime_bundle, run_holo
from .preflight import run_preflight, smoke_backend
from .result_bundle import package_results
from .runner import capture_run
from .screenspot import TRACE_PROFILES, load_screenspot_sample, run_screenspot_case
from .screenspot_benchmark import run_screenspot_benchmark
from .screenspot_resolution import run_resolution_ablation
from .screenspot_resolution_attribution import run_resolution_attribution
from .trajectory_contrast import run_multiframe_prompt_case

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_MANIFEST = PROJECT_ROOT / "benchmarks" / "frozen_eval_v2.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="holo-capture")
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("doctor", "smoke"):
        command = sub.add_parser(name)
        command.add_argument("--backend", choices=("scripted", "local", "hosted"), default=os.environ.get("BACKEND", "scripted"))
    run = sub.add_parser("run")
    run.add_argument("--backend", choices=("scripted", "local"), default=os.environ.get("BACKEND", "scripted"))
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--variant", type=int, choices=range(5), default=0)
    run.add_argument("--max-steps", type=int, default=int(os.environ.get("HOLO_MAX_STEPS", "8")))
    run.add_argument("--task", choices=("target-after-scroll", "cheapest"), default="target-after-scroll")
    run.add_argument("--eval-item", help="item_id from a frozen synthetic evaluation manifest")
    run.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    run.add_argument(
        "--large-ui-diagnostic",
        action="store_true",
        help="enlarge fixture typography for a non-frozen diagnostic replay of --eval-item",
    )
    run.add_argument(
        "--stop-on-click",
        action="store_true",
        help="finalize the trajectory immediately after the first click, whether or not it hits the task target",
    )
    run.add_argument(
        "--trace-generation-steps",
        type=int,
        choices=range(0, 257),
        default=int(os.environ.get("HOLO_TRACE_GENERATION_STEPS", "4")),
        help="generated token steps retained in each activation trace; 0 disables tracing",
    )
    run.add_argument("--redact", action="append", default=[], metavar="X1,Y1,X2,Y2")
    holo = sub.add_parser("holo")
    holo.add_argument("--backend", choices=("local", "hosted"), default=os.environ.get("BACKEND", "local"))
    holo.add_argument("--seed", type=int, default=0)
    holo.add_argument("--max-steps", type=int, default=int(os.environ.get("HOLO_MAX_STEPS", "8")))
    reset = sub.add_parser("reset")
    reset.add_argument("--seed", type=int, default=0)
    reset.add_argument("--variant", type=int, choices=range(5), default=0)
    reset.add_argument("--url", default=f"http://127.0.0.1:{os.environ.get('HOLO_FIXTURE_PORT', '8765')}")
    validate = sub.add_parser("validate")
    validate.add_argument("path", type=Path)
    validate.add_argument("--allow-runtime-only", action="store_true")
    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--backend", choices=("scripted", "local"), default=os.environ.get("BACKEND", "scripted"))
    benchmark.add_argument("--manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    benchmark.add_argument("--offset", type=int, default=0)
    benchmark.add_argument("--count", type=int, default=20)
    generate_eval = sub.add_parser("generate-eval")
    generate_eval.add_argument("--split", choices=SUPPORTED_SPLITS, default="test")
    generate_eval.add_argument("--count", type=int, default=120)
    generate_eval.add_argument("--output", type=Path, default=DEFAULT_EVAL_MANIFEST)
    attribution = sub.add_parser("attribution")
    attribution.add_argument("path", type=Path, help="instrumented trace or trajectory bundle")
    attribution.add_argument("--trace-index", type=int, default=0, help="trace within a multi-step trajectory")
    attribution.add_argument("--trace-root", type=Path, default=PROJECT_ROOT / "data" / "traces")
    attribution.add_argument("--output", type=Path)
    attribution.add_argument(
        "--all-frames",
        action="store_true",
        help="for a trajectory bundle, build a multi-frame action viewer and every linked trace viewer",
    )
    attribution_batch = sub.add_parser("attribution-batch")
    attribution_batch.add_argument("manifest", type=Path)
    attribution_batch.add_argument("--trace-root", type=Path, default=PROJECT_ROOT / "data" / "traces")
    attribution_batch.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data" / "attributions" / "batch"
    )
    attribution_batch.add_argument("--no-resume", action="store_true")
    package = sub.add_parser("package-results")
    package.add_argument("run", type=Path)
    package.add_argument("--trace-root", type=Path, default=PROJECT_ROOT / "data" / "traces")
    package.add_argument("--output", type=Path, required=True)
    screenspot = sub.add_parser("screenspot-case")
    screenspot.add_argument("sample_id")
    screenspot.add_argument("--annotations", type=Path, default=PROJECT_ROOT / "data" / "screenspot-pro" / "annotations")
    screenspot.add_argument("--images", type=Path, default=PROJECT_ROOT / "data" / "screenspot-pro" / "images")
    screenspot.add_argument("--control-prompt", action="append", default=[])
    screenspot.add_argument("--trace-generation-steps", type=int, choices=range(0, 257), default=0)
    screenspot.add_argument("--trace-profile", choices=TRACE_PROFILES)
    screenspot.add_argument("--output", type=Path)
    screenspot_benchmark = sub.add_parser("screenspot-benchmark")
    screenspot_benchmark.add_argument(
        "--annotations", type=Path, default=PROJECT_ROOT / "data" / "screenspot-pro" / "annotations"
    )
    screenspot_benchmark.add_argument(
        "--images", type=Path, default=PROJECT_ROOT / "data" / "screenspot-pro" / "images"
    )
    screenspot_benchmark.add_argument(
        "--trace-root", type=Path, default=PROJECT_ROOT / "data" / "traces"
    )
    screenspot_benchmark.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data" / "screenspot-pro" / "benchmark-runs" / "full-4b"
    )
    screenspot_benchmark.add_argument("--trace-profile", choices=TRACE_PROFILES, default="logprobs")
    screenspot_benchmark.add_argument("--trace-generation-steps", type=int, choices=range(1, 257), default=64)
    screenspot_benchmark.add_argument("--shard-index", type=int, default=0)
    screenspot_benchmark.add_argument("--num-shards", type=int, default=1)
    screenspot_benchmark.add_argument("--offset", type=int, default=0)
    screenspot_benchmark.add_argument("--count", type=int)
    screenspot_benchmark.add_argument("--no-resume", action="store_true")
    screenspot_benchmark.add_argument("--fail-fast", action="store_true")
    resolution = sub.add_parser("screenspot-resolution-ablation")
    resolution.add_argument("manifest", type=Path)
    resolution.add_argument(
        "--annotations", type=Path, default=PROJECT_ROOT / "data" / "screenspot-pro" / "annotations"
    )
    resolution.add_argument(
        "--images",
        type=Path,
        default=PROJECT_ROOT / "data" / "screenspot-pro" / "resolution-ablation-images",
    )
    resolution.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "screenspot-pro" / "resolution-ablation-runs" / "success-retention-v1",
    )
    resolution.add_argument("--no-resume", action="store_true")
    resolution_attribution = sub.add_parser("screenspot-resolution-attribution")
    resolution_attribution.add_argument("manifest", type=Path)
    resolution_attribution.add_argument(
        "--annotations", type=Path, default=PROJECT_ROOT / "data" / "screenspot-pro" / "annotations"
    )
    resolution_attribution.add_argument(
        "--images",
        type=Path,
        default=PROJECT_ROOT / "data" / "screenspot-pro" / "resolution-ablation-images",
    )
    resolution_attribution.add_argument("--trace-root", type=Path, default=PROJECT_ROOT / "data" / "traces")
    resolution_attribution.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "screenspot-pro" / "resolution-attribution-runs" / "pilot-v1",
    )
    resolution_attribution.add_argument("--offset", type=int, default=0)
    resolution_attribution.add_argument("--count", type=int)
    resolution_attribution.add_argument("--no-resume", action="store_true")
    contrast = sub.add_parser("screenspot-contrast")
    contrast.add_argument("case", type=Path, help="case.json written by screenspot-case")
    contrast.add_argument("--trace-root", type=Path, default=PROJECT_ROOT / "data" / "traces")
    contrast.add_argument("--output", type=Path)
    trajectory_case = sub.add_parser("trajectory-prompt-case")
    trajectory_case.add_argument("source_trace", type=Path)
    trajectory_case.add_argument("--target-instruction", required=True)
    trajectory_case.add_argument("--control-prompt", action="append", default=[])
    trajectory_case.add_argument("--target-bbox", required=True, metavar="X1,Y1,X2,Y2")
    trajectory_case.add_argument(
        "--frame-bbox",
        action="append",
        default=[],
        metavar="X1,Y1,X2,Y2|none",
        help="one target box or 'none' per retained frame",
    )
    trajectory_case.add_argument("--trace-generation-steps", type=int, choices=range(1, 257), default=64)
    trajectory_case.add_argument(
        "--max-frame-width",
        type=int,
        default=None,
        help="uniformly downsample every retained frame before matched tracing",
    )
    trajectory_case.add_argument("--output", type=Path)
    trajectory_contrast = sub.add_parser("trajectory-prompt-contrast")
    trajectory_contrast.add_argument("case", type=Path, help="case.json written by trajectory-prompt-case")
    trajectory_contrast.add_argument("--trace-root", type=Path, default=PROJECT_ROOT / "data" / "traces")
    trajectory_contrast.add_argument("--output", type=Path)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    base_url = os.environ.get("HOLO_BASE_URL", "http://127.0.0.1:8000/v1")
    model_id = os.environ.get("HOLO_MODEL", "Hcompany/Holo-3.1-4B")
    if args.command == "doctor":
        manifest, path = run_preflight(args.backend, base_url=base_url, model_id=model_id, output_root=PROJECT_ROOT / "data" / "manifests")
        _print({"manifest": str(path), **manifest})
        if not manifest["holo"]["available"]:
            raise SystemExit(1)
        if args.backend == "local" and not manifest["endpoint"].get("reachable"):
            raise SystemExit(1)
    elif args.command == "smoke":
        result = smoke_backend(args.backend, base_url=base_url, model_id=model_id)
        _print(result)
        if not result["ok"]:
            raise SystemExit(1)
    elif args.command == "run":
        scenario_config = None
        task = args.task
        if args.eval_item:
            scenario_config = manifest_item(load_manifest(args.eval_manifest), args.eval_item)
            if args.large_ui_diagnostic:
                scenario_config = large_ui_diagnostic(scenario_config)
            task = "cheapest"
        elif args.large_ui_diagnostic:
            raise SystemExit("--large-ui-diagnostic requires --eval-item")
        backend = ScriptedBackend() if args.backend == "scripted" else OpenAIBackend(
            base_url,
            model_id,
            os.environ.get("HOLO_MODEL_REVISION", "local-checkpoint"),
            os.environ.get("HOLO_PROCESSOR_REVISION", "local-checkpoint"),
            args.trace_generation_steps,
        )
        rectangles = tuple(_rectangle(value) for value in args.redact)
        try:
            bundle, result = capture_run(
                backend=backend,
                output_root=PROJECT_ROOT / "data" / "trajectories" / "v0",
                seed=args.seed,
                variant=args.variant,
                max_steps=args.max_steps,
                task=task,
                stop_on_click=args.stop_on_click,
                scenario_config=scenario_config,
                redactions=rectangles,
            )
        finally:
            close = getattr(backend, "close", None)
            if close:
                close()
        _print({"bundle": str(bundle), **result})
    elif args.command == "holo":
        if os.environ.get("HOLO_CAPTURE_CONSENT") != "I_HAVE_ISOLATED_THE_DESKTOP":
            raise SystemExit(
                "live capture refused: set HOLO_CAPTURE_CONSENT=I_HAVE_ISOLATED_THE_DESKTOP only after "
                "closing unrelated apps and confirming no permission prompt is visible"
            )
        runs_dir = PROJECT_ROOT / ".holo-runs"
        profile = PROJECT_ROOT / "browser-profiles" / "holo-fixture"
        with running_fixture(seed=args.seed) as fixture:
            fixture_config = fixture.config()
            result = run_holo(
                fixture_url=f"{fixture.base_url}/?seed={args.seed}",
                backend=args.backend,
                base_url=base_url,
                model_id=model_id,
                max_steps=args.max_steps,
                runs_dir=runs_dir,
                profile_dir=profile,
            )
            result["fixture_state"] = fixture.state()
            event_log = Path(result["event_log"]) if result["event_log"] else None
            bundle, validation = import_runtime_bundle(
                event_log=event_log,
                result=result,
                output_root=PROJECT_ROOT / "data" / "trajectories" / "v0",
                fixture_config=fixture_config,
                fixture_base_url=fixture.base_url,
                fixture_final_state=result["fixture_state"],
                backend=args.backend,
                model_id=model_id,
            )
            result["bundle"] = str(bundle)
            result["validation"] = validation
        _print(result)
        if result["returncode"]:
            raise SystemExit(result["returncode"])
    elif args.command == "reset":
        client = FixtureClient(args.url)
        try:
            _print(client.reset(args.seed, args.variant))
        finally:
            client.close()
    elif args.command == "validate":
        paths = [args.path] if (args.path / "manifest.json").exists() else sorted(args.path.glob("*/"))
        if not paths:
            raise SystemExit(f"no bundles found under {args.path}")
        results = []
        try:
            for path in paths:
                annotations_path = path / "annotations.json"
                annotations = json.loads(annotations_path.read_text()) if annotations_path.exists() else {}
                is_labeled_runtime_only = annotations.get("replay_ready") is False
                require_replay_ready = not (args.allow_runtime_only and is_labeled_runtime_only)
                results.append(validate_bundle(path, require_replay_ready=require_replay_ready))
        except ValidationError as exc:
            print(f"validation failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        _print(results)
    elif args.command == "benchmark":
        if args.offset < 0 or args.count < 1:
            raise SystemExit("benchmark offset must be non-negative and count must be positive")
        manifest = load_manifest(args.manifest)
        selected = manifest["items"][args.offset : args.offset + args.count]
        if not selected:
            raise SystemExit("benchmark selection is empty")
        results = []
        for item in selected:
            backend = ScriptedBackend() if args.backend == "scripted" else OpenAIBackend(
                base_url,
                model_id,
                os.environ.get("HOLO_MODEL_REVISION", "local-checkpoint"),
                os.environ.get("HOLO_PROCESSOR_REVISION", "local-checkpoint"),
            )
            try:
                bundle, result = capture_run(
                    backend=backend,
                    output_root=PROJECT_ROOT / "data" / "trajectories" / "v0",
                    seed=item["config"]["item_index"],
                    max_steps=16,
                    task="cheapest",
                    stop_on_click=True,
                    scenario_config=item["config"],
                )
            finally:
                close = getattr(backend, "close", None)
                if close:
                    close()
            results.append(
                {
                    "bundle": str(bundle),
                    "item_id": item["item_id"],
                    "success": result["annotations"]["task_success"],
                }
            )
        _print({"count": len(results), "successful": sum(item["success"] for item in results), "runs": results})
    elif args.command == "generate-eval":
        if args.count < 1:
            raise SystemExit("count must be positive")
        output = write_manifest(args.output, split=args.split, count=args.count)
        manifest = load_manifest(output)
        _print(
            {
                "output": str(output),
                "generator_version": manifest["generator_version"],
                "split": manifest["split"],
                "count": manifest["count"],
                "diversity": manifest["diversity"],
            }
        )
    elif args.command == "attribution":
        try:
            if args.all_frames:
                output = args.output or PROJECT_ROOT / "data" / "attributions" / args.path.name
                _print(write_trajectory_viewer(args.path, args.trace_root, output))
            else:
                trace_path = resolve_trace_path(args.path, trace_root=args.trace_root, trace_index=args.trace_index)
                attribution = load_attribution(trace_path)
                output = args.output or PROJECT_ROOT / "data" / "attributions" / trace_path.name
                _print(write_attribution_viewer(attribution, output))
        except AttributionError as exc:
            print(f"attribution failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
    elif args.command == "attribution-batch":
        _print(
            run_attribution_batch(
                args.manifest,
                trace_root=args.trace_root,
                output_root=args.output,
                resume=not args.no_resume,
            )
        )
    elif args.command == "package-results":
        _print(package_results(args.run, args.trace_root, args.output))
    elif args.command == "screenspot-case":
        sample = load_screenspot_sample(args.annotations, args.images, args.sample_id)
        output = args.output or PROJECT_ROOT / "data" / "screenspot-pro" / "runs" / args.sample_id
        _print(
            run_screenspot_case(
                sample=sample,
                controls=args.control_prompt,
                base_url=base_url,
                model_id=model_id,
                output_dir=output,
                trace_generation_steps=args.trace_generation_steps,
                trace_profile=args.trace_profile,
            )
        )
    elif args.command == "screenspot-benchmark":
        _print(
            run_screenspot_benchmark(
                annotation_root=args.annotations,
                image_root=args.images,
                base_url=base_url,
                model_id=model_id,
                trace_root=args.trace_root,
                output_dir=args.output,
                trace_profile=args.trace_profile,
                trace_generation_steps=args.trace_generation_steps,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                offset=args.offset,
                count=args.count,
                resume=not args.no_resume,
                fail_fast=args.fail_fast,
            )
        )
    elif args.command == "screenspot-resolution-ablation":
        _print(
            run_resolution_ablation(
                manifest_path=args.manifest,
                annotation_root=args.annotations,
                image_root=args.images,
                base_url=base_url,
                model_id=model_id,
                output_dir=args.output,
                resume=not args.no_resume,
            )
        )
    elif args.command == "screenspot-resolution-attribution":
        _print(
            run_resolution_attribution(
                manifest_path=args.manifest,
                annotation_root=args.annotations,
                image_root=args.images,
                base_url=base_url,
                model_id=model_id,
                trace_root=args.trace_root,
                output_dir=args.output,
                offset=args.offset,
                count=args.count,
                resume=not args.no_resume,
            )
        )
    elif args.command == "screenspot-contrast":
        case = json.loads(args.case.read_text())
        target_run = next(run for run in case["runs"] if run["role"] == "target")
        control_runs = tuple(run for run in case["runs"] if run["role"].startswith("control-"))
        if not control_runs:
            raise SystemExit("screenspot contrast requires at least one traced control prompt")
        trace_ids = [target_run.get("trace_id"), *[run.get("trace_id") for run in control_runs]]
        if not all(isinstance(trace_id, str) for trace_id in trace_ids):
            raise SystemExit("screenspot contrast requires target and control activation trace IDs")
        target = load_attribution(args.trace_root / trace_ids[0])
        controls = tuple(load_attribution(args.trace_root / trace_id) for trace_id in trace_ids[1:])
        comparison = build_prompt_contrast(
            target,
            controls,
            control_instructions=tuple(run["instruction"] for run in control_runs),
        )
        sample = case["sample"]
        click_record = target_run.get("click") or target_run.get("repaired_click")
        if not isinstance(click_record, dict):
            raise SystemExit("screenspot contrast requires a valid or deterministically repaired target click")
        click = click_record["pixel"]
        output = args.output or PROJECT_ROOT / "data" / "attributions" / f"screenspot-{sample['id']}"
        _print(
            write_prompt_contrast_viewer(
                comparison,
                output,
                sample_id=sample["id"],
                target_instruction=sample["instruction"],
                bbox=tuple(float(value) for value in sample["bbox"]),
                predicted_click=(float(click["x"]), float(click["y"])),
                correct=bool(target_run.get("grounding_correct", target_run["correct"])),
                format_valid=bool(target_run.get("format_valid", True)),
            )
        )
    elif args.command == "trajectory-prompt-case":
        if not args.control_prompt:
            raise SystemExit("trajectory prompt case requires at least one --control-prompt")
        frame_bboxes = tuple(_optional_rectangle(value) for value in args.frame_bbox)
        output = args.output or PROJECT_ROOT / "data" / "trajectory-prompt-cases" / args.source_trace.name
        _print(
            run_multiframe_prompt_case(
                source_trace=args.source_trace,
                target_instruction=args.target_instruction,
                controls=args.control_prompt,
                target_bbox=_rectangle(args.target_bbox),
                frame_bboxes=frame_bboxes,
                base_url=base_url,
                output_dir=output,
                trace_generation_steps=args.trace_generation_steps,
                max_frame_width=args.max_frame_width,
            )
        )
    elif args.command == "trajectory-prompt-contrast":
        case = json.loads(args.case.read_text())
        target_run = next(run for run in case["runs"] if run["role"] == "target")
        control_runs = tuple(run for run in case["runs"] if run["role"].startswith("control-"))
        if not control_runs:
            raise SystemExit("trajectory contrast requires at least one traced control prompt")
        trace_ids = [target_run.get("trace_id"), *[run.get("trace_id") for run in control_runs]]
        if not all(isinstance(trace_id, str) for trace_id in trace_ids):
            raise SystemExit("trajectory contrast requires target and control activation trace IDs")
        target = load_attribution(args.trace_root / trace_ids[0])
        compatible_controls = []
        compatible_runs = []
        excluded_controls = []
        for run, trace_id in zip(control_runs, trace_ids[1:], strict=True):
            control = load_attribution(args.trace_root / trace_id)
            try:
                parameter_steps(control, ("x", "y"))
            except AttributionError as exc:
                excluded_controls.append(
                    {"instruction": run["instruction"], "trace_id": trace_id, "reason": str(exc)}
                )
            else:
                compatible_controls.append(control)
                compatible_runs.append(run)
        controls = tuple(compatible_controls)
        control_runs = tuple(compatible_runs)
        if not controls:
            raise SystemExit("trajectory contrast has no controls with both x and y token spans")
        comparison = build_prompt_contrast(
            target,
            controls,
            control_instructions=tuple(run["instruction"] for run in control_runs),
        )
        click_record = target_run.get("click")
        if not isinstance(click_record, dict):
            raise SystemExit("trajectory contrast requires a schema-valid target click")
        click = click_record["pixel"]
        output = args.output or PROJECT_ROOT / "data" / "attributions" / f"trajectory-contrast-{args.case.parent.name}"
        frame_bboxes = tuple(
            tuple(float(value) for value in bbox) if bbox is not None else None
            for bbox in case["frame_bboxes"]
        )
        _print(
            write_prompt_contrast_viewer(
                comparison,
                output,
                sample_id=output.name,
                target_instruction=case["target_instruction"],
                bbox=tuple(float(value) for value in case["target_bbox"]),
                predicted_click=(float(click["x"]), float(click["y"])),
                correct=bool(target_run.get("grounding_correct", target_run.get("correct", False))),
                format_valid=bool(target_run.get("format_valid", True)),
                source_label="Synthetic hotel trajectory",
                frame_bboxes=frame_bboxes,
                excluded_controls=tuple(excluded_controls),
            )
        )


def _rectangle(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("redaction must be X1,Y1,X2,Y2")
    return parts


def _optional_rectangle(value: str) -> tuple[int, int, int, int] | None:
    if value.strip().lower() == "none":
        return None
    return _rectangle(value)


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
