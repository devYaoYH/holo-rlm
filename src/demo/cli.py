"""Command-line entry point for doctor, fixture, capture, validation, and Holo plumbing."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from attribution import (
    AttributionError,
    load_attribution,
    resolve_trace_path,
    write_attribution_viewer,
    write_trajectory_viewer,
)
from capture.validator import ValidationError, validate_bundle

from .backends import OpenAIBackend, ScriptedBackend
from .fixture import FixtureClient, running_fixture
from .holo_cli import import_runtime_bundle, run_holo
from .preflight import run_preflight, smoke_backend
from .runner import capture_run

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    run.add_argument(
        "--stop-on-click",
        action="store_true",
        help="finalize the trajectory immediately after the first click, whether or not it hits the task target",
    )
    run.add_argument(
        "--trace-generation-steps",
        type=int,
        choices=range(1, 257),
        default=int(os.environ.get("HOLO_TRACE_GENERATION_STEPS", "4")),
        help="generated token steps retained in each activation trace",
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
    benchmark.add_argument("--count", type=int, choices=range(1, 21), default=20)
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
                task=args.task,
                stop_on_click=args.stop_on_click,
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
        results = []
        for index in range(args.count):
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
                    seed=index % 4,
                    variant=(index // 4) % 5,
                    max_steps=2 if index % 5 == 4 else 8,
                )
            finally:
                close = getattr(backend, "close", None)
                if close:
                    close()
            results.append(
                {
                    "bundle": str(bundle),
                    "seed": index % 4,
                    "variant": (index // 4) % 5,
                    "success": result["annotations"]["task_success"],
                }
            )
        _print({"count": len(results), "successful": sum(item["success"] for item in results), "runs": results})
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


def _rectangle(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("redaction must be X1,Y1,X2,Y2")
    return parts


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
