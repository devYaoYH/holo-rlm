"""Bounded Holo CLI plumbing wrapper using a disposable browser profile."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from capture.bundle import BundleWriter
from capture.normalizer import classify_holo_event, omit_hidden_reasoning, read_jsonl, runtime_manifest_note
from capture.report import render_report
from capture.validator import validate_bundle


def run_holo(
    *,
    fixture_url: str,
    backend: str,
    base_url: str,
    model_id: str,
    max_steps: int,
    runs_dir: Path,
    profile_dir: Path,
) -> dict[str, Any]:
    executable = shutil.which("holo")
    if executable is None:
        raise RuntimeError("holo executable not found on PATH")
    browser_template = os.environ.get("HOLO_BROWSER_COMMAND")
    if not browser_template:
        raise RuntimeError("HOLO_BROWSER_COMMAND is required and must contain {url} and {profile}")
    if "{url}" not in browser_template or "{profile}" not in browser_template:
        raise RuntimeError("HOLO_BROWSER_COMMAND must contain both {url} and {profile} placeholders")
    runs_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)
    browser_command = shlex.split(browser_template.format(url=fixture_url, profile=str(profile_dir)))
    if Path(browser_command[0]).name == "open":
        raise RuntimeError("HOLO_BROWSER_COMMAND must launch an attached browser executable, not macOS 'open'")
    browser_process = subprocess.Popen(browser_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    task = (
        "On the already-open StayLocal LOCAL FIXTURE page only: inspect results, scroll down, then find "
        "Harbor Lantern Hotel again and open its local View details link. Never leave localhost. Stop after details open."
    )
    command = [
        executable,
        "run",
        "--runs-dir",
        str(runs_dir),
        "--max-steps",
        str(max_steps),
        "--max-time-s",
        "180",
        task,
    ]
    if backend == "local":
        command[2:2] = ["--base-url", base_url, "--model", model_id]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=240, check=False)
        logs = sorted(runs_dir.rglob("events.jsonl"), key=lambda path: path.stat().st_mtime)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "event_log": str(logs[-1]) if logs else None,
            "backend": backend,
            "replay_ready": False,
            "reason": "runtime stream lacks exact model-input bytes unless correlated with endpoint-side capture",
        }
    finally:
        if browser_process.poll() is None:
            browser_process.terminate()
            try:
                browser_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                browser_process.kill()
                browser_process.wait(timeout=5)


def import_runtime_bundle(
    *,
    event_log: Path | None,
    result: dict[str, Any],
    output_root: Path,
    fixture_config: dict[str, Any],
    fixture_base_url: str,
    fixture_final_state: dict[str, Any],
    backend: str,
    model_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Preserve runtime JSONL and its public projections as a non-replay-ready bundle."""

    writer = BundleWriter(
        output_root,
        backend=backend,
        fixture={
            "version": fixture_config["fixture_version"],
            "seed": fixture_config["seed"],
            "viewport": fixture_config["viewport"],
            "base_url": fixture_base_url,
        },
        task_id="holo-cli-find-harbor-lantern-v1",
        software={"capture": "0.1.0", "holo_event_contract": runtime_manifest_note(None)},
        model={"id": model_id if backend == "local" else "hosted-holo", "revision": None, "processor_revision": None},
        capture_consent=True,
    )
    step = 0
    normalized = 0
    if event_log is not None and event_log.is_file():
        for raw in read_jsonl(event_log):
            safe_raw = omit_hidden_reasoning(raw)
            raw_id = writer.add_raw_event(
                {"source": "holo-runtime-jsonl", "runtime_event": safe_raw, "hidden_reasoning_omitted": True}
            )
            projected = classify_holo_event(raw)
            if projected is None:
                continue
            event_type, data = projected
            if event_type == "model_response":
                current_step: int | None = step
                step += 1
            elif event_type in {"observation", "action_result"}:
                current_step = max(step - 1, 0)
            else:
                current_step = None
            writer.add_event(
                event_type,
                step=current_step,
                data=data,
                raw_event_refs=[raw_id],
                interpretability_replay_ready=False,
            )
            normalized += 1
    terminal_raw = writer.add_raw_event(
        {
            "source": "holo-cli-wrapper",
            "returncode": result["returncode"],
            "event_log_present": event_log is not None,
        }
    )
    writer.add_event(
        "termination",
        step=None,
        data={"terminal_reason": "holo_cli_exit", "returncode": result["returncode"], "replay_ready": False},
        raw_event_refs=[terminal_raw],
        interpretability_replay_ready=False,
    )
    annotations = {
        "target": {
            "hotel": fixture_config["target_name"],
            "expected_destination": fixture_config["expected_destination"],
        },
        "task_success": bool(fixture_final_state.get("details_open")),
        "terminal_reason": "fixture_success" if fixture_final_state.get("details_open") else "holo_cli_exit",
        "final_state": fixture_final_state,
        "replay_ready": False,
        "normalized_runtime_events": normalized,
        "manual_privacy_review": False,
    }
    bundle = writer.finalize(annotations)
    bundle = writer.finalize(annotations, extra_files={"report.html": render_report(bundle)})
    validation = validate_bundle(bundle, require_replay_ready=False)
    return bundle, validation
