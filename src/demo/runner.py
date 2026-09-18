"""Bounded capture loop and deterministic action replay."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import ImageDraw

from capture.bundle import BundleWriter, canonical_json, sha256_bytes
from capture.report import render_report
from capture.schema import ACTION_SCHEMA
from capture.validator import validate_bundle

from .backends import ActionBackend
from .fixture import running_fixture
from .renderer import render_fixture

INITIAL_TASK = "Inspect the deterministic booking results, note the options, and scroll down. Do not open a details link yet."


def _artifact(path: Path, bundle: Path, role: str) -> dict[str, Any]:
    data = path.read_bytes()
    result = {"path": path.relative_to(bundle).as_posix(), "sha256": sha256_bytes(data), "role": role}
    if path.suffix == ".png":
        from PIL import Image

        with Image.open(path) as image:
            result.update({"width": image.width, "height": image.height})
    return result


def capture_run(
    *,
    backend: ActionBackend,
    output_root: Path,
    seed: int,
    variant: int = 0,
    max_steps: int = 8,
    redactions: tuple[tuple[int, int, int, int], ...] = (),
) -> tuple[Path, dict[str, Any]]:
    with running_fixture(seed=seed, variant=variant) as fixture:
        config = fixture.config()
        initial_state = fixture.reset(seed, variant)
        follow_up_task = f"Now locate {config['target_name']} again and open its local View details link."
        writer = BundleWriter(
            output_root,
            backend=backend.name,
            fixture={
                "version": config["fixture_version"],
                "seed": config["seed"],
                "variant": config["variant"],
                "viewport": config["viewport"],
                "base_url": fixture.base_url,
            },
            task_id=f"find-target-after-scroll-v1-variant-{config['variant']}",
            software={"python": sys.version.split()[0], "platform": platform.platform(), "capture": "0.1.0"},
            model={
                "id": backend.model_id,
                "revision": backend.model_revision,
                "processor_revision": backend.processor_revision,
                "processor_config": {"renderer": "fixture-renderer-v1", "image_format": "png", "color_mode": "RGB"},
            },
            capture_consent=True,
            redactions=redactions,
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": INITIAL_TASK}]
        _event(writer, "user_turn", None, {"message": INITIAL_TASK, "turn": 0})
        success = False
        terminal_reason = "bounded_step_limit"
        target_seen_before_scroll = False
        target_interval: list[int] = []
        second_turn_step = 1
        invalid_error: str | None = None
        completed_steps = 0
        activation_trace_ids: list[str] = []

        for step in range(max_steps):
            state = fixture.state()
            image = render_fixture(config, state)
            if redactions:
                image = image.copy()
                draw = ImageDraw.Draw(image)
                for rectangle in redactions:
                    draw.rectangle(rectangle, fill=(0, 0, 0))
            frame_relative = f"frames/{step:04d}-model-input.png"
            image_meta = writer.write_image(frame_relative, image)
            image_ref = {**image_meta, "role": "model_input"}
            target_visible = _target_visible(config, state)
            if target_visible:
                target_interval.append(step)
            if step == 0 and target_visible:
                target_seen_before_scroll = True
            _event(
                writer,
                "observation",
                step,
                {"state": state, "target_visible": target_visible, "source": "exact-renderer-model-input"},
                [image_ref],
                replay_ready=True,
            )
            try:
                request, response, action = backend.decide(
                    step=step, messages=messages, image=image, config=config, state=state
                )
            except Exception as exc:
                invalid_error = f"{type(exc).__name__}: {exc}"
                terminal_reason = "backend_or_action_error"
                _event(writer, "termination", None, {"terminal_reason": terminal_reason, "error": invalid_error})
                break

            request_relative = f"requests/{step:04d}.json"
            response_relative = f"responses/{step:04d}.json"
            action_relative = f"actions/{step:04d}.json"
            writer.write_json(request_relative, request)
            writer.write_json(response_relative, response)
            writer.write_json(action_relative, {"raw_output": response, "normalized": action})
            request_ref = _artifact(writer.path / request_relative, writer.path, "model_request")
            response_ref = _artifact(writer.path / response_relative, writer.path, "model_response")
            action_ref = _artifact(writer.path / action_relative, writer.path, "proposed_action")
            activation_trace_id = response.get("instrumented_trace_id")
            if isinstance(activation_trace_id, str):
                activation_trace_ids.append(activation_trace_id)
            _event(
                writer,
                "model_request",
                step,
                {
                    "messages": request["messages"],
                    "image_refs": [{"path": image_ref["path"], "sha256": image_ref["sha256"]}],
                    "tool_schema_hash": hashlib.sha256(canonical_json(ACTION_SCHEMA)).hexdigest(),
                    "sampling": {"temperature": request.get("temperature"), "max_tokens": request.get("max_tokens")},
                    "generation": {"tool_choice": request.get("tool_choice")},
                    "model_id": backend.model_id,
                    "model_revision": backend.model_revision,
                    "processor_revision": backend.processor_revision,
                    "processor_configuration": {"renderer": "fixture-renderer-v1", "color_mode": "RGB"},
                    "conversation_id": writer.trajectory_id,
                    "session_id": None,
                    "reasoning_not_retained": True,
                },
                [request_ref, image_ref],
                replay_ready=True,
            )
            _event(
                writer,
                "model_response",
                step,
                {"externally_usable_response": response, "hidden_reasoning_retained": False},
                [response_ref],
                replay_ready=True,
                interpretability={"activation_trace_id": activation_trace_id},
            )
            _event(
                writer,
                "proposed_action",
                step,
                {"raw_output": response, "normalized": action},
                [action_ref],
                replay_ready=True,
            )
            before = fixture.state()
            after = fixture.apply(action)
            post_image = render_fixture(config, after)
            post_relative = f"frames/{step:04d}-post-action.png"
            post_meta = writer.write_image(post_relative, post_image)
            post_ref = {**post_meta, "role": "post_action"}
            action_success = action["action"] == "finish" or before != after
            fixture_assertion = bool(after.get("details_open") and after.get("destination") == config["expected_destination"])
            _event(
                writer,
                "action_result",
                step,
                {
                    "success": action_success,
                    "before_state": before,
                    "after_state": after,
                    "destination_url": f"{fixture.base_url}{after['destination']}",
                    "fixture_assertion": fixture_assertion,
                },
                [post_ref],
                replay_ready=True,
            )
            completed_steps += 1
            messages.append({"role": "assistant", "content": json.dumps(action, sort_keys=True)})
            if fixture_assertion:
                success = True
                terminal_reason = "fixture_success"
                _event(writer, "termination", None, {"terminal_reason": terminal_reason, "success": True})
                break
            if step == 0:
                messages.append({"role": "user", "content": follow_up_task})
                _event(
                    writer,
                    "user_turn",
                    None,
                    {"message": follow_up_task, "turn": 1, "occurs_before_step": second_turn_step},
                )
            if action["action"] == "finish":
                terminal_reason = "model_finish_without_fixture_success"
                _event(writer, "termination", None, {"terminal_reason": terminal_reason, "success": False})
                break
        else:
            _event(writer, "termination", None, {"terminal_reason": terminal_reason, "success": False})

        annotations = {
            "target": {"hotel": config["target_name"], "expected_destination": config["expected_destination"]},
            "task_success": success,
            "terminal_reason": terminal_reason,
            "target_frame_interval": [min(target_interval), max(target_interval)] if target_interval else None,
            "target_visible_region": "hotel result card" if target_interval else None,
            "target_seen_before_left_viewport": target_seen_before_scroll,
            "second_find_turn_step": second_turn_step,
            "completed_steps": completed_steps,
            "initial_state": initial_state,
            "final_state": fixture.state(),
            "error": invalid_error,
            "manual_privacy_review": False,
            "instrumented_trace_ids": activation_trace_ids,
        }
        bundle = writer.finalize(annotations)
        report = render_report(bundle)
        bundle = writer.finalize(annotations, extra_files={"report.html": report})
        validation = validate_bundle(bundle)
        replay = replay_bundle(bundle)
        return bundle, {"validation": validation, "replay": replay, "annotations": annotations}


def _event(
    writer: BundleWriter,
    event_type: str,
    step: int | None,
    data: dict[str, Any],
    artifacts: list[dict[str, Any]] | None = None,
    *,
    replay_ready: bool | None = None,
    interpretability: dict[str, Any] | None = None,
) -> None:
    raw_id = writer.add_raw_event(
        {
            "source": "outer-local-harness",
            "captured_at": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "step": step,
            "payload": data,
        }
    )
    writer.add_event(
        event_type,
        step=step,
        data=data,
        artifacts=artifacts,
        raw_event_refs=[raw_id],
        interpretability_replay_ready=replay_ready,
        interpretability=interpretability,
    )


def _target_visible(config: dict, state: dict) -> bool:
    target = next(hotel for hotel in config["hotels"] if hotel["id"] == config["target_id"])
    top = 250 + target["index"] * 262 - state["scroll_y"]
    return top < config["viewport"]["height"] and top + 244 > 78


def replay_bundle(bundle: Path) -> dict[str, Any]:
    manifest = json.loads((bundle / "manifest.json").read_text())
    annotations = json.loads((bundle / "annotations.json").read_text())
    seed = manifest["fixture"]["seed"]
    variant = manifest["fixture"].get("variant", 0)
    with running_fixture(seed=seed, variant=variant) as fixture:
        reset_state = fixture.reset(seed, variant)
        for action_path in sorted((bundle / "actions").glob("*.json")):
            action = json.loads(action_path.read_text())["normalized"]
            fixture.apply(action)
        final_state = fixture.state()
    expected = annotations["final_state"]
    reproducible = reset_state == annotations["initial_state"] and final_state == expected
    if not reproducible:
        raise ValueError(f"replay diverged: expected {expected}, got {final_state}")
    return {"reproducible": True, "actions": len(list((bundle / "actions").glob("*.json"))), "final_state": final_state}
