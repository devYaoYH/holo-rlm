"""Bounded capture loop and deterministic action replay."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import ImageDraw

from capture.bundle import BundleWriter, canonical_json, sha256_bytes
from capture.report import render_report
from capture.validator import validate_bundle

from .backends import ActionBackend, BackendDecisionError
from .fixture import running_fixture
from .renderer import render_fixture

INITIAL_TASK = "Inspect the deterministic booking results, note the options, and scroll down. Do not open a details link yet."
CHEAPEST_TASK = (
    "Find the hotel with the lowest nightly price across all deterministic search results. "
    "You have not seen all results yet: your first action must scroll down, and you must not click any currently "
    "visible hotel. Inspect every option, then open the cheapest hotel's local View details link. "
    "Take exactly one desktop action per turn."
)


def _native_action_history(action: dict[str, Any]) -> str:
    """Represent an applied action in the checkpoint's native tool-call syntax."""

    parameters = "\n".join(
        f"<parameter={key}>\n{value}\n</parameter>"
        for key, value in action.items()
        if key != "action"
    )
    return (
        "<tool_call>\n<function=desktop_action>\n"
        f"<parameter=action>\n{action['action']}\n</parameter>\n"
        f"{parameters}\n</function>\n</tool_call>"
    )


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
    task: str = "target-after-scroll",
    stop_on_click: bool = False,
    redactions: tuple[tuple[int, int, int, int], ...] = (),
) -> tuple[Path, dict[str, Any]]:
    with running_fixture(seed=seed, variant=variant) as fixture:
        config = fixture.config()
        initial_state = fixture.reset(seed, variant)
        if task == "target-after-scroll":
            objective = next(hotel for hotel in config["hotels"] if hotel["id"] == config["target_id"])
            initial_task = INITIAL_TASK
            follow_up_task: str | None = (
                f"Now locate {config['target_name']} again and open its local View details link."
            )
            task_id = f"find-target-after-scroll-v1-variant-{config['variant']}"
        elif task == "cheapest":
            objective = min(config["hotels"], key=lambda hotel: (hotel["price"], hotel["id"]))
            initial_task = CHEAPEST_TASK
            follow_up_task = None
            task_id = "find-cheapest-hotel-v1"
        else:
            raise ValueError(f"unsupported capture task: {task}")
        expected_destination = f"/details/{objective['id']}"
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
            task_id=task_id,
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
        messages: list[dict[str, Any]] = [{"role": "user", "content": initial_task}]
        _event(writer, "user_turn", None, {"message": initial_task, "turn": 0})
        success = False
        terminal_reason = "bounded_step_limit"
        target_seen_before_scroll = False
        target_interval: list[int] = []
        second_turn_step = 1 if follow_up_task else None
        invalid_error: str | None = None
        completed_steps = 0
        activation_trace_ids: list[str] = []
        seen_hotel_ids: set[str] = set()
        model_input_history: list[dict[str, Any]] = []
        captured_click = False

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
            request_image_refs = [*model_input_history, image_ref]
            target_visible = _target_visible(config, state, objective["id"])
            seen_hotel_ids.update(_visible_hotel_ids(config, state))
            if target_visible:
                target_interval.append(step)
            if step == 0 and target_visible:
                target_seen_before_scroll = True
            try:
                request, response, action = backend.decide(
                    step=step, messages=messages, image=image, config=config, state=state
                )
            except Exception as exc:
                invalid_error = f"{type(exc).__name__}: {exc}"
                terminal_reason = "backend_or_action_error"
                failure_artifacts = [image_ref]
                if isinstance(exc, BackendDecisionError):
                    request_relative = f"requests/{step:04d}.json"
                    response_relative = f"responses/{step:04d}.json"
                    writer.write_json(request_relative, exc.request)
                    writer.write_json(response_relative, exc.response)
                    failure_artifacts.extend(
                        [
                            _artifact(writer.path / request_relative, writer.path, "failed_model_request"),
                            _artifact(writer.path / response_relative, writer.path, "failed_model_response"),
                        ]
                    )
                    trace_id = exc.response.get("instrumented_trace_id")
                    if isinstance(trace_id, str):
                        activation_trace_ids.append(trace_id)
                _event(
                    writer,
                    "termination",
                    None,
                    {"terminal_reason": terminal_reason, "error": invalid_error, "failed_step": step},
                    failure_artifacts,
                )
                break

            _event(
                writer,
                "observation",
                step,
                {"state": state, "target_visible": target_visible, "source": "exact-renderer-model-input"},
                [image_ref],
                replay_ready=True,
            )

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
                    "image_refs": [
                        {"path": ref["path"], "sha256": ref["sha256"]}
                        for ref in request_image_refs
                    ],
                    "tool_schema_hash": hashlib.sha256(
                        canonical_json(request["tools"][0]["function"]["parameters"])
                    ).hexdigest(),
                    "sampling": {"temperature": request.get("temperature"), "max_tokens": request.get("max_tokens")},
                    "generation": {"tool_choice": request.get("tool_choice")},
                    "coordinate_space": "normalized_0_1000" if backend.name == "local" else "viewport_pixels",
                    "model_id": backend.model_id,
                    "model_revision": backend.model_revision,
                    "processor_revision": backend.processor_revision,
                    "processor_configuration": {"renderer": "fixture-renderer-v1", "color_mode": "RGB"},
                    "conversation_id": writer.trajectory_id,
                    "session_id": None,
                    "reasoning_not_retained": True,
                },
                [request_ref, *request_image_refs],
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
            seen_hotel_ids.update(_visible_hotel_ids(config, after))
            post_image = render_fixture(config, after)
            post_relative = f"frames/{step:04d}-post-action.png"
            post_meta = writer.write_image(post_relative, post_image)
            post_ref = {**post_meta, "role": "post_action"}
            action_success = action["action"] == "finish" or before != after
            fixture_assertion = bool(after.get("details_open") and after.get("destination") == expected_destination)
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
            captured_click = captured_click or action["action"] == "click"
            # Preserve the exact multimodal observation in conversation order.
            # The next request can therefore attend to every prior screenshot,
            # while the final appended screenshot remains the current frame.
            messages.append(deepcopy(request["messages"][-1]))
            model_input_history.append(image_ref)
            messages.append({"role": "assistant", "content": _native_action_history(action)})
            if fixture_assertion:
                success = True
                terminal_reason = "fixture_success"
                _event(writer, "termination", None, {"terminal_reason": terminal_reason, "success": True})
                break
            if stop_on_click and action["action"] == "click":
                terminal_reason = "click_issued"
                _event(writer, "termination", None, {"terminal_reason": terminal_reason, "success": False})
                break
            if task == "cheapest":
                all_results_seen = len(seen_hotel_ids) == len(config["hotels"])
                progress = (
                    "Every hotel result has now appeared across the frames you inspected. Do not scroll again. "
                    "Compare the observed nightly prices and immediately click View details for only the cheapest hotel."
                    if all_results_seen
                    else (
                        "Continue the same task. If results remain below, use a negative delta_y to inspect them and "
                        "do not reverse direction prematurely. Once every result has been inspected, open View details "
                        "for only the cheapest hotel."
                    )
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The previous action was applied; the current scroll offset is {after['scroll_y']}. "
                            f"{progress}"
                        ),
                    }
                )
            if step == 0 and follow_up_task is not None:
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
            "task": task,
            "target": {
                "hotel": objective["name"],
                "hotel_id": objective["id"],
                "price": objective["price"],
                "expected_destination": expected_destination,
            },
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
            "replay_ready": completed_steps > 0,
            "instrumented_trace_ids": activation_trace_ids,
            "captured_click": captured_click,
        }
        bundle = writer.finalize(annotations)
        report = render_report(bundle)
        bundle = writer.finalize(annotations, extra_files={"report.html": report})
        validation = validate_bundle(bundle, require_replay_ready=completed_steps > 0)
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


def _target_visible(config: dict, state: dict, hotel_id: str | None = None) -> bool:
    target_id = hotel_id or config["target_id"]
    target = next(hotel for hotel in config["hotels"] if hotel["id"] == target_id)
    top = 250 + target["index"] * 262 - state["scroll_y"]
    return top < config["viewport"]["height"] and top + 244 > 78


def _visible_hotel_ids(config: dict, state: dict) -> set[str]:
    viewport_height = int(config["viewport"]["height"])
    result = set()
    for hotel in config["hotels"]:
        price_y = 250 + hotel["index"] * 262 - state["scroll_y"] + 105
        if 78 < price_y < viewport_height:
            result.add(hotel["id"])
    return result


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
