import base64
import hashlib
import json
from pathlib import Path

import httpx
from PIL import Image

from capture.normalizer import classify_holo_event, omit_hidden_reasoning
from capture.validator import validate_bundle
from demo.backends import BackendDecisionError, ScriptedBackend, build_request
from demo.runner import _action_for_model_history, _safe_exception, capture_run


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_bundle_has_exact_input_linkage_and_all_event_types(valid_bundle: Path) -> None:
    result = validate_bundle(valid_bundle)
    assert result["replay_ready"] is True
    manifest = json.loads((valid_bundle / "manifest.json").read_text())
    events = _lines(valid_bundle / "events.jsonl")
    assert {event["event_type"] for event in events} == {
        "user_turn",
        "observation",
        "model_request",
        "model_response",
        "proposed_action",
        "action_result",
        "termination",
    }
    assert manifest["backend"] == "scripted"
    assert manifest["research_eligible"] is False
    request = json.loads((valid_bundle / "requests/0000.json").read_text())
    data_url = request["messages"][-1]["content"][-1]["image_url"]["url"]
    encoded = data_url.split(",", 1)[1]
    request_png = base64.b64decode(encoded)
    frame = (valid_bundle / "frames/0000-model-input.png").read_bytes()
    assert request_png == frame
    assert hashlib.sha256(frame).hexdigest() == manifest["files"]["frames/0000-model-input.png"]["sha256"]


def test_report_and_observable_annotations(valid_bundle: Path) -> None:
    report = (valid_bundle / "report.html").read_text()
    annotations = json.loads((valid_bundle / "annotations.json").read_text())
    assert "Step 0" in report
    assert "frames/0000-model-input.png" in report
    assert annotations["target_seen_before_left_viewport"] is True
    assert annotations["second_find_turn_step"] == 1
    assert "attention" not in json.dumps(annotations).lower()


def test_public_holo_event_projection_is_bounded() -> None:
    projected = classify_holo_event(
        {
            "type": "AgentEvent",
            "data": {
                "kind": "policy_event",
                "content": '{"note":"click"}',
                "reasoning_content": "private chain",
                "tool_reqs": [{"tool_name": "click", "args": {"x": 1, "y": 2}}],
            },
        }
    )
    assert projected is not None
    kind, data = projected
    assert kind == "model_response"
    assert data["reasoning_omitted"] is True
    assert "private chain" not in json.dumps(data)


def test_installed_runtime_envelope_and_reasoning_omission() -> None:
    raw = {
        "id": "event-1",
        "event": {
            "kind": "policy_event",
            "message": {"content": "usable action", "reasoning_content": "private chain"},
            "reasoning_content": "another private chain",
            "tool_reqs": [],
        },
    }
    projected = classify_holo_event(raw)
    assert projected == (
        "model_response",
        {"content": "usable action", "tool_requests": [], "reasoning_omitted": True},
    )
    safe = omit_hidden_reasoning(raw)
    assert "private chain" not in json.dumps(safe)
    assert safe["event"]["reasoning_content_omitted"] is True


def test_model_frames_have_manifest_dimensions(valid_bundle: Path) -> None:
    with Image.open(valid_bundle / "frames/0001-model-input.png") as image:
        assert image.size == (1280, 800)


def test_cheapest_task_targets_lowest_price(tmp_path: Path) -> None:
    bundle, result = capture_run(
        backend=ScriptedBackend(),
        output_root=tmp_path,
        seed=0,
        max_steps=3,
        task="cheapest",
    )
    annotations = json.loads((bundle / "annotations.json").read_text())
    assert result["annotations"]["task_success"] is True
    assert annotations["target"] == {
        "hotel": "Signal Quay Rooms",
        "hotel_id": "signal-quay",
        "price": 172,
        "expected_destination": "/details/signal-quay",
    }
    second_request = json.loads((bundle / "requests/0001.json").read_text())
    assert "frame 0" in second_request["messages"][2]["content"][0]["text"]
    assert "<function=desktop_action>" in second_request["messages"][3]["content"]
    assert "moderate wheel increments near 500" in second_request["messages"][4]["content"]
    assert "reverse with positive delta_y" in second_request["messages"][4]["content"]
    assert "frame 1" in second_request["messages"][5]["content"][0]["text"]
    historical_images = [
        part["image_url"]["url"]
        for message in second_request["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    ]
    assert len(historical_images) == 2
    assert base64.b64decode(historical_images[0].split(",", 1)[1]) == (
        bundle / "frames/0000-model-input.png"
    ).read_bytes()
    assert base64.b64decode(historical_images[1].split(",", 1)[1]) == (
        bundle / "frames/0001-model-input.png"
    ).read_bytes()


def test_invalid_completion_is_retained_without_an_orphan_step(tmp_path: Path) -> None:
    class InvalidBackend:
        name = "local"
        model_id = "invalid-test-model"
        model_revision = "test"
        processor_revision = "test"

        def decide(self, *, messages, image, **_kwargs):
            request = build_request(messages, image, self.model_id, normalized_coordinates=True)
            response = {"instrumented_trace_id": "trace-invalid", "choices": [{"message": {"content": "bad"}}]}
            raise BackendDecisionError("invalid output", request=request, response=response)

    bundle, result = capture_run(
        backend=InvalidBackend(),
        output_root=tmp_path,
        seed=0,
        max_steps=1,
        task="cheapest",
    )
    events = _lines(bundle / "events.jsonl")
    assert [event["event_type"] for event in events] == ["user_turn", "termination"]
    assert result["validation"]["replay_ready"] is False
    assert result["annotations"]["instrumented_trace_ids"] == ["trace-invalid"]


def test_stop_on_click_finalizes_after_first_issued_click(tmp_path: Path) -> None:
    class ImmediateClickBackend:
        name = "scripted"
        model_id = "click-test-model"
        model_revision = "test"
        processor_revision = "test"

        def decide(self, *, messages, image, **_kwargs):
            request = build_request(messages, image, self.model_id)
            response = {"choices": [{"message": {"content": '{"action":"click","x":1,"y":1}'}}]}
            return request, response, {"action": "click", "x": 1, "y": 1}

    bundle, result = capture_run(
        backend=ImmediateClickBackend(),
        output_root=tmp_path,
        seed=0,
        max_steps=4,
        task="cheapest",
        stop_on_click=True,
    )
    annotations = json.loads((bundle / "annotations.json").read_text())
    assert annotations["completed_steps"] == 1
    assert annotations["captured_click"] is True
    assert annotations["terminal_reason"] == "click_issued"
    assert result["replay"]["actions"] == 1
    assert len(list((bundle / "actions").glob("*.json"))) == 1


def test_http_failure_diagnostic_omits_remote_library_help_url() -> None:
    request = httpx.Request("POST", "http://127.0.0.1:8001/v1/chat/completions")
    response = httpx.Response(500, request=request)
    error = httpx.HTTPStatusError(
        "500 error; see https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500",
        request=request,
        response=response,
    )
    diagnostic = _safe_exception(error)
    assert diagnostic == "HTTPStatusError: 500 response from local inference endpoint"
    assert "https://" not in diagnostic


def test_repeated_downward_scroll_still_demands_visible_cheapest_click(tmp_path: Path) -> None:
    class RepeatedScrollBackend:
        name = "scripted"
        model_id = "scroll-test-model"
        model_revision = "test"
        processor_revision = "test"

        def decide(self, *, messages, image, **_kwargs):
            request = build_request(messages, image, self.model_id)
            response = {"choices": [{"message": {"content": '{"action":"scroll","delta_y":800}'}}]}
            return request, response, {"action": "scroll", "delta_y": 800}

    bundle, _result = capture_run(
        backend=RepeatedScrollBackend(),
        output_root=tmp_path,
        seed=0,
        max_steps=5,
        task="cheapest",
    )
    fifth_request = json.loads((bundle / "requests/0004.json").read_text())
    prompt = json.dumps(fifth_request["messages"])
    assert "cheapest hotel's card is visible now" in prompt
    assert "Do not scroll" in prompt
    assert "Immediately click View details" in prompt


def test_model_history_preserves_native_scroll_sign_before_fixture_projection() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "desktop_action", "arguments": '{"action":"scroll","delta_y":-500}'}}
                    ]
                }
            }
        ]
    }
    assert _action_for_model_history(response, {"action": "scroll", "delta_y": 500}) == {
        "action": "scroll",
        "delta_y": -500,
    }


def test_top_noop_prompts_downward_recovery(tmp_path: Path) -> None:
    class RepeatedUpBackend:
        name = "scripted"
        model_id = "scroll-test-model"
        model_revision = "test"
        processor_revision = "test"

        def decide(self, *, messages, image, **_kwargs):
            request = build_request(messages, image, self.model_id)
            response = {"choices": [{"message": {"content": '{"action":"scroll","delta_y":-500}'}}]}
            return request, response, {"action": "scroll", "delta_y": -500}

    bundle, _result = capture_run(
        backend=RepeatedUpBackend(),
        output_root=tmp_path,
        seed=0,
        max_steps=2,
        task="cheapest",
    )
    second_request = json.loads((bundle / "requests/0001.json").read_text())
    prompt = json.dumps(second_request["messages"])
    assert "reached the top" in prompt
    assert "negative delta_y near -500" in prompt
    assert "Do not scroll up again" in prompt
