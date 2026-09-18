import base64
import hashlib
import json
from pathlib import Path

from PIL import Image

from capture.normalizer import classify_holo_event, omit_hidden_reasoning
from capture.validator import validate_bundle


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
