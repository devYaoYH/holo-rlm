"""Best-effort Holo Agent API JSONL adapter; raw fields always remain untouched."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected an object")
                yield value


def classify_holo_event(raw: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Project documented public event kinds without assuming private runtime fields."""

    envelope_type = raw.get("type")
    if isinstance(raw.get("data"), dict):
        data = raw["data"]
    elif isinstance(raw.get("event"), dict):
        # hai-agent-runtime v0.1.10 events.jsonl envelope observed in the
        # installed binary's per-run artifact.
        data = raw["event"]
    else:
        data = raw
    kind = data.get("kind")
    if kind in {"message_event", "user_message_event"}:
        return "user_turn", {"message": data.get("message") or data.get("content"), "source": "holo-runtime"}
    if kind == "observation_event":
        observation = data.get("observation") if isinstance(data.get("observation"), dict) else {}
        return "observation", {
            "source": "holo-runtime-observation",
            "observation_kind": observation.get("kind"),
            "viewport_size": observation.get("viewport_size"),
            "image_present": isinstance(observation.get("image"), dict),
            "exact_model_input_not_claimed": True,
        }
    if kind == "policy_event":
        return "model_response", {
            "content": data.get("content") or (data.get("message") or {}).get("content"),
            "tool_requests": data.get("tool_reqs", []),
            "reasoning_omitted": True,
        }
    if kind == "tool_result":
        return "action_result", {"tool_request": data.get("tool_req"), "result": data.get("result")}
    if kind == "answer_event":
        return "termination", {"terminal_reason": "runtime_answer", "answer": data.get("answer")}
    if kind == "error_event" or envelope_type == "AgentErrorEvent":
        return "termination", {"terminal_reason": "runtime_error", "error": data.get("error")}
    return None


def omit_hidden_reasoning(raw: dict[str, Any]) -> dict[str, Any]:
    """Copy a runtime event while removing any backend-provided hidden reasoning."""

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key in {"reasoning", "reasoning_content", "hidden_reasoning"}:
                    result[f"{key}_omitted"] = True
                else:
                    result[key] = clean(item)
            return result
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(raw)


def runtime_manifest_note(cli_version: str | None) -> dict[str, Any]:
    return {
        "adapter": "public-agent-api-events-v0",
        "observed_event_kinds": [
            "message_event",
            "observation_event",
            "policy_event",
            "tool_result",
            "answer_event",
            "error_event",
        ],
        "cli_runtime_version": cli_version,
        "spike_date": datetime.now(UTC).date().isoformat(),
        "limitation": "The public stream does not expose exact model request image bytes; correlate endpoint-side captures for replay.",
    }
