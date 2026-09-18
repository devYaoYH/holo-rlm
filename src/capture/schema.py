"""Stable v0 constants and the constrained desktop action schema."""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "holo-trajectory-v0"
EVENT_TYPES = {
    "user_turn",
    "observation",
    "model_request",
    "model_response",
    "proposed_action",
    "action_result",
    "termination",
}

ACTION_SCHEMA: dict[str, Any] = {
    "title": "DesktopAction",
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "x", "y"],
            "properties": {
                "action": {"const": "click"},
                "x": {"type": "integer", "minimum": 0, "maximum": 1279},
                "y": {"type": "integer", "minimum": 0, "maximum": 799},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "delta_y"],
            "properties": {
                "action": {"const": "scroll"},
                "delta_y": {"type": "integer", "minimum": -1600, "maximum": 1600},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "milliseconds"],
            "properties": {
                "action": {"const": "wait"},
                "milliseconds": {"type": "integer", "minimum": 0, "maximum": 3000},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "summary"],
            "properties": {
                "action": {"const": "finish"},
                "summary": {"type": "string", "maxLength": 500},
            },
        },
    ],
}

RESERVED_FIELDS = {
    "safety": {
        "risk_class": None,
        "side_effect_boundary": None,
        "confirmation_stage": None,
        "recoverability": None,
    },
    "interpretability": {
        "input_token_ids_ref": None,
        "image_grid_ref": None,
        "replay_ready": None,
        "activation_trace_id": None,
    },
    "rlm": {"external_object_ref": None, "retrieval_calls": []},
}


def validate_action(action: object) -> dict[str, Any]:
    """Validate the small action union without pulling in a JSON Schema runtime."""

    if not isinstance(action, dict):
        raise ValueError("action must be a JSON object")
    name = action.get("action")
    expected: dict[str, type]
    if name == "click":
        expected = {"action": str, "x": int, "y": int}
        ranges = {"x": (0, 1279), "y": (0, 799)}
    elif name == "scroll":
        expected = {"action": str, "delta_y": int}
        ranges = {"delta_y": (-1600, 1600)}
    elif name == "wait":
        expected = {"action": str, "milliseconds": int}
        ranges = {"milliseconds": (0, 3000)}
    elif name == "finish":
        expected = {"action": str, "summary": str}
        ranges = {}
    else:
        raise ValueError(f"unsupported action: {name!r}")
    if set(action) != set(expected):
        raise ValueError(f"{name} action fields must be exactly {sorted(expected)}")
    for key, expected_type in expected.items():
        if type(action[key]) is not expected_type:  # bool is deliberately not an int here.
            raise ValueError(f"{key} must be {expected_type.__name__}")
    for key, (lower, upper) in ranges.items():
        if not lower <= action[key] <= upper:
            raise ValueError(f"{key} is outside [{lower}, {upper}]")
    if name == "finish" and len(action["summary"]) > 500:
        raise ValueError("finish summary is too long")
    return dict(action)
