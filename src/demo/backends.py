"""Scripted and OpenAI-compatible action decision backends."""

from __future__ import annotations

import base64
import copy
import json
from io import BytesIO
from typing import Any, Protocol

import httpx
from PIL import Image

from capture.schema import ACTION_SCHEMA, validate_action

from .renderer import hotel_click, hotel_scroll, target_click, target_scroll

SYSTEM_PROMPT = """You control a hotel-search interface. Do not leave the current site or attempt a booking. Return exactly one JSON object matching the supplied action schema, with no markdown or hidden reasoning."""
CHEAPEST_ONLY_SYSTEM_INSTRUCTION = (
    "CRITICAL CHEAPEST-HOTEL CONSTRAINT: VIEW DETAILS OF ONLY THE CHEAPEST HOTEL. "
    "Inspect all results before deciding. Never open View details for any other hotel, even temporarily. "
    "After every result has appeared, compare the observed prices. If the cheapest hotel's card is not visible, "
    "scroll back to it; click its View details button only when it is visible. Do not narrate. Emit only the desktop "
    "tool call. "
    "In the native desktop tool, use a negative delta_y to scroll down and a positive delta_y to scroll up."
)

MODEL_ACTION_SCHEMA = copy.deepcopy(ACTION_SCHEMA)
MODEL_ACTION_SCHEMA["oneOf"][0]["properties"]["x"]["maximum"] = 1000
MODEL_ACTION_SCHEMA["oneOf"][0]["properties"]["y"]["maximum"] = 1000
MODEL_ACTION_SCHEMA["oneOf"][1]["properties"]["delta_y"]["description"] = (
    "Native wheel delta: negative scrolls down toward later results; positive scrolls up toward earlier results."
)


class BackendDecisionError(ValueError):
    """A completion arrived, but it could not be reduced to one valid action."""

    def __init__(self, message: str, *, request: dict[str, Any], response: dict[str, Any]) -> None:
        super().__init__(message)
        self.request = request
        self.response = response


class ActionBackend(Protocol):
    name: str
    model_id: str
    model_revision: str | None
    processor_revision: str | None

    def decide(self, *, step: int, messages: list[dict[str, Any]], image: Image.Image, config: dict, state: dict) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: ...


def png_data_url(image: Image.Image) -> tuple[str, bytes]:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    data = buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(data).decode(), data


def project_model_action(action: dict[str, Any], image_size: tuple[int, int]) -> dict[str, Any]:
    """Project Holo's normalized pointer and wheel conventions into fixture coordinates."""

    if action.get("action") == "scroll":
        return {"action": "scroll", "delta_y": -int(action["delta_y"])}
    if action.get("action") != "click":
        return dict(action)
    width, height = image_size
    return {
        "action": "click",
        "x": round(int(action["x"]) * (width - 1) / 1000),
        "y": round(int(action["y"]) * (height - 1) / 1000),
    }


def is_cheapest_task(messages: list[dict[str, Any]]) -> bool:
    """Recognize the bounded cheapest-hotel objective from its user instruction."""

    return any(
        message.get("role") == "user" and "lowest nightly price" in str(message.get("content", "")).lower()
        for message in messages
    )


def build_request(
    messages: list[dict[str, Any]],
    image: Image.Image,
    model_id: str,
    *,
    normalized_coordinates: bool = False,
    frame_index: int | None = None,
) -> dict[str, Any]:
    image_url, _ = png_data_url(image)
    system_prompt = SYSTEM_PROMPT
    if is_cheapest_task(messages):
        system_prompt = f"{system_prompt}\n\n{CHEAPEST_ONLY_SYSTEM_INSTRUCTION}"
    request_messages = [{"role": "system", "content": system_prompt}, *messages]
    request_messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Exact fixture screenshot for frame {frame_index}; this is the current frame. Choose one action."
                        if frame_index is not None
                        else "Current exact fixture screenshot. Choose one action."
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    )
    return {
        "model": model_id,
        "messages": request_messages,
        "temperature": 0,
        "max_tokens": 256,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "desktop_action",
                    "parameters": MODEL_ACTION_SCHEMA if normalized_coordinates else ACTION_SCHEMA,
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "desktop_action"}},
    }


class ScriptedBackend:
    name = "scripted"
    model_id = "deterministic-reference-policy"
    model_revision = "v1"
    processor_revision = "fixture-renderer-v1"

    def __init__(self) -> None:
        self._seen_bottom = False

    def decide(self, *, step: int, messages: list[dict[str, Any]], image: Image.Image, config: dict, state: dict) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        request = build_request(messages, image, self.model_id, frame_index=step)
        cheapest_task = is_cheapest_task(messages)
        if cheapest_task:
            cheapest = min(config["hotels"], key=lambda hotel: (hotel["price"], hotel["id"]))
            if config.get("generator_version"):
                max_scroll = int(config["max_scroll"])
                if state["scroll_y"] >= max_scroll:
                    self._seen_bottom = True
                x, y = hotel_click(config, state, cheapest["id"])
                button_visible = config["layout"]["header_height"] < y < config["viewport"]["height"]
                if self._seen_bottom and button_visible:
                    action = {"action": "click", "x": x, "y": y}
                elif not self._seen_bottom:
                    action = {"action": "scroll", "delta_y": min(500, max_scroll - state["scroll_y"])}
                else:
                    target = hotel_scroll(config, cheapest["id"])
                    delta = max(-500, min(500, target - state["scroll_y"]))
                    action = {"action": "scroll", "delta_y": delta}
            elif step == 0:
                action = {"action": "scroll", "delta_y": hotel_scroll(config, cheapest["id"])}
            else:
                x, y = hotel_click(config, state, cheapest["id"])
                action = {"action": "click", "x": x, "y": y}
        elif step == 0:
            action = {"action": "scroll", "delta_y": config["initial_scroll_delta"]}
        elif step == 1:
            action = {"action": "scroll", "delta_y": target_scroll(config) - state["scroll_y"]}
        else:
            x, y = target_click(config, state)
            action = {"action": "click", "x": x, "y": y}
        response = {
            "id": f"scripted-{step:04d}",
            "model": self.model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(action)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        }
        return request, response, validate_action(action)


class OpenAIBackend:
    name = "local"

    def __init__(
        self,
        base_url: str,
        model_id: str,
        model_revision: str | None,
        processor_revision: str | None,
        trace_generation_steps: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.model_revision = model_revision
        self.processor_revision = processor_revision
        self.trace_generation_steps = trace_generation_steps
        # Native Metal generation plus compression of square rollout matrices
        # can exceed fifteen minutes on the first request after model load.
        self._http = httpx.Client(timeout=3600, trust_env=False)
        try:
            models = self._http.get(f"{self.base_url}/models").json().get("data", [])
            selected = next((item for item in models if item.get("id") == model_id), None)
            if selected and selected.get("revision"):
                self.model_revision = selected["revision"]
                self.processor_revision = selected["revision"]
        except Exception:
            pass  # The actual completion request will provide the authoritative connectivity error.

    def decide(self, *, step: int, messages: list[dict[str, Any]], image: Image.Image, config: dict, state: dict) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        request = build_request(
            messages,
            image,
            self.model_id,
            normalized_coordinates=True,
            frame_index=step,
        )
        request["chat_template_kwargs"] = {"enable_thinking": False}
        request["trace"] = {
            "capture_attentions": True,
            # Saliency uses attention rows, value norms, and prompt rollout.
            # Retaining every layer's hidden state for every generated token
            # adds substantial Metal memory pressure without affecting maps.
            "capture_hidden_states": False,
            "capture_kv": False,
            "capture_value_norms": True,
            "capture_rollout": True,
            "max_generation_steps": self.trace_generation_steps,
        }
        # The checkpoint may emit a short natural-language preamble before its
        # native tool call. A real scroll completion reached 64 tokens before
        # closing the final parameter tags, so 128 is the tested safe floor.
        # A larger requested trace window raises the generation budget with it.
        request["max_tokens"] = max(128, self.trace_generation_steps)
        response = self._http.post(f"{self.base_url}/chat/completions", json=request)
        response.raise_for_status()
        payload = response.json()
        try:
            message = payload["choices"][0]["message"]
            if message.get("tool_calls"):
                arguments = message["tool_calls"][0]["function"]["arguments"]
                action = json.loads(arguments) if isinstance(arguments, str) else arguments
            else:
                content = message.get("content", "")
                content = content.strip().removeprefix("```json").removesuffix("```").strip()
                action = json.loads(content)
            action = project_model_action(action, image.size)
            validated = validate_action(action)
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendDecisionError(
                f"completion did not contain exactly one valid action: {exc}",
                request=request,
                response=payload,
            ) from exc
        return request, payload, validated

    def close(self) -> None:
        self._http.close()
