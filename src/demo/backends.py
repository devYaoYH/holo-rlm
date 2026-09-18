"""Scripted and OpenAI-compatible action decision backends."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any, Protocol

import httpx
from PIL import Image

from capture.schema import ACTION_SCHEMA, validate_action

from .renderer import target_click, target_scroll

SYSTEM_PROMPT = """You control only a deterministic localhost booking fixture. Never navigate to an external URL or attempt a booking. Return exactly one JSON object matching the supplied action schema, with no markdown or hidden reasoning."""


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


def build_request(messages: list[dict[str, Any]], image: Image.Image, model_id: str) -> dict[str, Any]:
    image_url, _ = png_data_url(image)
    request_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
    request_messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Current exact fixture screenshot. Choose one action."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    )
    return {
        "model": model_id,
        "messages": request_messages,
        "temperature": 0,
        "max_tokens": 128,
        "tools": [{"type": "function", "function": {"name": "desktop_action", "parameters": ACTION_SCHEMA}}],
        "tool_choice": {"type": "function", "function": {"name": "desktop_action"}},
    }


class ScriptedBackend:
    name = "scripted"
    model_id = "deterministic-reference-policy"
    model_revision = "v1"
    processor_revision = "fixture-renderer-v1"

    def decide(self, *, step: int, messages: list[dict[str, Any]], image: Image.Image, config: dict, state: dict) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        request = build_request(messages, image, self.model_id)
        if step == 0:
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

    def __init__(self, base_url: str, model_id: str, model_revision: str | None, processor_revision: str | None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.model_revision = model_revision
        self.processor_revision = processor_revision
        self._http = httpx.Client(timeout=300, trust_env=False)
        try:
            models = self._http.get(f"{self.base_url}/models").json().get("data", [])
            selected = next((item for item in models if item.get("id") == model_id), None)
            if selected and selected.get("revision"):
                self.model_revision = selected["revision"]
                self.processor_revision = selected["revision"]
        except Exception:
            pass  # The actual completion request will provide the authoritative connectivity error.

    def decide(self, *, step: int, messages: list[dict[str, Any]], image: Image.Image, config: dict, state: dict) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        request = build_request(messages, image, self.model_id)
        request["chat_template_kwargs"] = {"enable_thinking": False}
        request["trace"] = {
            "capture_attentions": True,
            "capture_hidden_states": True,
            "capture_kv": False,
            "max_generation_steps": 4,
        }
        response = self._http.post(f"{self.base_url}/chat/completions", json=request)
        response.raise_for_status()
        payload = response.json()
        message = payload["choices"][0]["message"]
        if message.get("tool_calls"):
            arguments = message["tool_calls"][0]["function"]["arguments"]
            action = json.loads(arguments) if isinstance(arguments, str) else arguments
        else:
            content = message.get("content", "")
            content = content.strip().removeprefix("```json").removesuffix("```").strip()
            action = json.loads(content)
        return request, payload, validate_action(action)

    def close(self) -> None:
        self._http.close()
