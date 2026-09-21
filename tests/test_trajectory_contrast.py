from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from demo.trajectory_contrast import MULTIFRAME_SYSTEM_PROMPT, build_multiframe_prompt_request


def _source_request() -> dict:
    return {
        "model": "fixture-model",
        "messages": [
            {"role": "system", "content": "old system"},
            {"role": "user", "content": "Find the cheapest hotel."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Frame 0"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                ],
            },
            {"role": "assistant", "content": "<tool_call>scroll</tool_call>"},
            {"role": "user", "content": "The previous action was applied at offset 500."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Frame 1"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBB"}},
                ],
            },
            {"role": "user", "content": "Every price has appeared. Click the cheapest."},
        ],
        "max_tokens": 8,
    }


def test_multiframe_probe_changes_only_instruction_semantics() -> None:
    source = _source_request()
    target = build_multiframe_prompt_request(source, "Click the cheapest hotel.", trace_generation_steps=72)
    control = build_multiframe_prompt_request(source, "Click the logo.", trace_generation_steps=72)

    assert source["messages"][0]["content"] == "old system"
    assert target["messages"][0]["content"] == MULTIFRAME_SYSTEM_PROMPT
    assert target["messages"][1]["content"] == "Instruction: Click the cheapest hotel."
    assert control["messages"][1]["content"] == "Instruction: Click the logo."

    target_images = [
        part["image_url"]["url"]
        for message in target["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part["type"] == "image_url"
    ]
    control_images = [
        part["image_url"]["url"]
        for message in control["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part["type"] == "image_url"
    ]
    assert target_images == control_images == ["data:image/png;base64,AAA", "data:image/png;base64,BBB"]
    assert [m for m in target["messages"] if m["role"] == "assistant"] == [
        m for m in control["messages"] if m["role"] == "assistant"
    ]
    assert target["messages"][4]["content"] == control["messages"][4]["content"]
    assert target["messages"][6]["content"] == control["messages"][6]["content"]
    assert target["trace"]["capture_value_norms"] is True
    assert target["trace"]["capture_rollout"] is True
    assert target["trace"]["max_generation_steps"] == 72
    assert target["tools"][0]["function"]["parameters"]["required"] == ["action", "x", "y"]


def test_multiframe_probe_uniformly_downsamples_embedded_frames() -> None:
    source = _source_request()
    buffer = BytesIO()
    Image.new("RGB", (128, 80), "white").save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    for message in source["messages"]:
        if isinstance(message.get("content"), list):
            message["content"][1]["image_url"]["url"] = url

    request = build_multiframe_prompt_request(
        source,
        "Click the cheapest hotel.",
        trace_generation_steps=64,
        max_frame_width=64,
    )
    urls = [
        part["image_url"]["url"]
        for message in request["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part["type"] == "image_url"
    ]
    assert len(set(urls)) == 1
    with Image.open(BytesIO(base64.b64decode(urls[0].split(",", 1)[1]))) as image:
        assert image.size == (64, 40)
