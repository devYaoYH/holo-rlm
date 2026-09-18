from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from instrumented_holo.app import create_app
from instrumented_holo.model import parse_assistant_output
from instrumented_holo.settings import Settings
from instrumented_holo.traces import TraceWriter, append_trace_response


def test_native_tool_call_is_projected_without_hidden_reasoning() -> None:
    text = """<think>private chain</think>
<tool_call>
<function=desktop_action>
<parameter=action>
scroll
</parameter>
<parameter=delta_y>
600
</parameter>
</function>
</tool_call>"""
    visible, calls = parse_assistant_output(text)
    assert visible == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "desktop_action"
    assert json.loads(calls[0]["function"]["arguments"]) == {"action": "scroll", "delta_y": 600}
    assert "private chain" not in json.dumps(calls)


def test_prefilled_thinking_is_not_leaked_when_generation_truncates() -> None:
    assert parse_assistant_output("private unfinished chain", thinking_enabled=True) == ("", [])
    assert parse_assistant_output("visible answer", thinking_enabled=False) == ("visible answer", [])


class FakeEngine:
    model = None

    def complete(self, **kwargs):
        assert kwargs["tools"][0]["function"]["name"] == "desktop_action"
        assert kwargs["trace_options"].capture_hidden_states is True
        return (
            "<think>omit me</think><tool_call><function=desktop_action>"
            "<parameter=action>wait</parameter><parameter=milliseconds>250</parameter>"
            "</function></tool_call>",
            10,
            4,
            None,
            True,
        )


def test_openai_tools_contract_reaches_engine(tmp_path: Path) -> None:
    settings = Settings(model_path=tmp_path, trace_dir=tmp_path / "traces")
    client = TestClient(create_app(settings, engine=FakeEngine()))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Hcompany/Holo-3.1-4B",
            "messages": [{"role": "user", "content": "wait"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "desktop_action",
                        "parameters": {
                            "type": "object",
                            "properties": {"action": {"type": "string"}},
                        },
                    },
                }
            ],
            "trace": {"capture_hidden_states": True},
            "max_tokens": 8,
        },
    )
    response.raise_for_status()
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
    call = payload["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"action": "wait", "milliseconds": 250}
    assert payload["choices"][0]["message"]["content"] is None


def test_response_is_hashed_into_trace_manifest(tmp_path: Path) -> None:
    trace_id, writer = TraceWriter.create(tmp_path)
    writer.write_json("request.json", {"messages": []})
    writer.write_json("manifest.json", writer.manifest())
    append_trace_response(tmp_path, trace_id, {"choices": []})
    manifest = json.loads((tmp_path / trace_id / "manifest.json").read_text())
    names = {item["path"] for item in manifest["files"]}
    assert names == {"request.json", "response.json"}
