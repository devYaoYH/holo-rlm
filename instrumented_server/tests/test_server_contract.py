from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from instrumented_holo.app import create_app
from instrumented_holo.model import (
    InstrumentedHolo,
    generated_parameter_token_spans,
    parse_assistant_output,
)
from instrumented_holo.settings import Settings
from instrumented_holo.traces import TraceOptions, TraceWriter, append_trace_response


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


def test_compact_value_norm_capture_defaults_on_without_full_kv_capture() -> None:
    options = TraceOptions.from_request({"trace": {"capture_kv": False}})
    assert options.capture_value_norms is True
    assert options.capture_rollout is True
    assert options.capture_kv is False

    disabled = TraceOptions.from_request({"trace": {"capture_value_norms": False}})
    assert disabled.capture_value_norms is False

    logprobs = TraceOptions.from_request({"trace": {"capture_logprobs": True}})
    assert logprobs.capture_logprobs is True


def test_generated_parameter_spans_label_action_and_coordinates() -> None:
    pieces = (
        "<parameter=action>",
        "click",
        "</parameter><parameter=x>\n",
        "48",
        "2\n</parameter><parameter=y>",
        "351",
        "</parameter>",
    )
    offsets = []
    cursor = 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    labels, spans = generated_parameter_token_spans("".join(pieces), tuple(offsets))
    assert labels == (None, "action", None, "x", "x", "y", None)
    assert spans == (
        ("action", "click", (1,)),
        ("x", "482", (3, 4)),
        ("y", "351", (5,)),
    )


def test_generated_parameter_spans_label_official_localizer_json_coordinates() -> None:
    pieces = ('{"x":', "482", ',"y":', "351", "}")
    offsets = []
    cursor = 0
    for piece in pieces:
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    labels, spans = generated_parameter_token_spans("".join(pieces), tuple(offsets))
    assert labels == (None, "x", None, "y", None)
    assert spans == (("x", "482", (1,)), ("y", "351", (3,)))


def test_response_is_hashed_into_trace_manifest(tmp_path: Path) -> None:
    trace_id, writer = TraceWriter.create(tmp_path)
    writer.write_json("request.json", {"messages": []})
    writer.write_json("manifest.json", writer.manifest())
    append_trace_response(tmp_path, trace_id, {"choices": []})
    manifest = json.loads((tmp_path / trace_id / "manifest.json").read_text())
    names = {item["path"] for item in manifest["files"]}
    assert names == {"request.json", "response.json"}


def test_model_metadata_maps_captured_attention_blocks_to_transformer_layers(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "test",
                "text_config": {
                    "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"]
                },
            }
        )
    )
    engine = InstrumentedHolo(Settings(model_path=tmp_path, trace_dir=tmp_path / "traces"))
    assert engine.model_metadata()["attention_layer_indices"] == [1, 3]


def test_processor_metadata_records_effective_runtime_pixel_bounds(tmp_path: Path) -> None:
    engine = InstrumentedHolo(Settings(model_path=tmp_path, trace_dir=tmp_path / "traces"))
    engine.processor = SimpleNamespace(
        image_processor=SimpleNamespace(size={"shortest_edge": 65_536, "longest_edge": 262_144})
    )
    metadata = engine.processor_metadata()
    assert metadata["runtime_image_processor_size"] == {
        "shortest_edge": 65_536,
        "longest_edge": 262_144,
    }
