from PIL import Image

from demo.backends import (
    HOLO_DESKTOP_STEP_SCHEMA,
    MODEL_ACTION_SCHEMA,
    OpenAIBackend,
    build_request,
    project_model_action,
)


def test_holo_click_coordinates_project_from_normalized_space() -> None:
    assert project_model_action(
        {"tool_name": "click_desktop", "element": "target", "x": 835, "y": 551},
        (1280, 800),
    ) == {
        "action": "click",
        "x": 1068,
        "y": 440,
    }
    click_schema = MODEL_ACTION_SCHEMA["oneOf"][0]["properties"]
    assert click_schema["x"]["maximum"] == 1000
    assert click_schema["y"]["maximum"] == 1000
    scroll_schema = MODEL_ACTION_SCHEMA["oneOf"][1]["properties"]["delta_y"]
    assert "negative scrolls down" in scroll_schema["description"]
    assert project_model_action(
        {
            "tool_name": "scroll_desktop",
            "element": "results",
            "x": 500,
            "y": 500,
            "direction": "down",
            "scroll_size": 10,
        },
        (1280, 800),
    ) == {
        "action": "scroll",
        "delta_y": 500,
    }
    assert project_model_action(
        {
            "tool_name": "scroll_desktop",
            "element": "results",
            "x": 500,
            "y": 500,
            "direction": "down",
            "scroll_size": 100,
        },
        (1280, 800),
    ) == {"action": "scroll", "delta_y": 1600}


def test_official_holo_desktop_contract_has_expected_action_tools() -> None:
    definitions = HOLO_DESKTOP_STEP_SCHEMA["$defs"]
    assert definitions["click_desktop"]["required"] == ["tool_name", "element", "x", "y"]
    assert definitions["scroll_desktop"]["required"] == [
        "tool_name",
        "element",
        "x",
        "y",
        "direction",
    ]
    assert definitions["scroll_desktop"]["properties"]["direction"]["enum"] == [
        "up",
        "down",
        "left",
        "right",
    ]
    assert "delta_y" not in definitions["scroll_desktop"]["properties"]


def test_cheapest_task_gets_exclusive_system_constraint() -> None:
    request = build_request(
        [
            {
                "role": "user",
                "content": "Find the hotel with the lowest nightly price across all deterministic search results.",
            }
        ],
        Image.new("RGB", (16, 16)),
        "test-model",
        normalized_coordinates=True,
    )
    system_prompt = request["messages"][0]
    assert system_prompt["role"] == "system"
    assert "VIEW DETAILS OF ONLY THE CHEAPEST HOTEL" in system_prompt["content"]
    assert "Never open View details for any other hotel" in system_prompt["content"]
    assert "If the cheapest hotel's card is not visible, scroll back to it" in system_prompt["content"]
    assert "deterministic localhost booking fixture" not in system_prompt["content"]
    assert "do not scroll again" not in system_prompt["content"].lower()
    assert "scroll_desktop with direction='down'" in system_prompt["content"]
    assert "scroll_size between 8 and 12" in system_prompt["content"]
    assert request["temperature"] == 0.8
    assert request["chat_template_kwargs"] == {"enable_thinking": True}
    assert request["structured_outputs"]["json"] == HOLO_DESKTOP_STEP_SCHEMA
    assert "tools" not in request
    observation = request["messages"][-1]["content"]
    assert observation[0]["text"].startswith("<observation>")
    assert observation[-1]["text"].strip() == "</observation>"


def test_official_desktop_request_retains_at_most_three_screenshots() -> None:
    messages = [{"role": "user", "content": "Find the lowest nightly price."}]
    for index in range(4):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"<observation>frame {index}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{index}"}},
                        {"type": "text", "text": "</observation>"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": '{"note":"remember price","thought":"scroll","tool_calls":[]}',
                },
            ]
        )

    request = build_request(
        messages,
        Image.new("RGB", (16, 16)),
        "test-model",
        normalized_coordinates=True,
    )
    visual_messages = [
        message
        for message in request["messages"]
        if isinstance(message.get("content"), list)
        and any(part.get("type") == "image_url" for part in message["content"])
    ]
    assert len(visual_messages) == 3
    assert sum(
        isinstance(message.get("content"), str)
        and "three-screenshot retention window" in message["content"]
        for message in request["messages"]
    ) == 2


def test_non_cheapest_task_does_not_get_cheapest_constraint() -> None:
    request = build_request(
        [{"role": "user", "content": "Scroll down and inspect the results."}],
        Image.new("RGB", (16, 16)),
        "test-model",
    )
    assert "VIEW DETAILS OF ONLY THE CHEAPEST HOTEL" not in request["messages"][0]["content"]


def test_local_backend_keeps_128_token_floor_for_complete_native_tool_call(monkeypatch) -> None:
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

        def raise_for_status(self):
            return None

    class Client:
        def __init__(self, **_kwargs):
            self.request = None

        def get(self, _url):
            return Response({"data": []})

        def post(self, _url, *, json):
            self.request = json
            return Response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"note":null,"thought":"continue","tool_calls":['
                                    '{"tool_name":"scroll_desktop","element":"hotel results",'
                                    '"x":500,"y":500,"direction":"down","scroll_size":10}]}'
                                )
                            }
                        }
                    ]
                }
            )

        def close(self):
            return None

    monkeypatch.setattr("demo.backends.httpx.Client", Client)
    backend = OpenAIBackend("http://localhost/v1", "test-model", "test", "test", trace_generation_steps=64)
    _request, _response, action = backend.decide(
        step=0,
        messages=[{"role": "user", "content": "Scroll down."}],
        image=Image.new("RGB", (1280, 800)),
        config={},
        state={},
    )
    assert backend._http.request["max_tokens"] == 384
    assert backend._http.request["trace"]["max_generation_steps"] == 64
    assert action == {"action": "scroll", "delta_y": 500}


def test_local_backend_can_disable_activation_tracing(monkeypatch) -> None:
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

        def raise_for_status(self):
            return None

    class Client:
        def __init__(self, **_kwargs):
            self.request = None

        def get(self, _url):
            return Response({"data": []})

        def post(self, _url, *, json):
            self.request = json
            return Response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"note":null,"thought":"continue","tool_calls":['
                                    '{"tool_name":"scroll_desktop","element":"hotel results",'
                                    '"x":500,"y":500,"direction":"down","scroll_size":10}]}'
                                )
                            }
                        }
                    ]
                }
            )

        def close(self):
            return None

    monkeypatch.setattr("demo.backends.httpx.Client", Client)
    backend = OpenAIBackend("http://localhost/v1", "test-model", "test", "test", trace_generation_steps=0)
    backend.decide(
        step=0,
        messages=[{"role": "user", "content": "Scroll down."}],
        image=Image.new("RGB", (1024, 720)),
        config={},
        state={},
    )

    assert "trace" not in backend._http.request
    assert backend._http.request["max_tokens"] == 384
