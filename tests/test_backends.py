from PIL import Image

from demo.backends import MODEL_ACTION_SCHEMA, build_request, project_model_action


def test_holo_click_coordinates_project_from_normalized_space() -> None:
    assert project_model_action({"action": "click", "x": 835, "y": 551}, (1280, 800)) == {
        "action": "click",
        "x": 1068,
        "y": 440,
    }
    click_schema = MODEL_ACTION_SCHEMA["oneOf"][0]["properties"]
    assert click_schema["x"]["maximum"] == 1000
    assert click_schema["y"]["maximum"] == 1000
    scroll_schema = MODEL_ACTION_SCHEMA["oneOf"][1]["properties"]["delta_y"]
    assert "negative scrolls down" in scroll_schema["description"]
    assert project_model_action({"action": "scroll", "delta_y": -1600}, (1280, 800)) == {
        "action": "scroll",
        "delta_y": 1600,
    }


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
    assert "negative delta_y to scroll down" in system_prompt["content"]


def test_non_cheapest_task_does_not_get_cheapest_constraint() -> None:
    request = build_request(
        [{"role": "user", "content": "Scroll down and inspect the results."}],
        Image.new("RGB", (16, 16)),
        "test-model",
    )
    assert "VIEW DETAILS OF ONLY THE CHEAPEST HOTEL" not in request["messages"][0]["content"]
