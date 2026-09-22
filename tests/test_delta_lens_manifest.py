from pathlib import Path

from demo.backends import project_model_action
from scripts.build_delta_lens_manifest import build_manifest


def test_delta_lens_manifest_uses_official_protocols_and_common_hotel_history(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path / "manifest.json")
    cases = {case["id"]: case for case in manifest["cases"]}

    assert len(cases) == 8
    assert {case["score_reduction"] for case in cases.values()} == {"mean"}
    screen = cases["powerpoint_windows_59"]
    assert screen["protocol"] == "hcompany_element_localization_v1"
    assert [part["type"] for part in screen["request"]["messages"][0]["content"]] == [
        "image_path",
        "text",
    ]
    assert "tools" not in screen["request"]
    assert screen["request"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert screen["image_max_pixels"] == 16_777_216

    hotel_cases = [cases[f"hotel_test_0035_large_ui_step_{step}"] for step in range(3)]
    assert {case["image_max_pixels"] for case in hotel_cases} == {16_777_216}
    system = hotel_cases[0]["request"]["messages"][0]["content"]
    assert "You control a hotel-search interface" in system
    assert "If the cheapest hotel's card is not visible, scroll back to it" in system
    assert "localhost" not in system.lower()
    assert "fixture" not in system.lower()
    assert "do not scroll again" not in system.lower()
    assert [
        sum(
            part.get("type") == "image_path"
            for message in case["request"]["messages"]
            for part in (message.get("content") if isinstance(message.get("content"), list) else [])
        )
        for case in hotel_cases
    ] == [1, 2, 3]
    assert hotel_cases[0]["candidates"][0]["tool_action"] == {
        "tool_name": "scroll_desktop",
        "element": "hotel search results list",
        "x": 500,
        "y": 500,
        "direction": "down",
        "scroll_size": 10,
    }
    assert hotel_cases[2]["candidates"][0]["tool_action"] == {
        "tool_name": "click_desktop",
        "element": "Lumen Harbor Rooms View details button",
        "x": 642,
        "y": 616,
        "button": "left",
    }
    assert hotel_cases[2]["candidates"][1]["tool_action"] == {
        "tool_name": "click_desktop",
        "element": "visible non-cheapest hotel View details button",
        "x": 642,
        "y": 257,
        "button": "left",
    }
    assert hotel_cases[2]["case_metadata"]["coordinate_space"] == "normalized_0_1000"
    assert project_model_action(
        hotel_cases[2]["candidates"][0]["tool_action"],
        tuple(hotel_cases[2]["case_metadata"]["viewport"]),
    ) == {"action": "click", "x": 657, "y": 443}
    assert hotel_cases[2]["case_metadata"]["oracle_click_pixel"] == [657, 443]
    for case in hotel_cases:
        assert "tools" not in case["request"]
        click = case["request"]["structured_outputs"]["json"]["$defs"]["click_desktop"]
        assert "[0, 1000]" in click["properties"]["x"]["description"]
        assert "[0, 1000]" in click["properties"]["y"]["description"]
