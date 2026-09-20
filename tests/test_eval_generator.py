import json
from collections import Counter
from pathlib import Path

import pytest

from apps.booking_fixture.generator import (
    GENERATOR_VERSION,
    build_manifest,
    config_sha256,
    generate_config,
    load_manifest,
)
from demo.backends import ScriptedBackend
from demo.fixture import running_fixture
from demo.renderer import hotel_click, hotel_scroll, render_fixture
from demo.runner import capture_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = PROJECT_ROOT / "benchmarks" / "frozen_eval_v2.json"


def test_generator_is_deterministic_and_axes_are_independently_seeded() -> None:
    first = generate_config(37, "test")
    assert first == generate_config(37, "test")
    assert first != generate_config(38, "test")
    assert set(first["axis_seeds"]) == {"content", "order", "prices", "visuals", "layout"}
    assert len(set(first["axis_seeds"].values())) == 5


def test_frozen_manifest_exactly_matches_generator_and_hashes() -> None:
    frozen = load_manifest(FROZEN_MANIFEST)
    assert frozen == build_manifest(split="test", count=120)
    assert frozen["generator_version"] == GENERATOR_VERSION
    assert frozen["count"] == 120
    assert frozen["diversity"]["cheapest_placements"] == {"bottom": 40, "middle": 40, "top": 40}
    assert frozen["diversity"]["runner_up_margin_bands"] == {"close": 40, "medium": 40, "wide": 40}
    assert frozen["diversity"]["unique_orderings"] == 120
    assert frozen["diversity"]["unique_price_vectors"] == 120
    assert len({item["item_id"] for item in frozen["items"]}) == 120
    for item in frozen["items"]:
        assert item["config_sha256"] == config_sha256(item["config"])


def test_manifest_verification_rejects_modified_config(tmp_path: Path) -> None:
    manifest = build_manifest(split="test", count=2)
    manifest["items"][0]["config"]["hotels"][0]["price"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="config hash mismatch"):
        load_manifest(path)


def test_frozen_eval_has_balanced_price_and_layout_diversity() -> None:
    configs = [item["config"] for item in load_manifest(FROZEN_MANIFEST)["items"]]
    assert {len(config["hotels"]) for config in configs} == {6, 7, 8, 9, 10}
    assert {config["cheapest_index"] for config in configs} == set(range(10))
    assert Counter(config["cheapest_placement"] for config in configs) == {
        "top": 40,
        "middle": 40,
        "bottom": 40,
    }
    assert {config["currency"] for config in configs} == {"$", "€", "£"}
    assert Counter(
        "close"
        if config["runner_up_gap"] <= 3
        else "medium"
        if config["runner_up_gap"] <= 18
        else "wide"
        for config in configs
    ) == {"close": 40, "medium": 40, "wide": 40}
    assert len(
        {
            min(config["hotels"], key=lambda hotel: (hotel["price"], hotel["id"]))["name"]
            for config in configs
        }
    ) > 8
    assert len({tuple(hotel["id"] for hotel in config["hotels"]) for config in configs}) > 100
    assert len({tuple(hotel["price"] for hotel in config["hotels"]) for config in configs}) == 120
    assert len({tuple(config["viewport"].values()) for config in configs}) == 5
    assert len({config["layout"]["price_x_offset"] for config in configs}) > 20
    assert len({config["layout"]["price_y_offset"] for config in configs}) == 4
    assert len({config["layout"]["card_height"] for config in configs}) == 5
    for config in configs:
        prices = [hotel["price"] for hotel in config["hotels"]]
        assert len(prices) == len(set(prices))
        assert prices.index(min(prices)) == config["cheapest_index"]
        cheapest = config["hotels"][config["cheapest_index"]]
        state = {"scroll_y": hotel_scroll(config, cheapest["id"])}
        x, y = hotel_click(config, state, cheapest["id"])
        assert config["layout"]["header_height"] < y < config["viewport"]["height"]
        assert 0 < x < config["viewport"]["width"]


def test_train_dev_and_test_content_namespaces_do_not_overlap() -> None:
    names_by_split = {
        split: {hotel["name"] for index in range(20) for hotel in generate_config(index, split)["hotels"]}
        for split in ("train", "dev", "test")
    }
    assert names_by_split["train"].isdisjoint(names_by_split["dev"])
    assert names_by_split["train"].isdisjoint(names_by_split["test"])
    assert names_by_split["dev"].isdisjoint(names_by_split["test"])


@pytest.mark.parametrize("item_index", [0, 7, 18, 39, 64, 89, 119])
def test_generated_geometry_renders_and_target_button_is_clickable(item_index: int) -> None:
    config = generate_config(item_index, "test")
    cheapest = min(config["hotels"], key=lambda hotel: (hotel["price"], hotel["id"]))
    with running_fixture(seed=item_index, scenario_config=config) as fixture:
        scroll_y = hotel_scroll(config, cheapest["id"])
        state = fixture.apply({"action": "scroll", "delta_y": scroll_y})
        image = render_fixture(config, state)
        assert image.size == (config["viewport"]["width"], config["viewport"]["height"])
        x, y = hotel_click(config, state, cheapest["id"])
        final = fixture.apply({"action": "click", "x": x, "y": y})
        assert final["details_open"] is True
        assert final["selected_hotel_id"] == cheapest["id"]


def test_generated_scenario_capture_validates_and_replays(tmp_path: Path) -> None:
    config = generate_config(17, "test")
    bundle, result = capture_run(
        backend=ScriptedBackend(),
        output_root=tmp_path,
        seed=17,
        max_steps=16,
        task="cheapest",
        scenario_config=config,
    )
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert result["annotations"]["task_success"] is True
    assert result["replay"]["reproducible"] is True
    assert manifest["fixture"]["item_id"] == "test-0017"
    assert manifest["fixture"]["generator_version"] == GENERATOR_VERSION
    assert manifest["fixture"]["scenario_config"] == config
