import json
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from apps.booking_fixture.server import FixtureServer, fixture_config
from demo.fixture import running_fixture
from demo.renderer import target_click, target_scroll


def test_fixture_refuses_non_local_bind() -> None:
    with pytest.raises(ValueError, match="localhost"):
        FixtureServer(("0.0.0.0", 0))


def test_seeded_config_is_deterministic_and_varied() -> None:
    assert fixture_config(0) == fixture_config(4)
    assert fixture_config(0) != fixture_config(1)
    for seed in range(4):
        config = fixture_config(seed)
        targets = [hotel for hotel in config["hotels"] if hotel["id"] == config["target_id"]]
        assert len(targets) == 1
        assert config["expected_destination"] == "/details/harbor-lantern"
    assert len({fixture_config(0, variant)["target_name"] for variant in range(5)}) == 5
    assert len({fixture_config(0, variant)["initial_scroll_delta"] for variant in range(5)}) == 5


def test_scripted_trace_reaches_only_local_success_marker() -> None:
    with running_fixture(seed=0) as fixture:
        config = fixture.config()
        assert fixture.reset(0) == fixture.reset(0)
        fixture.apply({"action": "scroll", "delta_y": 1000})
        state = fixture.state()
        fixture.apply({"action": "scroll", "delta_y": target_scroll(config) - state["scroll_y"]})
        x, y = target_click(config, fixture.state())
        final = fixture.apply({"action": "click", "x": x, "y": y})
        assert final["details_open"] is True
        assert final["destination"] == "/details/harbor-lantern"
        page = urlopen(f"{fixture.base_url}{final['destination']}").read().decode()
        assert 'data-test="hotel-details-open"' in page
        with pytest.raises(HTTPError):
            urlopen(f"{fixture.base_url}/external")


def test_visible_state_endpoint() -> None:
    with running_fixture(seed=2) as fixture:
        page = urlopen(f"{fixture.base_url}/?seed=2").read().decode()
        assert 'id="fixture-state"' in page
        state = json.loads(urlopen(f"{fixture.base_url}/api/state").read())
        assert state["seed"] == 2
