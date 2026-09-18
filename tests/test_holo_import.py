import json

from apps.booking_fixture.server import fixture_config
from capture.validator import validate_bundle
from demo.holo_cli import import_runtime_bundle


def test_runtime_jsonl_is_preserved_but_not_replay_ready(tmp_path) -> None:
    event_log = tmp_path / "source.jsonl"
    event_log.write_text(
        json.dumps(
            {
                "type": "AgentEvent",
                "data": {
                    "kind": "policy_event",
                    "content": "safe summary",
                    "tool_reqs": [{"tool_name": "scroll", "args": {"delta_y": 500}}],
                },
            }
        )
        + "\n"
    )
    config = fixture_config(0)
    state = {"seed": 0, "variant": 0, "scroll_y": 0, "details_open": False, "destination": "/", "action_count": 0}
    bundle, validation = import_runtime_bundle(
        event_log=event_log,
        result={"returncode": 0},
        output_root=tmp_path / "bundles",
        fixture_config=config,
        fixture_base_url="http://127.0.0.1:8765",
        fixture_final_state=state,
        backend="hosted",
        model_id="unused",
    )
    assert validation["replay_ready"] is False
    assert validate_bundle(bundle, require_replay_ready=False)["raw_events"] == 2
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["research_eligible"] is False
