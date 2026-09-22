"""Summarize paired free-generation hotel trajectories and golden-route bounds."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demo.renderer import hotel_click, hotel_scroll  # noqa: E402


DEFAULT_INPUT = (
    ROOT
    / "data/remote-results/hotel-freegen-paired-v1-20260922/hotel-freegen-paired-v1"
)
DEFAULT_MANIFEST = ROOT / "benchmarks/frozen_eval_v2.json"
DEFAULT_OUTPUT = ROOT / "artifacts/screenspot-presentation/hotel-freegen-paired-v1-summary.json"


def _golden_route(config: dict) -> dict:
    scroll_y = 0
    seen_bottom = False
    scroll_offsets = [0]
    actions: list[dict] = []
    cheapest_id = config["cheapest_id"]
    for _ in range(8):
        max_scroll = int(config["max_scroll"])
        if scroll_y >= max_scroll:
            seen_bottom = True
        x, y = hotel_click(config, {"scroll_y": scroll_y}, cheapest_id)
        button_visible = config["layout"]["header_height"] < y < config["viewport"]["height"]
        if seen_bottom and button_visible:
            actions.append({"action": "click", "x": x, "y": y})
            break
        if not seen_bottom:
            delta = min(500, max_scroll - scroll_y)
        else:
            target = hotel_scroll(config, cheapest_id)
            delta = max(-500, min(500, target - scroll_y))
        scroll_y = max(0, min(max_scroll, scroll_y + delta))
        scroll_offsets.append(scroll_y)
        actions.append({"action": "scroll", "delta_y": delta})
    else:
        raise ValueError(f"golden route did not terminate for {config['item_id']}")
    return {
        "actions": actions,
        "scroll_action_count": sum(action["action"] == "scroll" for action in actions),
        "scroll_offsets": scroll_offsets,
        "total_action_count": len(actions),
    }


def _action_sequence(bundle: Path) -> list[str]:
    actions = []
    for path in sorted((bundle / "actions").glob("*.json")):
        payload = json.loads(path.read_text())
        actions.append(str(payload["normalized"]["action"]))
    return actions


def _model_rows(root: Path, model: str, configs: dict[str, dict]) -> list[dict]:
    rows = []
    for bundle in sorted((root / model / "bundles").iterdir()):
        if not bundle.is_dir():
            continue
        annotations = json.loads((bundle / "annotations.json").read_text())
        config = configs[bundle.name]
        rows.append(
            {
                "item_id": bundle.name,
                "task_success": bool(annotations["task_success"]),
                "captured_click": bool(annotations["captured_click"]),
                "completed_steps": int(annotations["completed_steps"]),
                "terminal_reason": annotations["terminal_reason"],
                "error": annotations.get("error"),
                "action_sequence": _action_sequence(bundle),
                "cheapest_placement": config["cheapest_placement"],
                "runner_up_gap": int(config["runner_up_gap"]),
                "currency": config["currency"],
                "hotel_count": len(config["hotels"]),
                "viewport": config["viewport"],
                "golden_route": _golden_route(config),
            }
        )
        rows[-1]["golden_route_within_three_scrolls"] = rows[-1]["golden_route"]["scroll_action_count"] <= 3
    return rows


def _aggregate(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "success_count": sum(row["task_success"] for row in rows),
        "click_count": sum(row["captured_click"] for row in rows),
        "valid_action_through_step_limit_count": sum(row["error"] is None for row in rows),
        "terminal_reason_counts": dict(sorted(Counter(row["terminal_reason"] for row in rows).items())),
        "positive_case_ids": [row["item_id"] for row in rows if row["task_success"]],
    }


def _diversity(rows: list[dict]) -> dict:
    return {
        "cheapest_placements": dict(sorted(Counter(row["cheapest_placement"] for row in rows).items())),
        "currencies": dict(sorted(Counter(row["currency"] for row in rows).items())),
        "hotel_counts": dict(sorted(Counter(str(row["hotel_count"]) for row in rows).items())),
        "runner_up_gap_range": [min(row["runner_up_gap"] for row in rows), max(row["runner_up_gap"] for row in rows)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    configs = {item["item_id"]: item["config"] for item in manifest["items"]}
    holo = _model_rows(args.input, "holo", configs)
    qwen = _model_rows(args.input, "qwen", configs)
    item_ids = [row["item_id"] for row in holo]
    if item_ids != [row["item_id"] for row in qwen]:
        raise ValueError("Holo and Qwen item sets differ")
    eligible_ids = [row["item_id"] for row in holo if row["golden_route_within_three_scrolls"]]
    excluded_ids = [row["item_id"] for row in holo if not row["golden_route_within_three_scrolls"]]
    holo_primary = [row for row in holo if row["item_id"] in eligible_ids]
    qwen_primary = [row for row in qwen if row["item_id"] in eligible_ids]

    output = {
        "experiment": "hotel_freegen_paired_v1",
        "protocol": {
            "mode": "independent free generation",
            "checkpoints": ["Holo-3.1-4B", "Qwen3.5-4B"],
            "tool_contract": "HoloDesktop runtime 0.1.10",
            "coordinate_space": "normalized_0_1000",
            "maximum_model_actions": 4,
            "visual_history": "current screenshot plus at most two retained screenshots",
            "trace_capture": "Holo test-0010: three successful action traces; Qwen test-0010: one pre-action trace before invalid structured output",
            "selection": "six primary frozen cases whose deterministic golden route requires no more than three scroll actions before the click; one retained diagnostic case is excluded after audit correction",
        },
        "item_ids": item_ids,
        "primary_item_ids": eligible_ids,
        "excluded_diagnostic_item_ids": excluded_ids,
        "primary_diversity": _diversity(holo_primary),
        "all_collected_diversity": _diversity(holo),
        "holo": {"primary_aggregate": _aggregate(holo_primary), "all_collected_aggregate": _aggregate(holo), "cases": holo},
        "qwen": {"primary_aggregate": _aggregate(qwen_primary), "all_collected_aggregate": _aggregate(qwen), "cases": qwen},
        "interpretation": {
            "headline": "Under the four-action cap, Holo completes two of six audited primary cases and Qwen completes none.",
            "failure_breakdown": "Both checkpoints show action-interface fragility: invalid structured outputs, excessive scrolling, or incorrect clicks.",
            "claim_boundary": "Small exploratory synthetic slice; test-0000 is retained as a diagnostic but excluded because its deterministic golden route needs four scrolls. The experiment does not isolate visual understanding from policy, formatting, or stopping behavior.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
