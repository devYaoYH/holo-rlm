"""Build slide-ready crops and measurements for the native free hotel rollout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from attribution import load_attribution


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/remote-results/hotel-official-tools-native-20260922-rerun/extracted"
RUN = SOURCE / "run/traj-20260922T055823Z-397ab3c2af"
TRACE_ID = "trace-20260922T060113Z-54ddd44b7a3c"
TRACE = SOURCE / "traces" / TRACE_ID
VIEWER = SOURCE / "final-click-attribution"
OUTPUT = ROOT / "artifacts/screenspot-presentation/hotel-freegen-official-tools-v1"

# Keep the text, prices, and buttons at source resolution; only discard the
# decorative image column and fixture debug panel.
CROP = (455, 73, 785, 720)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    attribution = load_attribution(TRACE)
    y_span = next(span for span in attribution.generated_spans if span.parameter == "y")
    frame_masses: list[float] = []
    for frame in attribution.frames:
        maps = [
            attribution.aggregate(step, method="value_norm_rollout", frame_index=frame.index)
            for step in y_span.steps
        ]
        frame_masses.append(float(np.mean(np.stack(maps), axis=0).sum()))
    total_mass = sum(frame_masses)
    frame_shares = [mass / total_mass for mass in frame_masses]

    generated: list[dict[str, object]] = []
    for path in sorted((RUN / "actions").glob("*.json")):
        action = json.loads(path.read_text())
        message = action["raw_output"]["choices"][0]["message"]
        if isinstance(message.get("content"), str):
            structured = json.loads(message["content"])
            model_call = structured["tool_calls"][0]
        else:
            function = message["tool_calls"][0]["function"]
            arguments = function["arguments"]
            model_call = json.loads(arguments) if isinstance(arguments, str) else arguments
        generated.append({"model_call": model_call, "applied_action": action["normalized"]})

    assets: list[dict[str, object]] = []
    for frame_index in range(3):
        source = VIEWER / f"saliency-value-norm-rollout-frame-{frame_index:03d}-{y_span.id}.png"
        with Image.open(source).convert("RGB") as image:
            if image.size != (1024, 720):
                raise ValueError(f"unexpected source image size: {image.size}")
            crop = image.crop(CROP)
        output = OUTPUT / f"final-y-attribution-frame-{frame_index}.png"
        crop.save(output, optimize=True)
        assets.append(
            {
                "frame": frame_index,
                "path": output.relative_to(ROOT).as_posix(),
                "width": crop.width,
                "height": crop.height,
                "sha256": sha256(output),
            }
        )

    annotations = json.loads((RUN / "annotations.json").read_text())
    manifest = json.loads((RUN / "manifest.json").read_text())
    config = manifest["fixture"]["scenario_config"]
    hotels = {hotel["id"]: hotel for hotel in config["hotels"]}
    selected = hotels[annotations["final_state"]["selected_hotel_id"]]
    oracle = hotels[annotations["target"]["hotel_id"]]
    final_model_call = generated[-1]["model_call"]
    final_applied = generated[-1]["applied_action"]
    layout = config["layout"]
    oracle_button_left = layout["content_left"] + layout["button_x_offset"]
    oracle_button_top = (
        layout["first_card_y"]
        + oracle["index"] * (layout["card_height"] + layout["card_gap"])
        + layout["button_y_offset"]
        - annotations["final_state"]["scroll_y"]
    )
    summary = {
        "schema_version": 1,
        "trajectory_id": RUN.name,
        "trace_id": TRACE_ID,
        "protocol": {
            "generation": "free",
            "teacher_forcing": False,
            "system_prompt": "repository_checked_in_hotel_prompt",
            "tool_schema": "HoloDesktop runtime 0.1.10 structured tool set",
            "tool_serialization": "structured_outputs JSON",
            "grammar_constrained_decoding": False,
            "input_resolution": [1024, 720],
            "patch_grid": [22, 32],
            "raw_vision_grid": [44, 64],
            "max_pixels": 16_777_216,
            "downsampled_before_model": False,
            "maximum_retained_screenshots": 3,
            "coordinates": "normalized 0-1000",
            "attribution": "value_norm_rollout",
            "attributed_span": y_span.label,
        },
        "visualization_crop_xyxy": list(CROP),
        "actions": generated,
        "completed_steps": annotations["completed_steps"],
        "task_success": annotations["task_success"],
        "selected_hotel": annotations["final_state"]["selected_hotel_id"],
        "selected_name": selected["name"],
        "selected_price": selected["price"],
        "oracle_hotel": annotations["target"]["hotel_id"],
        "oracle_name": oracle["name"],
        "oracle_price": oracle["price"],
        "final_model_click": [final_model_call["x"], final_model_call["y"]],
        "final_projected_click": [final_applied["x"], final_applied["y"]],
        "oracle_button_bbox_pixel": [
            oracle_button_left,
            oracle_button_top,
            oracle_button_left + layout["button_width"],
            oracle_button_top + layout["button_height"],
        ],
        "retained_scroll_offsets": [500, 1000, annotations["final_state"]["scroll_y"]],
        "final_y_frame_mass": frame_masses,
        "final_y_frame_share": frame_shares,
        "assets": assets,
        "source_archive_sha256": "534832999ef9b4b5e06f81a79f4430af7ee3ff58ad262703f5c5fd24a3c48d5c",
        "attribution_archive_sha256": "26da859fdc5b42371e2f0a158bf1d201c2a12fe54ce1593ed2f87276131a91b6",
    }
    (OUTPUT / "hotel-freegen-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
