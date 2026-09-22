"""Build slide-ready crops and measurements for the native free hotel rollout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from attribution import load_attribution


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/remote-results/hotel-freegen-native-20260922/extracted"
RUN = SOURCE / "run/traj-20260922T035615Z-0b7c877373"
TRACE_ID = "trace-20260922T035643Z-a6a081aa747b"
TRACE = SOURCE / "traces" / TRACE_ID
VIEWER = SOURCE / "hotel-test0035-freegen-attribution" / TRACE_ID
OUTPUT = ROOT / "artifacts/screenspot-presentation/hotel-freegen-native-v1"

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
        generated.append(action["raw_output"]["choices"][0]["message"]["tool_calls"][0]["function"])

    assets: list[dict[str, object]] = []
    for frame_index in range(3):
        source = VIEWER / f"saliency-value-norm-rollout-frame-{frame_index:03d}-parameter-2.png"
        with Image.open(source).convert("RGB") as image:
            if image.size != (1024, 720):
                raise ValueError(f"unexpected source image size: {image.size}")
            crop = image.crop(CROP)
        if frame_index == 2:
            draw = ImageDraw.Draw(crop)
            x0, y0, _, _ = CROP
            # Final-frame buttons: generated click selects Juniper; oracle is Lumen.
            draw.rounded_rectangle((601 - x0, 157 - y0, 714 - x0, 215 - y0), radius=8, outline="#F05A3C", width=6)
            draw.rounded_rectangle((601 - x0, 414 - y0, 714 - x0, 472 - y0), radius=8, outline="#18A875", width=6)
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
    summary = {
        "schema_version": 1,
        "trajectory_id": RUN.name,
        "trace_id": TRACE_ID,
        "protocol": {
            "generation": "free",
            "teacher_forcing": False,
            "input_resolution": [1024, 720],
            "patch_grid": [22, 32],
            "max_pixels": 16_777_216,
            "downsampled_before_model": False,
            "coordinates": "normalized 0-1000",
            "attribution": "value_norm_rollout",
            "attributed_span": y_span.label,
        },
        "actions": generated,
        "completed_steps": annotations["completed_steps"],
        "task_success": annotations["task_success"],
        "selected_hotel": annotations["final_state"]["selected_hotel_id"],
        "selected_name": "Juniper Signal Inn",
        "selected_price": 312,
        "oracle_hotel": annotations["target"]["hotel_id"],
        "oracle_name": annotations["target"]["hotel"],
        "oracle_price": annotations["target"]["price"],
        "final_model_click": [634, 225],
        "final_projected_click": [649, 162],
        "final_y_frame_mass": frame_masses,
        "final_y_frame_share": frame_shares,
        "recall_stress_controls": {
            "items": ["test-0015", "test-0075"],
            "completed_steps": [0, 0],
            "model_native_arguments": '{"action":"scroll","-delta_y":-500}',
            "outcome": "invalid tool call before first action",
        },
        "assets": assets,
        "source_archive_sha256": "32a18a4c053b60127f52e2ddc449351b00250c3a20c7e05be2f5fda410f84843",
    }
    (OUTPUT / "hotel-freegen-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
