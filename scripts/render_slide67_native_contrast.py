"""Render readable slide crops from native ScreenSpot prompt-contrast outputs."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/remote-results/slide67-native-controls-20260922"
OUTPUT = ROOT / "artifacts/screenspot-presentation/slide67-native-contrast-v1"


CASES = {
    "powerpoint_windows_63": {
        "result": RESULTS / "extracted-63/run/slide67-native-controls-63/contrast",
        "crop": (250, 100, 1000, 680),
        "tight_crop": (470, 330, 760, 550),
        "click": (601.92, 437.4),
    },
    "powerpoint_windows_54": {
        "result": RESULTS / "extracted-54/run/slide67-native-controls-54/contrast",
        "crop": (0, 0, 900, 340),
        "tight_crop": (250, 0, 500, 190),
        "click": (357.12, 131.4),
    },
}


def _annotate(image: Image.Image, bbox: tuple[float, ...], click: tuple[float, float]) -> Image.Image:
    rendered = image.convert("RGB")
    draw = ImageDraw.Draw(rendered)
    line_width = 8
    x1, y1, x2, y2 = bbox
    draw.rectangle((x1, y1, x2, y2), outline=(24, 168, 117), width=line_width)
    x, y = click
    radius = 18
    for width, color in ((12, (16, 21, 28)), (6, (255, 255, 255))):
        draw.line((x - radius, y, x + radius, y), fill=color, width=width)
        draw.line((x, y - radius, x, y + radius), fill=color, width=width)
    return rendered


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema_version": 1,
        "definition": {
            "causal": "value-norm rollout minus mean captured generated-token map strictly before the x/y value span",
            "prompt_difference": "L1-normalized target-instruction value-norm rollout minus the mean of four same-image diverse-instruction maps",
        },
        "cases": {},
    }
    for sample_id, case in CASES.items():
        result_dir = case["result"]
        analysis = json.loads((result_dir / "analysis.json").read_text())
        bbox = tuple(float(value) for value in analysis["bbox"])
        click = case["click"]
        case_dir = OUTPUT / sample_id
        case_dir.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, str] = {}
        for method, source_name in (
            ("causal", "preview-causal.png"),
            ("prompt_difference", "preview-target-minus-prompt-baseline.png"),
        ):
            with Image.open(result_dir / source_name) as opened:
                annotated = _annotate(opened, bbox, click)
            destination = case_dir / f"{method}-context.png"
            annotated.crop(case["crop"]).save(destination)
            outputs[method] = str(destination.relative_to(ROOT))
            if "tight_crop" in case:
                tight = case_dir / f"{method}-tight.png"
                annotated.crop(case["tight_crop"]).resize((1000, 760), Image.Resampling.LANCZOS).save(tight)
                outputs[f"{method}_tight"] = str(tight.relative_to(ROOT))

        summary["cases"][sample_id] = {
            "instruction": analysis["target_instruction"],
            "bbox": analysis["bbox"],
            "predicted_click": analysis["predicted_click"],
            "correct": analysis["correct"],
            "controls": analysis["controls"],
            "stability": analysis["stability"],
            "metrics": {
                method: analysis["metrics"][method][0]
                for method in ("causal", "prompt_difference")
            },
            "outputs": outputs,
        }
    (OUTPUT / "slide67-native-contrast-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({"output": str(OUTPUT), "cases": len(CASES)}, indent=2))


if __name__ == "__main__":
    main()
