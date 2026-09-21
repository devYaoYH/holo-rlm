"""Matched screenshot case study built on the reusable activation-patching API."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageChops

from .activation_patching import (
    ActivationPatchingRunner,
    ImageRegion,
    attention_hook,
    capture_hook,
    capture_pre_hook,
    changed_pixel_fraction,
    coordinate_center,
    coordinate_tool_candidate,
    overlapping_patch_indices,
    parse_box,
    png_data_url,
    prediction_hook,
    require_patch_alignment,
    residual_hook,
    swap_equal_tiles,
)
from .model import InstrumentedHolo
from .settings import Settings

# Re-export the geometry helpers to preserve the original module API.
__all__ = ["coordinate_center", "overlapping_patch_indices", "parse_box", "swap_equal_tiles"]

SCREENSPOT_SYSTEM_PROMPT = """You are evaluating one GUI-grounding instruction on one static screenshot.
Return exactly one desktop_action tool call that clicks the requested visible target. Coordinates are normalized
integers from 0 to 1000, with (0, 0) at the top-left and (1000, 1000) at the bottom-right. Do not scroll, type,
wait, navigate, or explain. Each coordinate value must contain digits only: no quotes, commas, units, or prose.
Base the click only on the screenshot and instruction."""

CLICK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "x", "y"],
    "properties": {
        "action": {"const": "click"},
        "x": {"type": "integer", "minimum": 0, "maximum": 1000},
        "y": {"type": "integer", "minimum": 0, "maximum": 1000},
    },
}


def _messages(image: Image.Image, instruction: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SCREENSPOT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Instruction: {instruction}"},
                {"type": "image_url", "image_url": {"url": png_data_url(image)}},
            ],
        },
    ]


def _tools() -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": "desktop_action", "parameters": CLICK_SCHEMA}}]


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    settings = Settings.from_environment()
    engine = InstrumentedHolo(settings)
    engine.load()
    model = engine.model
    assert model is not None

    source_path = args.image.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as opened:
        clean_image = opened.convert("RGB")
    if args.corrupted_image is None:
        corrupted_source_path: Path | None = None
        corrupted_image = swap_equal_tiles(clean_image, args.target_swap_box, args.distractor_swap_box)
        corruption_operation = "lossless equal-size tile swap"
    else:
        corrupted_source_path = args.corrupted_image.expanduser().resolve()
        with Image.open(corrupted_source_path) as opened:
            corrupted_image = opened.convert("RGB")
        if corrupted_image.size != clean_image.size:
            raise ValueError("clean and corrupted images must have identical dimensions")
        corruption_operation = "caller-supplied matched image"
    clean_path = output_dir / "clean.png"
    corrupted_path = output_dir / "corrupted.png"
    clean_image.save(clean_path)
    corrupted_image.save(corrupted_path)

    correct_coordinate = coordinate_center(args.target_box, clean_image.size)
    distractor_coordinate = coordinate_center(args.distractor_box, clean_image.size)
    candidates = (
        coordinate_tool_candidate("original_target_slot", correct_coordinate),
        coordinate_tool_candidate("distractor_slot", distractor_coordinate),
    )

    # Equalize the regional comparison before mapping boxes to token positions.
    # This keeps region size from becoming an intervention confound.
    provisional_grid = (12, 20)
    target_candidates = overlapping_patch_indices(args.target_box, clean_image.size, provisional_grid)
    distractor_candidates = overlapping_patch_indices(args.distractor_box, clean_image.size, provisional_grid)
    matched_patch_count = min(len(target_candidates), len(distractor_candidates))
    regions = {
        "target": ImageRegion(0, args.target_box, matched_patch_count),
        "distractor": ImageRegion(0, args.distractor_box, matched_patch_count),
    }
    runner = ActivationPatchingRunner(engine, reduction="sum")
    clean = runner.prepare(
        name="clean",
        messages=_messages(clean_image, args.instruction),
        tools=_tools(),
        candidates=candidates,
        regions=regions,
    )
    # Recompute with the processor-derived grid if a future checkpoint changes it.
    grid = clean.image_grids[0]
    actual_matched_count = min(
        len(overlapping_patch_indices(args.target_box, clean_image.size, grid)),
        len(overlapping_patch_indices(args.distractor_box, clean_image.size, grid)),
    )
    if actual_matched_count != matched_patch_count:
        regions = {
            "target": ImageRegion(0, args.target_box, actual_matched_count),
            "distractor": ImageRegion(0, args.distractor_box, actual_matched_count),
        }
        clean = runner.prepare(
            name="clean",
            messages=_messages(clean_image, args.instruction),
            tools=_tools(),
            candidates=candidates,
            regions=regions,
        )
    corrupted = runner.prepare(
        name="corrupted",
        messages=_messages(corrupted_image, args.instruction),
        tools=_tools(),
        candidates=candidates,
        regions=regions,
    )
    require_patch_alignment(clean, corrupted)

    layers = model.model.language_model.layers
    if not 0 <= args.residual_layer < len(layers):
        raise RuntimeError(f"residual layer {args.residual_layer} is outside the model")
    if not 0 <= args.attention_layer < len(layers) or not hasattr(
        layers[args.attention_layer], "self_attn"
    ):
        raise RuntimeError(f"layer {args.attention_layer} is not a conventional full-attention layer")
    attention = layers[args.attention_layer].self_attn
    if not 0 <= args.attention_head < int(model.config.text_config.num_attention_heads):
        raise RuntimeError("attention head is outside the configured query-head range")
    for layer_index in args.mlp_layers:
        if not 0 <= layer_index < len(layers):
            raise RuntimeError(f"MLP layer {layer_index} is outside the model")

    capture: dict[str, Any] = {}
    capture_hooks: list[tuple[Any, Literal["pre", "forward"], Callable[..., Any]]] = [
        (layers[args.residual_layer], "forward", capture_hook(capture, "visual_residual")),
        (attention.o_proj, "pre", capture_pre_hook(capture, "attention_head_input")),
    ]
    capture_hooks.extend(
        (layers[layer_index].mlp, "forward", capture_hook(capture, f"mlp_{layer_index}"))
        for layer_index in args.mlp_layers
    )

    started = time.time()
    clean_score = runner.score(clean, tuple(capture_hooks))
    corrupted_score = runner.score(corrupted)
    clean_margin = clean_score.margin_nats
    corrupted_margin = corrupted_score.margin_nats
    clean_gap = clean_margin - corrupted_margin
    components: list[dict[str, Any]] = []

    def evaluate(
        name: str,
        module: Any,
        hook_kind: Literal["pre", "forward"],
        patch: Callable[..., Any],
        ablate: Callable[..., Any],
        scope: str,
    ) -> None:
        patched = runner.score(corrupted, ((module, hook_kind, patch),))
        ablated = runner.score(clean, ((module, hook_kind, ablate),))
        restoration = patched.margin_nats - corrupted_margin
        ablation_drop = clean_margin - ablated.margin_nats
        components.append(
            {
                "component": name,
                "scope": scope,
                "patched_corrupted_margin_nats": patched.margin_nats,
                "restoration_nats": restoration,
                "recovery_fraction": restoration / clean_gap if abs(clean_gap) > 1e-9 else None,
                "ablated_clean_margin_nats": ablated.margin_nats,
                "ablation_drop_nats": ablation_drop,
                "patched_sequence_log_prob": {
                    "correct": patched.candidate_log_prob_nats[0],
                    "distractor": patched.candidate_log_prob_nats[1],
                },
                "ablated_sequence_log_prob": {
                    "correct": ablated.candidate_log_prob_nats[0],
                    "distractor": ablated.candidate_log_prob_nats[1],
                },
            }
        )
        print(
            f"{name}: restoration={restoration:+.3f} nats; ablation_drop={ablation_drop:+.3f} nats",
            flush=True,
        )

    clean_image_positions = clean.image_positions[0]
    corrupted_image_positions = corrupted.image_positions[0]
    visual_source = capture["visual_residual"]
    evaluate(
        f"layer_{args.residual_layer}_visual_token_residuals",
        layers[args.residual_layer],
        "forward",
        residual_hook(corrupted_image_positions, visual_source),
        residual_hook(clean_image_positions, None, clean_image_positions),
        "all image-token positions",
    )
    attention_source = capture["attention_head_input"]
    evaluate(
        f"layer_{args.attention_layer}_head_{args.attention_head}_attention_output",
        attention.o_proj,
        "pre",
        attention_hook(
            corrupted, attention_source, head=args.attention_head, head_dim=int(attention.head_dim)
        ),
        attention_hook(clean, None, head=args.attention_head, head_dim=int(attention.head_dim)),
        "head slice before o_proj at coordinate prediction queries",
    )
    for layer_index in args.mlp_layers:
        evaluate(
            f"layer_{layer_index}_mlp_output",
            layers[layer_index].mlp,
            "forward",
            prediction_hook(corrupted, capture[f"mlp_{layer_index}"]),
            prediction_hook(clean, None),
            "MLP output at coordinate prediction queries",
        )
    evaluate(
        f"layer_{args.residual_layer}_target_patch_residuals",
        layers[args.residual_layer],
        "forward",
        residual_hook(corrupted.region_positions["target"], visual_source),
        residual_hook(clean.region_positions["target"], None, clean_image_positions),
        "image patches overlapping the original target tile",
    )
    evaluate(
        f"layer_{args.residual_layer}_distractor_patch_residuals",
        layers[args.residual_layer],
        "forward",
        residual_hook(corrupted.region_positions["distractor"], visual_source),
        residual_hook(clean.region_positions["distractor"], None, clean_image_positions),
        "matched image patches overlapping the adjacent distractor tile",
    )

    changed_bbox = ImageChops.difference(clean_image, corrupted_image).getbbox()
    target_patch_indices = [
        position - clean_image_positions[0] for position in clean.region_positions["target"]
    ]
    distractor_patch_indices = [
        position - clean_image_positions[0] for position in clean.region_positions["distractor"]
    ]
    result = {
        "schema_version": 2,
        "experiment": "matched_tile_swap_teacher_forced_activation_patching",
        "instruction": args.instruction,
        "model": engine.model_metadata(),
        "source_image": str(source_path),
        "corrupted_source_image": str(corrupted_source_path) if corrupted_source_path else None,
        "clean_image": str(clean_path),
        "corrupted_image": str(corrupted_path),
        "image_sha256": {
            "clean": hashlib.sha256(clean_path.read_bytes()).hexdigest(),
            "corrupted": hashlib.sha256(corrupted_path.read_bytes()).hexdigest(),
        },
        "corruption": {
            "operation": corruption_operation,
            "target_swap_box": list(args.target_swap_box) if corrupted_source_path is None else None,
            "distractor_swap_box": list(args.distractor_swap_box) if corrupted_source_path is None else None,
            "changed_bbox": list(changed_bbox) if changed_bbox else None,
            "changed_pixel_fraction": changed_pixel_fraction(clean_image, corrupted_image),
            "layout_preserved": corrupted_source_path is None,
            "pixel_method": "crop both source rectangles first, then paste each crop into the other box",
            "resized_or_resampled": False if corrupted_source_path is None else None,
        },
        "coordinates": {
            "correct": {"normalized": list(correct_coordinate), "pixel_box": list(args.target_box)},
            "distractor": {
                "normalized": list(distractor_coordinate),
                "pixel_box": list(args.distractor_box),
            },
        },
        "metric": {
            "name": "teacher_forced_coordinate_sequence_log_likelihood_ratio",
            "formula": "sum ln p(original-target-slot x/y tokens) - sum ln p(distractor-slot x/y tokens)",
            "units": "nats (natural-log probability ratio)",
            "score_reduction": "sum",
            "coordinate_tokens_per_sequence": len(clean.scored_token_ids[0]),
            "syntax_tokens_scored": False,
            "equal_scored_token_count_required_for_sum": True,
            "odds_ratio_interpretation": "exp(margin_nats) = target-sequence probability / distractor-sequence probability",
        },
        "patch_grid": {
            "rows": grid[0],
            "columns": grid[1],
            "image_token_count": len(clean_image_positions),
            "target_patch_count": len(clean.region_positions["target"]),
            "distractor_patch_count": len(clean.region_positions["distractor"]),
            "target_patch_indices": target_patch_indices,
            "distractor_patch_indices": distractor_patch_indices,
        },
        "intervention_protocol": {
            "patching": "replace the named corrupted activation with its clean-run value under the same candidate suffix",
            "visual_ablation": "replace all visual-token residuals with their image-token mean",
            "regional_ablation": "replace selected residuals with the mean of remaining image tokens",
            "head_and_mlp_ablation": "zero the named output at coordinate prediction queries",
            "one_component_per_forward_pass": True,
        },
        "baselines": {
            "clean_margin_nats": clean_margin,
            "corrupted_margin_nats": corrupted_margin,
            "clean_minus_corrupted_nats": clean_gap,
            "clean_target_over_distractor_probability_ratio": math.exp(clean_margin),
            "corrupted_target_over_distractor_probability_ratio": math.exp(corrupted_margin),
            "clean_sequence_log_prob": {
                "correct": clean_score.candidate_log_prob_nats[0],
                "distractor": clean_score.candidate_log_prob_nats[1],
            },
            "corrupted_sequence_log_prob": {
                "correct": corrupted_score.candidate_log_prob_nats[0],
                "distractor": corrupted_score.candidate_log_prob_nats[1],
            },
        },
        "interventions": components,
        "runtime_seconds": time.time() - started,
        "interpretation_guardrail": (
            "Recovery fraction is an additive fraction of the clean-minus-corrupted log-margin gap, not a "
            "fraction of probability mass. One matched request pair remains a case study, not a population claim."
        ),
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[3]
    result = argparse.ArgumentParser(prog="holo-causal-experiment")
    result.add_argument(
        "--image",
        type=Path,
        default=project_root / "data/screenspot-pro/images/powerpoint_windows_59.png",
    )
    result.add_argument(
        "--corrupted-image",
        type=Path,
        help="optional prebuilt corruption; if omitted, swap the two equal-sized boxes",
    )
    result.add_argument("--instruction", default='Create a "Psychedelic vibrant" presentation')
    result.add_argument("--target-box", type=parse_box, default=(1289, 563, 1489, 702))
    result.add_argument("--distractor-box", type=parse_box, default=(1069, 563, 1269, 702))
    result.add_argument("--target-swap-box", type=parse_box, default=(1289, 563, 1489, 735))
    result.add_argument("--distractor-swap-box", type=parse_box, default=(1069, 563, 1269, 735))
    result.add_argument("--residual-layer", type=int, default=19)
    result.add_argument("--attention-layer", type=int, default=19)
    result.add_argument("--attention-head", type=int, default=10)
    result.add_argument("--mlp-layers", type=int, nargs="+", default=(15, 19, 23))
    result.add_argument(
        "--output",
        type=Path,
        default=project_root / "artifacts/causal-intervention/powerpoint_windows_59_swap",
    )
    return result


def main(argv: list[str] | None = None) -> None:
    payload = run_experiment(parser().parse_args(argv))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
