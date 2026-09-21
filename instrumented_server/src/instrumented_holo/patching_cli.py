"""Manifest-driven activation patching for aligned multimodal request pairs."""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .activation_patching import (
    ActivationPatchingRunner,
    CandidateSequence,
    ImageRegion,
    coordinate_tool_candidate,
    png_data_url,
    swap_equal_tiles,
    validate_box,
)
from .interventions import IntervenableHolo, InterventionConfig
from .model import InstrumentedHolo, decode_data_image
from .settings import Settings


@dataclass(frozen=True)
class ManifestPlan:
    path: Path
    raw: dict[str, Any]
    clean_request: dict[str, Any]
    corrupted_request: dict[str, Any]
    candidates: tuple[CandidateSequence, CandidateSequence]
    regions: dict[str, ImageRegion]
    interventions: tuple[InterventionConfig, ...]


def load_manifest_plan(manifest_path: Path) -> ManifestPlan:
    """Parse and validate all model-independent parts of an experiment."""

    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("patch manifest must use schema_version 1")
    base = manifest_path.parent
    clean_request = _load_json_or_inline(manifest["clean_request"], base)
    if "corrupted_request" in manifest:
        corrupted_request = _load_json_or_inline(manifest["corrupted_request"], base)
    else:
        corrupted_request = _apply_corruption(clean_request, manifest["corruption"])
    candidates = _candidates(manifest["candidates"])
    regions = _regions(manifest.get("regions", {}))
    interventions = tuple(
        InterventionConfig.from_dict(item, index=index)
        for index, item in enumerate(manifest["interventions"])
    )
    if not interventions:
        raise ValueError("patch manifest must declare at least one intervention")
    for item in interventions:
        representation = item.representation
        if representation.unit == "image_region" and representation.region not in regions:
            raise ValueError(
                f"intervention {item.name!r} refers to undeclared region {representation.region!r}"
            )
    clean_images = [decode_data_image(_image_url(part)) for part in _image_parts(clean_request)]
    corrupted_images = [decode_data_image(_image_url(part)) for part in _image_parts(corrupted_request)]
    if len(clean_images) != len(corrupted_images):
        raise ValueError("clean and corrupted requests must contain the same number of images")
    if tuple(image.size for image in clean_images) != tuple(image.size for image in corrupted_images):
        raise ValueError("clean and corrupted request images must have identical dimensions")
    for name, region in regions.items():
        if not 0 <= region.image_index < len(clean_images):
            raise ValueError(f"region {name!r} refers to missing image {region.image_index}")
        validate_box(region.box, clean_images[region.image_index].size)
    return ManifestPlan(
        path=manifest_path,
        raw=manifest,
        clean_request=clean_request,
        corrupted_request=corrupted_request,
        candidates=candidates,
        regions=regions,
        interventions=interventions,
    )


def manifest_summary(plan: ManifestPlan) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "valid": True,
        "manifest": str(plan.path),
        "experiment": str(plan.raw.get("name", plan.path.stem)),
        "candidate_labels": [candidate.label for candidate in plan.candidates],
        "regions": sorted(plan.regions),
        "interventions": [item.to_dict() for item in plan.interventions],
    }


def _load_json_or_inline(value: Any, base: Path) -> dict[str, Any]:
    if isinstance(value, str):
        path = Path(value).expanduser()
        path = path.resolve() if path.is_absolute() else (base / path).resolve()
        payload = json.loads(path.read_text())
    elif isinstance(value, dict):
        payload = copy.deepcopy(value)
    else:
        raise ValueError("request must be an inline object or a JSON path")
    for message in payload.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            if part.get("type") != "image_path":
                continue
            image_path = Path(part["path"]).expanduser()
            image_path = image_path.resolve() if image_path.is_absolute() else (base / image_path).resolve()
            from PIL import Image

            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            content[index] = {"type": "image_url", "image_url": {"url": png_data_url(image)}}
    return payload


def _image_parts(request: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for message in request.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            result.extend(part for part in content if part.get("type") == "image_url")
    return result


def _image_url(part: Mapping[str, Any]) -> str:
    value = part["image_url"]
    return value if isinstance(value, str) else str(value["url"])


def _apply_corruption(request: dict[str, Any], specification: Mapping[str, Any]) -> dict[str, Any]:
    if specification.get("type") != "swap_equal_tiles":
        raise ValueError(f"unsupported corruption type: {specification.get('type')!r}")
    result = copy.deepcopy(request)
    image_index = int(specification.get("image_index", 0))
    parts = _image_parts(result)
    if not 0 <= image_index < len(parts):
        raise ValueError(f"corruption refers to missing image {image_index}")
    image = decode_data_image(_image_url(parts[image_index]))
    first = tuple(int(value) for value in specification["first_box"])
    second = tuple(int(value) for value in specification["second_box"])
    swapped = swap_equal_tiles(image, first, second)  # type: ignore[arg-type]
    parts[image_index]["image_url"] = {"url": png_data_url(swapped)}
    return result


def _candidates(raw: list[dict[str, Any]]) -> tuple[CandidateSequence, CandidateSequence]:
    if len(raw) != 2:
        raise ValueError("activation patching requires exactly two scored candidates")
    result = []
    for candidate in raw:
        label = str(candidate["label"])
        if "coordinate" in candidate:
            coordinate = tuple(int(value) for value in candidate["coordinate"])
            if len(coordinate) != 2:
                raise ValueError(f"candidate {label!r} coordinate must contain x and y")
            if any(value < 0 or value > 1000 for value in coordinate):
                raise ValueError(f"candidate {label!r} coordinate must stay within 0..1000")
            result.append(coordinate_tool_candidate(label, coordinate))  # type: ignore[arg-type]
        else:
            spans = tuple(tuple(int(value) for value in span) for span in candidate["scored_spans"])
            text = str(candidate["text"])
            if not spans:
                raise ValueError(f"candidate {label!r} must contain at least one scored span")
            if any(len(span) != 2 or span[0] < 0 or span[1] <= span[0] or span[1] > len(text) for span in spans):
                raise ValueError(f"candidate {label!r} has an invalid scored span")
            result.append(CandidateSequence(label, text, spans))  # type: ignore[arg-type]
    return result[0], result[1]


def _regions(raw: Mapping[str, Any]) -> dict[str, ImageRegion]:
    return {
        name: ImageRegion(
            image_index=int(value["image_index"]),
            box=tuple(int(item) for item in value["box"]),  # type: ignore[arg-type]
            limit=int(value["limit"]) if value.get("limit") is not None else None,
        )
        for name, value in raw.items()
    }


def run_manifest(
    manifest_path: Path,
    *,
    output_dir: Path,
    save_activations: bool = False,
) -> dict[str, Any]:
    plan = load_manifest_plan(manifest_path)
    manifest = plan.raw
    clean_request = plan.clean_request
    corrupted_request = plan.corrupted_request
    candidates = plan.candidates
    region_specs = plan.regions
    interventions = plan.interventions

    engine = InstrumentedHolo(Settings.from_environment())
    engine.load()
    runner = ActivationPatchingRunner(engine, reduction=manifest.get("score_reduction", "sum"))
    clean = runner.prepare(
        name="clean",
        messages=clean_request["messages"],
        tools=clean_request.get("tools", []),
        candidates=candidates,
        regions=region_specs,
        chat_template_kwargs=clean_request.get("chat_template_kwargs"),
    )
    corrupted = runner.prepare(
        name="corrupted",
        messages=corrupted_request["messages"],
        tools=corrupted_request.get("tools", []),
        candidates=candidates,
        regions=region_specs,
        chat_template_kwargs=corrupted_request.get("chat_template_kwargs"),
    )
    started = time.time()
    intervenable = IntervenableHolo(runner, interventions, regions=region_specs)
    output = intervenable.compare(clean, corrupted)
    rows = [effect.to_dict() for effect in output.effects]
    for effect in output.effects:
        print(
            json.dumps(
                {
                    "name": effect.config.name,
                    "restoration_nats": effect.restoration_nats,
                    "ablation_drop_nats": effect.ablation_drop_nats,
                }
            ),
            flush=True,
        )

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_activations:
        np.savez_compressed(
            output_dir / "clean_activations.npz",
            **{
                key.replace(":", "_"): value.detach().to(dtype=engine.torch.float16).cpu().numpy()
                for key, value in intervenable.source_activations.items()
            },
        )
    result = {
        "schema_version": 1,
        "experiment": str(manifest.get("name", manifest_path.stem)),
        "manifest": str(plan.path),
        "model": engine.model_metadata(),
        "score_reduction": runner.reduction,
        "metric": {
            "name": "teacher_forced_candidate_log_probability_margin",
            "formula": "candidate_0_log_prob_nats - candidate_1_log_prob_nats",
            "units": "nats",
            "probability_ratio": "exp(margin_nats)",
        },
        "candidates": [candidate.label for candidate in candidates],
        "image_grids": [list(grid) for grid in clean.image_grids],
        "region_positions": {key: list(value) for key, value in clean.region_positions.items()},
        "baselines": {
            "clean_margin_nats": output.clean.margin_nats,
            "corrupted_margin_nats": output.corrupted.margin_nats,
            "clean_minus_corrupted_nats": output.clean_corrupted_gap_nats,
            "clean_candidate_log_prob_nats": list(output.clean.candidate_log_prob_nats),
            "corrupted_candidate_log_prob_nats": list(output.corrupted.candidate_log_prob_nats),
        },
        "interventions": rows,
        "clean_activations": str(output_dir / "clean_activations.npz") if save_activations else None,
        "runtime_seconds": time.time() - started,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output_dir / "manifest.snapshot.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="holo-activation-patch")
    result.add_argument("manifest", type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--save-activations", action="store_true")
    result.add_argument(
        "--validate-only",
        action="store_true",
        help="validate requests, corruption, candidates, regions, and intervention configs without loading Holo",
    )
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.validate_only:
        print(json.dumps(manifest_summary(load_manifest_plan(args.manifest)), indent=2))
        return
    if args.output is None:
        parser().error("--output is required unless --validate-only is used")
    print(json.dumps(run_manifest(args.manifest, output_dir=args.output, save_activations=args.save_activations), indent=2))


if __name__ == "__main__":
    main()
