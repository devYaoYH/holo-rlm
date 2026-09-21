"""Versioned deterministic generator for diverse synthetic booking tasks."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

GENERATOR_VERSION = "booking-synth-v2"
LARGE_UI_DIAGNOSTIC_VERSION = "booking-synth-v2-large-ui-diagnostic"
MANIFEST_SCHEMA_VERSION = 1
SUPPORTED_SPLITS = ("train", "dev", "test")

_NAMES = {
    "train": [
        "Maple Crossing Inn",
        "Cedar Market Hotel",
        "Willow Court Rooms",
        "Birch Station Lodge",
        "Pine Garden Suites",
        "Oak Bridge House",
        "Elm Terrace Hotel",
        "Aspen Corner Inn",
        "Alder Square Rooms",
        "Spruce Wharf Lodge",
        "Chestnut Lane Hotel",
        "Rowan Park Suites",
    ],
    "dev": [
        "Quartz Arcade Hotel",
        "Saffron Gate Rooms",
        "Topaz Crescent Inn",
        "Violet Foundry Lodge",
        "Umber Gallery Suites",
        "Silver Atrium Hotel",
        "Opal Exchange Rooms",
        "Crimson Pavilion Inn",
        "Ivory Workshop Hotel",
        "Azure Promenade Lodge",
        "Ochre Assembly Suites",
        "Pearl District Rooms",
    ],
    "test": [
        "Beacon Reach Hotel",
        "Cobalt Basin Rooms",
        "Ember Quay Inn",
        "Fjord Compass Lodge",
        "Grove Meridian Suites",
        "Haven Orbit Hotel",
        "Indigo Current Rooms",
        "Juniper Signal Inn",
        "Aurora Ledger Hotel",
        "Delta Lookout Lodge",
        "Kestrel Passage Suites",
        "Lumen Harbor Rooms",
    ],
}

_DESCRIPTIONS = {
    "train": ("beside the market", "near the garden", "by the station", "along the bridge"),
    "dev": ("inside the arts quarter", "near the arcade", "beside the gallery", "off the crescent"),
    "test": ("near the tidal basin", "beside the lookout", "along the passage", "by the signal tower"),
}

_VIEWPORTS = ((1024, 720), (1152, 768), (1280, 800), (1366, 768), (1440, 900))
_CURRENCIES = ("$", "€", "£")
_GAP_BANDS = ((1, 3), (8, 18), (35, 70))


def _axis_seed(split: str, item_index: int, axis: str) -> int:
    payload = f"{GENERATOR_VERSION}:{split}:{item_index}:{axis}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _rng(split: str, item_index: int, axis: str) -> random.Random:
    return random.Random(_axis_seed(split, item_index, axis))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _colour(rng: random.Random) -> str:
    return f"#{rng.randint(45, 150):02x}{rng.randint(70, 165):02x}{rng.randint(85, 185):02x}"


def default_layout(viewport: dict[str, int] | None = None) -> dict[str, Any]:
    """Return the legacy layout expressed as explicit geometry."""

    viewport = viewport or {"width": 1280, "height": 800}
    return {
        "header_height": 78,
        "content_left": 160,
        "first_card_y": 250,
        "card_width": min(960, viewport["width"] - 200),
        "card_height": 244,
        "card_gap": 18,
        "image_width": 270,
        "name_x_offset": 305,
        "name_y_offset": 35,
        "description_y_offset": 70,
        "rating_x_offset": 788,
        "rating_y_offset": 35,
        "price_x_offset": 868,
        "price_y_offset": 105,
        "button_x_offset": 870,
        "button_width": 105,
        "button_y_offset": 160,
        "button_height": 65,
    }


def _layout(split: str, item_index: int) -> tuple[dict[str, int], dict[str, Any]]:
    rng = _rng(split, item_index, "layout")
    width, height = _VIEWPORTS[(item_index + rng.randrange(len(_VIEWPORTS))) % len(_VIEWPORTS)]
    viewport = {"width": width, "height": height}
    content_left = rng.choice((72, 96, 120, 144, 168))
    card_width = width - 2 * content_left
    card_height = rng.choice((210, 228, 246, 268, 286))
    card_gap = rng.choice((12, 18, 24, 30))
    image_width = rng.choice((190, 225, 260, 300))
    first_card_y = rng.choice((205, 230, 255, 285))
    price_fraction = rng.choice((0.62, 0.72, 0.82, 0.88))
    button_fraction = rng.choice((0.63, 0.75, 0.84))
    button_width = rng.choice((96, 112, 128))
    button_x_offset = min(int(card_width * button_fraction), card_width - button_width - 12)
    return viewport, {
        "header_height": rng.choice((64, 72, 78, 88)),
        "content_left": content_left,
        "first_card_y": first_card_y,
        "card_width": card_width,
        "card_height": card_height,
        "card_gap": card_gap,
        "image_width": min(image_width, card_width // 3),
        "name_x_offset": min(image_width + 28, card_width // 2),
        "name_y_offset": rng.choice((24, 32, 40)),
        "description_y_offset": rng.choice((58, 70, 82)),
        "rating_x_offset": min(int(card_width * 0.73), card_width - 125),
        "rating_y_offset": rng.choice((24, 34, 44)),
        "price_x_offset": min(int(card_width * price_fraction), card_width - 72),
        "price_y_offset": rng.choice((82, 104, 126, 146)),
        "button_x_offset": button_x_offset,
        "button_width": button_width,
        "button_y_offset": card_height - rng.choice((76, 68, 60)),
        "button_height": rng.choice((42, 50, 58)),
    }


def generate_config(item_index: int, split: str = "test") -> dict[str, Any]:
    """Generate one deterministic scenario with independently seeded axes."""

    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    if item_index < 0:
        raise ValueError("item_index must be non-negative")

    count = 6 + item_index % 5
    content_rng = _rng(split, item_index, "content")
    order_rng = _rng(split, item_index, "order")
    price_rng = _rng(split, item_index, "prices")
    visual_rng = _rng(split, item_index, "visuals")
    viewport, layout = _layout(split, item_index)

    names = content_rng.sample(_NAMES[split], count)
    order_rng.shuffle(names)
    cheapest_placement = ("top", "middle", "bottom")[item_index % 3]
    placement_offset = (item_index // 3) % 2
    if cheapest_placement == "top":
        cheapest_index = placement_offset
    elif cheapest_placement == "middle":
        cheapest_index = (count - 1) // 2 + placement_offset
    else:
        cheapest_index = count - 1 - placement_offset
    currency = _CURRENCIES[(item_index // 2 + price_rng.randrange(len(_CURRENCIES))) % len(_CURRENCIES)]
    cheapest_price = price_rng.randint(79, 219)
    gap_band_index = (item_index // 3 + item_index % 3) % len(_GAP_BANDS)
    gap_low, gap_high = _GAP_BANDS[gap_band_index]
    runner_up_gap = price_rng.randint(gap_low, gap_high)
    other_prices_list = [cheapest_price + runner_up_gap]
    seen_prices = set(other_prices_list)
    while len(other_prices_list) < count - 1:
        candidate = cheapest_price + price_rng.randint(runner_up_gap + 1, runner_up_gap + 190)
        if candidate not in seen_prices:
            seen_prices.add(candidate)
            other_prices_list.append(candidate)
    price_rng.shuffle(other_prices_list)

    hotels = []
    other_index = 0
    descriptions = _DESCRIPTIONS[split]
    for index, name in enumerate(names):
        price = cheapest_price if index == cheapest_index else other_prices_list[other_index]
        if index != cheapest_index:
            other_index += 1
        hotel_id = f"{split}-{item_index:04d}-{_slug(name)}"
        hotels.append(
            {
                "id": hotel_id,
                "name": name,
                "price": price,
                "currency": currency,
                "description": f"Synthetic rooms {content_rng.choice(descriptions)}.",
                "colors": [_colour(visual_rng), _colour(visual_rng)],
                "rating": f"{visual_rng.randint(72, 96) / 10:.1f}",
                "initials": "".join(word[0] for word in name.split()[:2]),
                "index": index,
            }
        )

    stride = layout["card_height"] + layout["card_gap"]
    content_bottom = layout["first_card_y"] + (count - 1) * stride + layout["card_height"]
    max_scroll = max(0, content_bottom - viewport["height"] + layout["header_height"] + 24)
    cheapest = hotels[cheapest_index]
    return {
        "fixture_version": GENERATOR_VERSION,
        "generator_version": GENERATOR_VERSION,
        "dataset_split": split,
        "item_id": f"{split}-{item_index:04d}",
        "item_index": item_index,
        "seed": item_index,
        "variant": 0,
        "axis_seeds": {
            axis: f"{_axis_seed(split, item_index, axis):016x}"
            for axis in ("content", "order", "prices", "visuals", "layout")
        },
        "viewport": viewport,
        "layout": layout,
        "max_scroll": max_scroll,
        "currency": currency,
        "cheapest_id": cheapest["id"],
        "cheapest_index": cheapest_index,
        "cheapest_placement": cheapest_placement,
        "runner_up_gap": runner_up_gap,
        "target_id": cheapest["id"],
        "target_name": cheapest["name"],
        "expected_destination": f"/details/{cheapest['id']}",
        "initial_scroll_delta": min(500, max_scroll),
        "hotels": hotels,
    }


def large_ui_diagnostic(config: dict[str, Any]) -> dict[str, Any]:
    """Return a non-frozen diagnostic variant with readable text at the vision-token budget.

    The source geometry and price placement remain unchanged. Only typography grows,
    so a shifted-layout frozen item can be compared without pretending the result is
    part of the immutable v2 benchmark.
    """

    result = deepcopy(config)
    result["fixture_version"] = LARGE_UI_DIAGNOSTIC_VERSION
    result["generator_version"] = LARGE_UI_DIAGNOSTIC_VERSION
    result["dataset_split"] = f"{config.get('dataset_split', 'unknown')}-diagnostic"
    result["item_id"] = f"{config.get('item_id', 'scenario')}-large-ui"
    result["layout"].update(
        {
            "font_size": 15,
            "bold_font_size": 16,
            "price_font_size": 19,
            "button_font_size": 16,
            "price_y_offset": min(
                result["layout"]["price_y_offset"],
                result["layout"]["button_y_offset"] - 28,
            ),
        }
    )
    return result


def canonical_config(config: dict[str, Any]) -> bytes:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def config_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_config(config)).hexdigest()


def summarize_configs(configs: list[dict[str, Any]]) -> dict[str, Any]:
    def counts(values: list[Any]) -> dict[str, int]:
        return {str(key): value for key, value in sorted(Counter(values).items(), key=lambda item: str(item[0]))}

    margin_bands = [
        "close" if config["runner_up_gap"] <= 3 else "medium" if config["runner_up_gap"] <= 18 else "wide"
        for config in configs
    ]
    return {
        "result_counts": counts([len(config["hotels"]) for config in configs]),
        "cheapest_placements": counts([config["cheapest_placement"] for config in configs]),
        "cheapest_absolute_rows": sorted({config["cheapest_index"] for config in configs}),
        "runner_up_margin_bands": counts(margin_bands),
        "currencies": counts([config["currency"] for config in configs]),
        "viewports": counts(
            [f"{config['viewport']['width']}x{config['viewport']['height']}" for config in configs]
        ),
        "unique_cheapest_names": len(
            {
                min(config["hotels"], key=lambda hotel: (hotel["price"], hotel["id"]))["name"]
                for config in configs
            }
        ),
        "unique_orderings": len({tuple(hotel["name"] for hotel in config["hotels"]) for config in configs}),
        "unique_price_vectors": len({tuple(hotel["price"] for hotel in config["hotels"]) for config in configs}),
        "unique_price_columns": len({config["layout"]["price_x_offset"] for config in configs}),
        "card_heights": sorted({config["layout"]["card_height"] for config in configs}),
        "price_y_offsets": sorted({config["layout"]["price_y_offset"] for config in configs}),
    }


def build_manifest(*, split: str = "test", count: int = 120) -> dict[str, Any]:
    if count < 1:
        raise ValueError("count must be positive")
    items = []
    for item_index in range(count):
        config = generate_config(item_index, split)
        items.append(
            {
                "item_id": config["item_id"],
                "config_sha256": config_sha256(config),
                "config": config,
            }
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "split": split,
        "count": count,
        "diversity": summarize_configs([item["config"] for item in items]),
        "items": items,
    }


def write_manifest(path: Path, *, split: str = "test", count: int = 120) -> Path:
    manifest = build_manifest(split=split, count=count)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return path


def load_manifest(path: Path, *, verify: bool = True) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema_version')}")
    if manifest.get("generator_version") != GENERATOR_VERSION:
        raise ValueError(f"unsupported generator version: {manifest.get('generator_version')}")
    items = manifest.get("items")
    if not isinstance(items, list) or manifest.get("count") != len(items):
        raise ValueError("manifest count does not match items")
    if verify:
        for item in items:
            digest = config_sha256(item["config"])
            if digest != item.get("config_sha256"):
                raise ValueError(f"config hash mismatch for {item.get('item_id')}")
    return manifest


def manifest_item(manifest: dict[str, Any], item_id: str) -> dict[str, Any]:
    try:
        return next(item["config"] for item in manifest["items"] if item["item_id"] == item_id)
    except StopIteration as exc:
        raise ValueError(f"unknown manifest item: {item_id}") from exc
