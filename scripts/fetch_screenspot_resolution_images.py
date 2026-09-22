#!/usr/bin/env python3
"""Download only the public ScreenSpot-Pro images referenced by an ablation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote

import httpx
from PIL import Image

DEFAULT_BASE_URL = (
    "https://huggingface.co/datasets/yyyang/UI-Grounding-Benchmarks/resolve/main/"
    "ScreenSpot-Pro/images"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("manifest", type=Path)
    result.add_argument("--output", type=Path, default=Path("data/screenspot-pro/resolution-ablation-images"))
    result.add_argument("--base-url", default=DEFAULT_BASE_URL)
    return result


def main() -> None:
    args = parser().parse_args()
    manifest = json.loads(args.manifest.read_text())
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for case in manifest["cases"]:
            relative = Path(case["img_filename"])
            destination = args.output / relative
            if not destination.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                url = f"{args.base_url.rstrip('/')}/{quote(relative.as_posix(), safe='/')}?download=true"
                response = client.get(url)
                response.raise_for_status()
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                temporary.write_bytes(response.content)
                temporary.replace(destination)
            with Image.open(destination) as image:
                if list(image.size) != case["image_size"]:
                    raise ValueError(f"dimension mismatch for {case['id']}: {image.size} != {case['image_size']}")
            print(destination)


if __name__ == "__main__":
    main()
