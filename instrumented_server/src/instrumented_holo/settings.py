"""Runtime settings for the local model server."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_model_path() -> Path:
    return Path(__file__).resolve().parents[3] / "models" / "Holo-3.1-4B"


def _default_trace_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "traces"


@dataclass(frozen=True)
class Settings:
    """Configuration sourced from environment variables or explicit CLI options."""

    model_path: Path = field(default_factory=_default_model_path)
    trace_dir: Path = field(default_factory=_default_trace_dir)
    host: str = "127.0.0.1"
    port: int = 8000
    device: str = "auto"
    dtype: str = "auto"
    load_strategy: str = "stream"
    allow_dtype_conversion: bool = False
    eager_attention: bool = True
    default_max_tokens: int = 128
    image_min_pixels: int = 65_536
    image_max_pixels: int = 262_144

    @classmethod
    def from_environment(cls) -> Settings:
        return cls(
            model_path=Path(os.environ.get("HOLO_MODEL_PATH", _default_model_path())),
            trace_dir=Path(os.environ.get("HOLO_TRACE_DIR", _default_trace_dir())),
            host=os.environ.get("HOLO_SERVER_HOST", "127.0.0.1"),
            port=int(os.environ.get("HOLO_SERVER_PORT", "8000")),
            device=os.environ.get("HOLO_DEVICE", "auto"),
            dtype=os.environ.get("HOLO_DTYPE", "auto"),
            load_strategy=os.environ.get("HOLO_LOAD_STRATEGY", "stream"),
            allow_dtype_conversion=os.environ.get("HOLO_ALLOW_DTYPE_CONVERSION", "0") == "1",
            eager_attention=os.environ.get("HOLO_EAGER_ATTENTION", "1") != "0",
            default_max_tokens=int(os.environ.get("HOLO_DEFAULT_MAX_TOKENS", "128")),
            image_min_pixels=int(os.environ.get("HOLO_IMAGE_MIN_PIXELS", "65536")),
            image_max_pixels=int(os.environ.get("HOLO_IMAGE_MAX_PIXELS", "262144")),
        )
