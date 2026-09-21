"""Preflight checks for running Holo experiments on a remote GPU host."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .settings import Settings


def gpu_report(settings: Settings) -> dict[str, Any]:
    import torch

    model_path = settings.model_path.expanduser().resolve()
    trace_dir = settings.trace_dir.expanduser().resolve()
    trace_parent = trace_dir if trace_dir.exists() else trace_dir.parent
    shards = sorted(model_path.glob("*.safetensors")) if model_path.is_dir() else []
    required = {
        "config.json": (model_path / "config.json").is_file(),
        "tokenizer_config.json": (model_path / "tokenizer_config.json").is_file(),
        "model.safetensors.index.json": (model_path / "model.safetensors.index.json").is_file(),
    }
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "total_memory_gib": properties.total_memory / 2**30,
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )
    usage = shutil.disk_usage(trace_parent)
    issues = []
    if not model_path.is_dir():
        issues.append(f"model directory does not exist: {model_path}")
    if not all(required.values()):
        issues.append("model directory is missing required checkpoint metadata")
    if not shards:
        issues.append("model directory contains no safetensor shards")
    if settings.device in {"auto", "cuda"} and not torch.cuda.is_available():
        issues.append("CUDA is not available to PyTorch")
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        issues.append("selected CUDA runtime does not report BF16 support")
    return {
        "schema_version": 1,
        "ok": not issues,
        "issues": issues,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "runtime": {
            "torch": version("torch"),
            "torchvision": version("torchvision"),
            "transformers": version("transformers"),
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "devices": cuda_devices,
        },
        "configuration": {
            "device": settings.device,
            "dtype": settings.dtype,
            "eager_attention": settings.eager_attention,
            "image_min_pixels": settings.image_min_pixels,
            "image_max_pixels": settings.image_max_pixels,
            "model_path": str(model_path),
            "trace_dir": str(trace_dir),
        },
        "checkpoint": {
            "required_files": required,
            "safetensor_shards": len(shards),
            "safetensor_bytes": sum(path.stat().st_size for path in shards),
        },
        "storage": {
            "trace_filesystem": str(trace_parent),
            "free_gib": usage.free / 2**30,
            "total_gib": usage.total / 2**30,
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="holo-gpu-doctor")
    result.add_argument("--require-cuda", action="store_true")
    result.add_argument("--output", type=Path)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    report = gpu_report(Settings.from_environment())
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
    print(payload, end="")
    if not report["ok"] or (args.require_cuda and not report["runtime"]["cuda_available"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
