"""Read-only runtime/backend diagnostics and generated manifest."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


def holo_diagnostics() -> dict[str, Any]:
    executable = shutil.which("holo")
    if executable is None:
        return {"available": False, "executable": None, "runtime_version": None, "doctor_ok": False}
    result = subprocess.run([executable, "doctor"], capture_output=True, text=True, timeout=30, check=False)
    output = result.stdout + result.stderr
    version_match = re.search(r"managed install \(v([^\)]+)\)", output)
    return {
        "available": True,
        "executable": executable,
        "runtime_version": version_match.group(1) if version_match else None,
        "doctor_ok": result.returncode == 0,
        "login_configured": "signed in as" in output or "HAI_API_KEY set" in output,
        "permissions": {
            "screen_recording": "manual-verification-required",
            "accessibility": "manual-verification-required",
        },
    }


def endpoint_diagnostics(base_url: str, model_id: str) -> dict[str, Any]:
    if not _is_local_url(base_url):
        raise ValueError("local backend URL must use localhost/127.0.0.1/[::1]")
    result: dict[str, Any] = {"base_url": base_url, "endpoint_type": "openai-compatible-local", "reachable": False}
    try:
        with httpx.Client(timeout=3, trust_env=False) as client:
            response = client.get(f"{base_url.rstrip('/')}/models")
            response.raise_for_status()
            models = response.json().get("data", [])
            selected = next((model for model in models if model.get("id") == model_id), None)
        result.update(
            {
                "reachable": True,
                "advertised_models": [model.get("id") for model in models],
                "model_available": any(model.get("id") == model_id for model in models),
                "model_revision": selected.get("revision") if selected else None,
                "structured_actions": (selected or {}).get("capabilities", {}).get(
                    "structured_output", "schema-validated-json-fallback"
                ),
                "native_function_calling": (selected or {}).get("capabilities", {}).get("function_calling"),
                "activation_tracing": (selected or {}).get("capabilities", {}).get("activation_tracing"),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_preflight(backend: str, *, base_url: str, model_id: str, output_root: Path) -> tuple[dict[str, Any], Path]:
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "backend": backend,
        "model_id": model_id if backend == "local" else ("hosted-holo" if backend == "hosted" else "deterministic-reference-policy"),
        "model_revision": os.environ.get("HOLO_MODEL_REVISION") if backend == "local" else None,
        "processor_revision": os.environ.get("HOLO_PROCESSOR_REVISION") if backend == "local" else None,
        "holo": holo_diagnostics(),
        "endpoint": endpoint_diagnostics(base_url, model_id) if backend == "local" else {"endpoint_type": backend},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"preflight-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{backend}.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest, path


def smoke_backend(backend: str, *, base_url: str, model_id: str) -> dict[str, Any]:
    if backend == "scripted":
        return {"ok": True, "backend": backend, "action": {"action": "finish", "summary": "smoke"}}
    if backend == "hosted":
        diagnostics = holo_diagnostics()
        return {"ok": diagnostics.get("doctor_ok") and diagnostics.get("login_configured"), "backend": backend, "holo": diagnostics}
    diagnostics = endpoint_diagnostics(base_url, model_id)
    if not diagnostics.get("reachable"):
        return {"ok": False, "backend": backend, "endpoint": diagnostics}
    with httpx.Client(timeout=300, trust_env=False) as client:
        response = client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json={"model": model_id, "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 2, "temperature": 0},
        )
        response.raise_for_status()
        payload = response.json()
    return {"ok": bool(payload.get("choices")), "backend": backend, "model": payload.get("model")}


def _is_local_url(url: str) -> bool:
    from urllib.parse import urlparse

    return urlparse(url).hostname in {"localhost", "127.0.0.1", "::1"}
