#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required; install it on the remote host and rerun." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; this bootstrap expects an NVIDIA GPU host." >&2
  exit 2
fi
if [[ -z "${HOLO_MODEL_PATH:-}" ]]; then
  echo "Set HOLO_MODEL_PATH to the unpacked Holo-3.1-4B checkpoint." >&2
  exit 2
fi

export HOLO_DEVICE="${HOLO_DEVICE:-cuda}"
export HOLO_DTYPE="${HOLO_DTYPE:-bfloat16}"
export HOLO_TRACE_DIR="${HOLO_TRACE_DIR:-$repo_root/data/traces}"

uv sync --dev --frozen
uv run pytest
(
  cd instrumented_server
  uv sync --dev --frozen
  uv run pytest
  uv run holo-gpu-doctor --require-cuda --output ../data/manifests/remote-gpu-doctor.json
)

echo "Remote Holo environment is ready. Start the server from instrumented_server/."
