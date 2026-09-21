#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$repo_root/benchmarks/delta_lens/qwen35_4b_vs_holo31_4b.json"

for required_name in HOLO_MODEL_PATH QWEN_MODEL_PATH DELTA_OUTPUT_DIR; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "Set $required_name before running the remote delta-lens experiment." >&2
    exit 2
  fi
done
if [[ ! -d "$HOLO_MODEL_PATH" ]]; then
  echo "Holo checkpoint directory is missing: $HOLO_MODEL_PATH" >&2
  exit 2
fi
if [[ ! -d "$QWEN_MODEL_PATH" ]]; then
  echo "Qwen checkpoint directory is missing: $QWEN_MODEL_PATH" >&2
  exit 2
fi
if [[ -e "$DELTA_OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing output: $DELTA_OUTPUT_DIR" >&2
  exit 2
fi
smoke_output="${DELTA_OUTPUT_DIR}-smoke"
if [[ -e "$smoke_output" ]]; then
  echo "Refusing to overwrite existing smoke output: $smoke_output" >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required on the remote host." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required on the remote host." >&2
  exit 2
fi

export HOLO_DEVICE=cuda
export HOLO_DTYPE=bfloat16
export HOLO_EAGER_ATTENTION=0
export HOLO_IMAGE_MIN_PIXELS=65536
export HOLO_IMAGE_MAX_PIXELS=16777216
export HOLO_TRACE_DIR="${HOLO_TRACE_DIR:-$(dirname "$DELTA_OUTPUT_DIR")/traces}"
export HOLO_PROCESSOR_PATH="$HOLO_MODEL_PATH"

mkdir -p "$(dirname "$DELTA_OUTPUT_DIR")" "$HOLO_TRACE_DIR" "$repo_root/data/manifests"
nvidia-smi

cd "$repo_root"
uv sync --dev --frozen
uv run pytest

cd "$repo_root/instrumented_server"
uv sync --dev --frozen
uv run pytest

HOLO_MODEL_PATH="$HOLO_MODEL_PATH" \
  uv run holo-gpu-doctor --require-cuda \
  --output "$repo_root/data/manifests/delta-lens-holo-gpu-doctor.json"
HOLO_MODEL_PATH="$QWEN_MODEL_PATH" \
  uv run holo-gpu-doctor --require-cuda \
  --output "$repo_root/data/manifests/delta-lens-qwen-gpu-doctor.json"

PYTHONPATH="$repo_root/instrumented_server/src" \
  uv run python "$repo_root/scripts/audit_delta_lens_inputs.py" "$manifest" \
  --processor "$HOLO_MODEL_PATH" \
  --output "$repo_root/data/manifests/delta-lens-input-audit.json"

uv run holo-delta-lens "$manifest" \
  --base-model "$QWEN_MODEL_PATH" \
  --tuned-model "$HOLO_MODEL_PATH" \
  --processor "$HOLO_MODEL_PATH" \
  --case-id powerpoint_windows_59 \
  --output "$smoke_output"

python3 - "$smoke_output/delta-lens.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
cases = result["comparison"]["cases"]
if len(cases) != 1 or cases[0]["id"] != "powerpoint_windows_59":
    raise SystemExit("delta-lens smoke result did not contain the expected aligned case")
print("Delta-lens two-checkpoint smoke passed.")
PY

uv run holo-delta-lens "$manifest" \
  --base-model "$QWEN_MODEL_PATH" \
  --tuned-model "$HOLO_MODEL_PATH" \
  --processor "$HOLO_MODEL_PATH" \
  --output "$DELTA_OUTPUT_DIR"

cd "$repo_root"
result_archive="${DELTA_OUTPUT_DIR}.tar.gz"
uv run holo-capture package-results "$DELTA_OUTPUT_DIR" \
  --trace-root "$HOLO_TRACE_DIR" \
  --output "$result_archive"
(
  cd "$(dirname "$result_archive")"
  sha256sum -c "$(basename "${result_archive}.sha256")"
)

echo "Delta-lens run complete."
echo "Results: $DELTA_OUTPUT_DIR"
echo "Archive: $result_archive"
echo "Checksum: ${result_archive}.sha256"
