import hashlib
import json
import shutil
from pathlib import Path

import pytest

from capture.validator import ValidationError, validate_bundle


def _copy(valid_bundle: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "bundle"
    shutil.copytree(valid_bundle, destination)
    return destination


def _rehash(bundle: Path, relative: str) -> None:
    path = bundle / relative
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data = path.read_bytes()
    manifest["files"][relative] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_rejects_mismatched_hash(valid_bundle: Path, tmp_path: Path) -> None:
    bundle = _copy(valid_bundle, tmp_path)
    path = bundle / "frames/0000-model-input.png"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValidationError, match="hash mismatch"):
        validate_bundle(bundle)


def test_rejects_missing_screenshot(valid_bundle: Path, tmp_path: Path) -> None:
    bundle = _copy(valid_bundle, tmp_path)
    (bundle / "frames/0000-model-input.png").unlink()
    with pytest.raises(ValidationError, match="missing or unsafe artifact"):
        validate_bundle(bundle)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("injected", "api_key=abcdefghijklmnop", "API key"),
        ("injected", "Authorization: Bearer abcdefghijklmnop", "authorization header"),
        ("injected", "Cookie: session=abcdefghijklmnop", "cookie"),
        ("injected", "https://example.com/checkout", "non-local URL"),
    ],
)
def test_rejects_secrets_cookies_and_external_urls(
    valid_bundle: Path, tmp_path: Path, field: str, value: str, message: str
) -> None:
    bundle = _copy(valid_bundle, tmp_path)
    annotation_path = bundle / "annotations.json"
    annotations = json.loads(annotation_path.read_text())
    annotations[field] = value
    annotation_path.write_text(json.dumps(annotations, indent=2, sort_keys=True) + "\n")
    _rehash(bundle, "annotations.json")
    with pytest.raises(ValidationError, match=message):
        validate_bundle(bundle)
