import hashlib
import json
import tarfile
from pathlib import Path

from scripts.package_delta_lens_deployment import BUNDLE_ROOT, package, verify


def test_deployment_bundle_is_minimal_and_checksummed(tmp_path: Path) -> None:
    archive = tmp_path / "deployment.tar.gz"
    result = package(archive)

    assert hashlib.sha256(archive.read_bytes()).hexdigest() == result["sha256"]
    assert archive.with_suffix(".gz.sha256").read_text().startswith(result["sha256"])
    with tarfile.open(archive, "r:gz") as bundle:
        names = set(bundle.getnames())
        manifest = json.load(bundle.extractfile(f"{BUNDLE_ROOT}/deployment-manifest.json"))

    assert f"{BUNDLE_ROOT}/scripts/run_delta_lens_remote.sh" in names
    assert f"{BUNDLE_ROOT}/benchmarks/delta_lens/qwen35_4b_vs_holo31_4b.json" in names
    assert f"{BUNDLE_ROOT}/data/manifests/delta-lens-input-audit.json" in names
    assert not any("/models/" in name or "/.git/" in name or "/data/traces/" in name for name in names)
    assert manifest["entrypoint"] == "scripts/run_delta_lens_remote.sh"
    assert manifest["entry_count"] == result["entry_count"]
    assert verify(archive)["sidecar_verified"] is True
