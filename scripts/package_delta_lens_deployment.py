"""Create a minimal, checksummed deployment bundle for the remote delta-lens run."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "deployment" / "holo-delta-lens-vastai.tar.gz"
BUNDLE_ROOT = "holo-delta-lens-deployment"

INCLUDES = (
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("README.md"),
    Path("src"),
    Path("apps"),
    Path("tests"),
    Path("benchmarks"),
    Path("docs"),
    Path("instrumented_server/pyproject.toml"),
    Path("instrumented_server/uv.lock"),
    Path("instrumented_server/README.md"),
    Path("instrumented_server/src"),
    Path("instrumented_server/tests"),
    Path("scripts/bootstrap_remote_gpu.sh"),
    Path("scripts/build_delta_lens_manifest.py"),
    Path("scripts/audit_delta_lens_inputs.py"),
    Path("scripts/package_delta_lens_deployment.py"),
    Path("scripts/run_delta_lens_remote.sh"),
    Path("data/manifests/delta-lens-input-audit.json"),
    Path("data/screenspot-pro/annotations/powerpoint_windows.json"),
    Path("data/screenspot-pro/annotations/photoshop_windows.json"),
    Path("data/screenspot-pro/annotations/vscode_macos.json"),
    Path("data/screenspot-pro/images/powerpoint_windows_59.png"),
    Path("data/screenspot-pro/images/powerpoint_windows_48.png"),
    Path("data/screenspot-pro/images/photoshop_windows_1.png"),
    Path("data/screenspot-pro/images/photoshop_windows_9.png"),
    Path("data/screenspot-pro/images/vscode_macos_3.png"),
    Path("data/trajectories/v0/traj-20260921T080837Z-2a0fda4eb0/manifest.json"),
    Path("data/trajectories/v0/traj-20260921T080837Z-2a0fda4eb0/frames/0000-model-input.png"),
    Path("data/trajectories/v0/traj-20260921T080837Z-2a0fda4eb0/frames/0001-model-input.png"),
    Path("data/trajectories/v0/traj-20260921T080837Z-2a0fda4eb0/frames/0002-model-input.png"),
)

EXCLUDED_PARTS = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".DS_Store"}


def _files() -> tuple[Path, ...]:
    result: set[Path] = set()
    for relative in INCLUDES:
        source = PROJECT_ROOT / relative
        if not source.exists():
            raise FileNotFoundError(source)
        candidates = [source] if source.is_file() else source.rglob("*")
        for candidate in candidates:
            if not candidate.is_file() or candidate.is_symlink():
                continue
            project_relative = candidate.relative_to(PROJECT_ROOT)
            if any(part in EXCLUDED_PARTS for part in project_relative.parts):
                continue
            result.add(project_relative)
    return tuple(sorted(result))


def _git_value(*args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def package(output: Path) -> dict:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = _files()
    entries = []
    for relative in files:
        source = PROJECT_ROOT / relative
        payload = source.read_bytes()
        entries.append(
            {
                "path": relative.as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    status = _git_value("status", "--short") or ""
    diff = subprocess.run(
        ["git", "diff", "--binary", "--", *[path.as_posix() for path in files]],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    ).stdout
    manifest = {
        "schema_version": 1,
        "bundle": "holo-qwen-delta-lens-deployment",
        "source_commit": _git_value("rev-parse", "HEAD"),
        "source_dirty": bool(status),
        "source_status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "entry_count": len(entries),
        "entries": entries,
        "excluded": ["models", "credentials", "traces", "prior result archives", "git metadata"],
        "entrypoint": "scripts/run_delta_lens_remote.sh",
    }
    with tarfile.open(output, "w:gz") as archive:
        for relative in files:
            archive.add(PROJECT_ROOT / relative, arcname=f"{BUNDLE_ROOT}/{relative.as_posix()}")
        encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo(f"{BUNDLE_ROOT}/deployment-manifest.json")
        info.size = len(encoded)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(encoded))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {output.name}\n")
    return {
        "archive": str(output),
        "sha256": digest,
        "sha256_file": str(sidecar),
        "bytes": output.stat().st_size,
        **{key: manifest[key] for key in ("source_commit", "source_dirty", "entry_count", "entrypoint")},
    }


def verify(archive_path: Path) -> dict:
    archive_path = archive_path.expanduser().resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        manifest_name = f"{BUNDLE_ROOT}/deployment-manifest.json"
        if manifest_name not in members:
            raise ValueError("deployment archive has no manifest")
        manifest_file = archive.extractfile(members[manifest_name])
        if manifest_file is None:
            raise ValueError("deployment manifest is not a regular file")
        manifest = json.load(manifest_file)
        if manifest["entry_count"] != len(manifest["entries"]):
            raise ValueError("deployment manifest entry count is inconsistent")
        for entry in manifest["entries"]:
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe deployment path: {relative}")
            name = f"{BUNDLE_ROOT}/{relative.as_posix()}"
            member = members.get(name)
            if member is None or not member.isfile():
                raise ValueError(f"deployment entry is missing: {relative}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"deployment entry is unreadable: {relative}")
            payload = extracted.read()
            if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError(f"deployment entry failed hash verification: {relative}")
        unexpected = [
            name
            for name in members
            if name != manifest_name and not any(
                name == f"{BUNDLE_ROOT}/{entry['path']}" for entry in manifest["entries"]
            )
        ]
        if unexpected:
            raise ValueError(f"deployment archive has unexpected entries: {unexpected[:3]}")
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    sidecar = archive_path.with_suffix(archive_path.suffix + ".sha256")
    if sidecar.is_file() and sidecar.read_text().split()[0] != digest:
        raise ValueError("deployment archive sidecar checksum does not match")
    return {
        "archive": str(archive_path),
        "sha256": digest,
        "sidecar_verified": sidecar.is_file(),
        "entry_count": manifest["entry_count"],
        "source_commit": manifest["source_commit"],
        "source_dirty": manifest["source_dirty"],
        "entrypoint": manifest["entrypoint"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    result = verify(args.verify) if args.verify else package(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
