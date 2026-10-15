#!/usr/bin/env python3
"""Capture a reproducible governance baseline from an exact Git revision."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def capture(*, revision: str, version: str, manifest_output: Path, inventory_output: Path) -> dict[str, object]:
    resolved = _run(["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], cwd=ROOT)
    if resolved.returncode != 0:
        raise RuntimeError(resolved.stderr.strip() or f"unknown Git revision: {revision}")
    commit = resolved.stdout.strip()

    with tempfile.TemporaryDirectory(prefix="astrograph-git-baseline-") as temporary:
        temporary_root = Path(temporary)
        archive_path = temporary_root / "baseline.tar"
        archived = _run(
            ["git", "archive", "--format=tar", "--output", str(archive_path), commit],
            cwd=ROOT,
        )
        if archived.returncode != 0:
            raise RuntimeError(archived.stderr.strip() or "git archive failed")
        source_root = temporary_root / "source"
        source_root.mkdir()
        with tarfile.open(archive_path, "r") as archive:
            archive.extractall(source_root, filter="data")

        files = {
            path.relative_to(source_root).as_posix(): _sha256(path)
            for path in sorted(source_root.rglob("*"))
            if path.is_file()
        }
        tree_size = sum(path.stat().st_size for path in source_root.rglob("*") if path.is_file())
        manifest = {
            "schema_version": "astrograph-received-source-manifest.v1",
            "version": version,
            "git_commit": commit,
            "git_archive_sha256": _sha256(archive_path),
            "tree_size_bytes": tree_size,
            "files": files,
        }

        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(source_root / "src")
        collected = _run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
                "-c",
                "/dev/null",
                "tests",
            ],
            cwd=source_root,
            env=env,
        )
        if collected.returncode != 0:
            raise RuntimeError(
                "pytest collection failed:\n" + collected.stdout + "\n" + collected.stderr
            )
        nodeids = sorted(
            line.strip()
            for line in collected.stdout.splitlines()
            if "::" in line and not line.startswith("<")
        )
        inventory = {
            "schema_version": "astrograph-test-nodeid-inventory.v1",
            "version": version,
            "git_commit": commit,
            "test_count": len(nodeids),
            "nodeids": nodeids,
        }

    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    inventory_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    inventory_output.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "git_commit": commit,
        "git_archive_sha256": manifest["git_archive_sha256"],
        "tree_size_bytes": tree_size,
        "file_count": len(files),
        "test_count": len(nodeids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--test-inventory", type=Path, required=True)
    args = parser.parse_args()
    report = capture(
        revision=args.revision,
        version=args.version,
        manifest_output=args.manifest.resolve(),
        inventory_output=args.test_inventory.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
