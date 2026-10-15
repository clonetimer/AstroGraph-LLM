#!/usr/bin/env python3
"""Build and fresh-unpack validate a clean AstroGraph source package."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_DIR_NAMES = {
    ".git", ".claude", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv",
    "__pycache__", "worktrees", "build", "dist", "astrograph_core.egg-info",
}
EXCLUDED_PREFIXES = {"scripts/integration/out"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def _excluded(relative: Path) -> bool:
    if EXCLUDED_DIR_NAMES.intersection(relative.parts):
        return True
    normalized = relative.as_posix()
    if any(normalized == prefix or normalized.startswith(prefix + "/") for prefix in EXCLUDED_PREFIXES):
        return True
    return relative.suffix in EXCLUDED_SUFFIXES


def _git_visible_files(source: Path) -> list[Path] | None:
    if not (source / ".git").exists():
        return None
    result = subprocess.run(
        [
            "git", "-C", str(source), "ls-files", "--cached", "--others",
            "--exclude-standard", "-z",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return [
        source / item.decode("utf-8", errors="strict")
        for item in result.stdout.split(b"\0")
        if item
    ]


def _copy_clean(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    git_visible = _git_visible_files(source)
    candidates = git_visible if git_visible is not None else source.rglob("*")
    for path in sorted(candidates):
        relative = path.relative_to(source)
        if _excluded(relative):
            continue
        destination = target / relative
        if path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def _purge_workspace_state(root: Path) -> None:
    """Remove transient runtime/test state generated inside the staged tree.

    The clean copy step excludes workspace state present in the source tree, but
    validators and smoke imports can generate new caches after staging.  Stable
    archives must therefore purge transient state again immediately before
    metadata freeze/zip creation.
    """
    # Explicit generated prefixes first.
    for prefix in EXCLUDED_PREFIXES:
        path = root / prefix
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    # Remove generated cache/build directories anywhere in the staged tree.
    dynamic_names = set(EXCLUDED_DIR_NAMES) | {"__pycache__"}
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if path.name in dynamic_names:
            shutil.rmtree(path, ignore_errors=True)

    # Remove bytecode files that may have been written outside a cache directory.
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in EXCLUDED_SUFFIXES:
            path.unlink(missing_ok=True)


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(cwd / "src")
    return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)


def _zip_tree(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root.parent).as_posix())


def build(output: Path, report_output: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="astrograph-stable-build-") as tmp:
        build_root = Path(tmp) / "AstroGraph"
        _copy_clean(ROOT, build_root)

        # Metadata is regenerated from the exact staged tree, not inherited from the workspace.
        refresh = _run([sys.executable, "scripts/tools/refresh_distribution_metadata.py"], cwd=build_root)
        if refresh.returncode != 0:
            raise RuntimeError(f"metadata refresh failed:\n{refresh.stdout}\n{refresh.stderr}")

        stage_report_path = build_root / "stable_source_validation.json"
        stage_validation = _run(
            [sys.executable, "scripts/release/validate_stable_source.py", "--root", str(build_root), "--output", str(stage_report_path)],
            cwd=build_root,
        )
        if stage_validation.returncode != 0:
            raise RuntimeError(f"staged validation failed:\n{stage_validation.stdout}\n{stage_validation.stderr}")

        # The validation report itself changes the content inventory, so remove it before the final metadata freeze.
        stage_report_path.unlink(missing_ok=True)
        _purge_workspace_state(build_root)
        refresh_final = _run([sys.executable, "scripts/tools/refresh_distribution_metadata.py"], cwd=build_root)
        if refresh_final.returncode != 0:
            raise RuntimeError(f"final metadata refresh failed:\n{refresh_final.stdout}\n{refresh_final.stderr}")

        # Validate the exact tree that will be zipped without writing into it.
        exact_validation = _run(
            [sys.executable, "scripts/release/validate_stable_source.py", "--root", str(build_root)],
            cwd=build_root,
        )
        if exact_validation.returncode != 0:
            raise RuntimeError(f"exact staged validation failed:\n{exact_validation.stdout}\n{exact_validation.stderr}")
        exact_report = json.loads(exact_validation.stdout)

        # Exact validation can itself import modules and create bytecode/cache
        # state in some Python environments.  Purge once more before zipping.
        _purge_workspace_state(build_root)
        _zip_tree(build_root, output)

        with tempfile.TemporaryDirectory(prefix="astrograph-stable-unpack-") as unpack_tmp:
            unpack_dir = Path(unpack_tmp)
            with zipfile.ZipFile(output, "r") as archive:
                archive.extractall(unpack_dir)
            fresh_root = unpack_dir / "AstroGraph"
            fresh_validation = _run(
                [sys.executable, "scripts/release/validate_stable_source.py", "--root", str(fresh_root)],
                cwd=fresh_root,
            )
            if fresh_validation.returncode != 0:
                raise RuntimeError(f"fresh-unpack validation failed:\n{fresh_validation.stdout}\n{fresh_validation.stderr}")
            fresh_report = json.loads(fresh_validation.stdout)

            smoke_tests = _run(
                [
                    sys.executable, "-m", "pytest", "-q",
                    "tests/test_stable_runtime_convergence.py",
                    "tests/test_governance_candidate_safety.py",
                    "tests/test_model_release_loading.py",
                    "tests/test_api_authentication.py",
                    "--disable-warnings",
                ],
                cwd=fresh_root,
            )
            if smoke_tests.returncode != 0:
                raise RuntimeError(f"fresh-unpack smoke tests failed:\n{smoke_tests.stdout}\n{smoke_tests.stderr}")

        report = {
            "schema_version": "astrograph-stable-source-build-report.v1",
            "status": "PASS",
            "source_version": json.loads((ROOT / "release.json").read_text(encoding="utf-8"))["version"],
            "artifact": str(output),
            "artifact_bytes": output.stat().st_size,
            "staged_validation": exact_report,
            "fresh_unpack_validation": fresh_report,
            "fresh_unpack_smoke_tests": smoke_tests.stdout.strip(),
            "excluded_workspace_state": sorted(EXCLUDED_DIR_NAMES),
            "excluded_prefixes": sorted(EXCLUDED_PREFIXES),
            "qualification_boundary": "Architecture/source-package convergence only; formal G0-G4 status remains unchanged.",
        }
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.output.resolve(), args.report.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
