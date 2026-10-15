#!/usr/bin/env python3
"""Create a truth-blind Sat-Agent source and interpreter identity receipt.

Run this utility with the independently managed Sat-Agent interpreter.  It
hashes source/configuration inputs and the active package environment, but it
does not import AstroGraph, read telemetry rows, inspect labels, create a seed,
or execute a simulator campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from packaging.requirements import Requirement


HASHED_SUFFIXES = frozenset({".py", ".json", ".yaml", ".yml", ".toml", ".txt"})
TREE_CATEGORIES = {
    "runtime_source": ("src",),
    "generator_source": ("scripts",),
    "configuration_inputs": ("configs",),
    "scenario_templates": ("src/sat_sim/experiment_templates",),
    "capability_and_effect_maps": (
        "src/sat_sim/capabilities",
        "src/sat_sim/fault_mechanisms",
        "configs/evidence",
    ),
}
CONTROL_FILES = (
    "pyproject.toml",
    "requirements-lock.txt",
    "src/sat_sim/release_manifest.json",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _assert_within(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("source audit input escapes the injected root") from exc
    return resolved


def _iter_category_files(root: Path, members: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for member in members:
        candidate = _assert_within(root, root / member)
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        if candidate.is_file():
            if candidate.suffix.lower() in HASHED_SUFFIXES:
                files.add(candidate)
            continue
        for path in candidate.rglob("*"):
            if path.is_file() and path.suffix.lower() in HASHED_SUFFIXES:
                files.add(_assert_within(root, path))
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def _tree_record(root: Path, members: Iterable[str]) -> dict[str, Any]:
    files = _iter_category_files(root, members)
    records = []
    total_bytes = 0
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "file_count": len(records),
        "total_bytes": total_bytes,
        "manifest_sha256": _canonical_sha256(records),
        "individual_paths_emitted": False,
    }


def _installed_distributions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name") or "").strip().lower().replace("_", "-")
        if name:
            versions[name] = str(distribution.version)
    return dict(sorted(versions.items()))


def _direct_dependency_receipt(pyproject: dict[str, Any]) -> dict[str, Any]:
    declared = pyproject.get("project", {}).get("dependencies", [])
    installed = _installed_distributions()
    active: list[dict[str, Any]] = []
    failures: list[str] = []
    for raw in declared:
        requirement = Requirement(str(raw))
        if requirement.marker and not requirement.marker.evaluate():
            continue
        normalized_name = requirement.name.lower().replace("_", "-")
        observed = installed.get(normalized_name)
        satisfied = observed is not None and (
            not requirement.specifier or requirement.specifier.contains(observed, prereleases=True)
        )
        active.append(
            {
                "distribution": normalized_name,
                "declared_specifier": str(requirement.specifier),
                "observed_version": observed,
                "satisfied": satisfied,
            }
        )
        if not satisfied:
            failures.append(normalized_name)
    return {
        "active_direct_dependency_count": len(active),
        "requirements": sorted(active, key=lambda item: item["distribution"]),
        "all_active_direct_dependencies_satisfied": not failures,
        "failed_distribution_names": sorted(failures),
    }


def _pip_check() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    output = completed.stdout.strip().encode("utf-8")
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "return_code": completed.returncode,
        "output_sha256": _sha256_bytes(output),
        "output_emitted": False,
    }


def build_receipt(root: Path, archive: Path, repair_receipt: Path) -> dict[str, Any]:
    root = root.resolve()
    archive = archive.resolve()
    repair_receipt = repair_receipt.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not repair_receipt.is_file():
        raise FileNotFoundError(repair_receipt)

    pyproject_path = _assert_within(root, root / "pyproject.toml")
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    expected_name = "satellite-simulation-platform"
    expected_version = str(project.get("version"))
    if project.get("name") != expected_name:
        raise ValueError("injected source is not the expected Sat-Agent distribution")

    control_hashes: dict[str, str] = {}
    for relative in CONTROL_FILES:
        path = _assert_within(root, root / relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        control_hashes[relative] = _sha256_file(path)

    trees = {
        category: _tree_record(root, members)
        for category, members in TREE_CATEGORIES.items()
    }
    installed = _installed_distributions()
    environment_manifest_sha256 = _canonical_sha256(installed)
    direct = _direct_dependency_receipt(pyproject)
    pip_check = _pip_check()

    import_checks: dict[str, dict[str, Any]] = {}
    for module_name in ("sat_sim", "Basilisk"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # fail-closed receipt, without leaking a local path
            import_checks[module_name] = {
                "status": "FAIL",
                "error_type": type(exc).__name__,
            }
        else:
            import_checks[module_name] = {"status": "PASS"}

    observed_project_version = installed.get(expected_name)
    archive_record = {
        "token": "${SAT_AGENT_SOURCE_ARCHIVE}",
        "size_bytes": archive.stat().st_size,
        "sha256": _sha256_file(archive),
    }
    repair_record = {
        "token": "${ASTROGRAPH_ROOT}/governance/evidence/0.7.1-extension/"
        "p018_sat_agent_time_axis_repair_receipt.v1.json",
        "sha256": _sha256_file(repair_receipt),
        "role": "AUTHORIZED_57_FILE_PUBLIC_TIME_AXIS_OVERLAY",
    }
    identity_material = {
        "archive_sha256": archive_record["sha256"],
        "repair_receipt_sha256": repair_record["sha256"],
        "control_hashes": control_hashes,
        "tree_hashes": {
            key: value["manifest_sha256"] for key, value in trees.items()
        },
        "environment_manifest_sha256": environment_manifest_sha256,
    }
    failures = []
    if observed_project_version != expected_version:
        failures.append("PROJECT_VERSION_MISMATCH")
    if not direct["all_active_direct_dependencies_satisfied"]:
        failures.append("DIRECT_DEPENDENCY_MISMATCH")
    if pip_check["status"] != "PASS":
        failures.append("PIP_CHECK_FAILED")
    if any(item["status"] != "PASS" for item in import_checks.values()):
        failures.append("REQUIRED_IMPORT_FAILED")
    if "astrograph" in installed:
        failures.append("ASTROGRAPH_DISTRIBUTION_PRESENT_IN_SAT_AGENT_ENVIRONMENT")

    return {
        "schema_version": "astrograph.p018.sat-agent-source-environment-receipt.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "P018_S0_ZERO_FIT",
        "status": "PASS" if not failures else "FAIL",
        "source_id": "SAT_AGENT_PLATFORM",
        "source_count": 1,
        "source_revision": {
            "repository_revision_status": (
                "AVAILABLE" if (root / ".git").exists() else "NOT_AVAILABLE"
            ),
            "revision_anchor_policy": (
                "SOURCE_ARCHIVE_SHA256_PLUS_AUTHORIZED_REPAIR_RECEIPT_SHA256"
            ),
            "archive": archive_record,
            "authorized_overlay": repair_record,
            "effective_source_identity_sha256": _canonical_sha256(identity_material),
        },
        "source_controls": {
            "root_token": "${SAT_AGENT_PLATFORM_ROOT}",
            "distribution_name": expected_name,
            "declared_version": expected_version,
            "release_id": json.loads(
                (root / "src/sat_sim/release_manifest.json").read_text(encoding="utf-8")
            ).get("release_id"),
            "control_file_sha256": control_hashes,
            "aggregate_trees": trees,
        },
        "independent_environment": {
            "interpreter_token": "${SAT_AGENT_PYTHON}",
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "observed_project_version": observed_project_version,
            "distribution_count": len(installed),
            "distribution_manifest_sha256": environment_manifest_sha256,
            "individual_environment_paths_emitted": False,
            "astrograph_distribution_present": "astrograph" in installed,
            "direct_dependencies": direct,
            "pip_check": pip_check,
            "import_checks": import_checks,
        },
        "execution_receipt": {
            "read_only_source_audit": True,
            "simulator_execution_count": 0,
            "campaign_generation_count": 0,
            "telemetry_row_read_count": 0,
            "truth_or_label_value_read_count": 0,
            "model_fit_count": 0,
            "threshold_selection_count": 0,
            "split_creation_count": 0,
            "seed_creation_or_reservation_count": 0,
            "formal_hidden_acceptance_truth_accessed": False,
        },
        "failure_codes": failures,
        "readiness": {
            "source_and_environment_identity_complete": not failures,
            "generator_replay_completed_by_this_receipt": False,
            "stage_s1_fit_authorized": False,
            "stage_b_authorized": False,
            "g2_pass": False,
            "g3_pass": False,
        },
        "claim_boundary": (
            "Source and independent-environment identity evidence only. The effective "
            "revision is the source archive plus the authorized 57-file public time-axis "
            "overlay. No telemetry, label, truth, seed, fit, split, performance result, "
            "Stage S1, Stage B, cross-source or flight claim is produced."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--repair-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = {args.root.resolve(), args.archive.resolve(), args.repair_receipt.resolve()}
    output = args.output.resolve()
    if output in inputs:
        raise ValueError("output must not overwrite an input")
    receipt = build_receipt(args.root, args.archive, args.repair_receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": receipt["status"], "output": str(output)}))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
