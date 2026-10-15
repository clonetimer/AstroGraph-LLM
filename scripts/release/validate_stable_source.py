#!/usr/bin/env python3
"""Validate the AstroGraph Stable Architecture Convergence source boundary.

This validator is intentionally narrower than flight/qualification governance. It proves
that the source package is clean, internally reproducible, and wired through the canonical
Runtime/API/CLI and stable contracts. It does not change G0-G4 release qualification.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_distribution_validator(root: Path):
    path = root / "scripts" / "data_driven" / "validate_distribution.py"
    spec = importlib.util.spec_from_file_location("astrograph_distribution_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load distribution validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check(checks: list[dict[str, Any]], errors: list[dict[str, str]], *, name: str, passed: bool, detail: str, code: str) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})
    if not passed:
        errors.append({"code": code, "detail": detail})


def _absolute_developer_path_hits(root: Path) -> list[str]:
    hits: list[str] = []
    for base_name in ("src", "tests", "scripts"):
        base = root / base_name
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix in {".pyc", ".pyo"}:
                continue
            relative = path.relative_to(root).as_posix()
            if relative.startswith("scripts/integration/out/"):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            forbidden_root = "/mnt/" + "PRO6000_disk"
            if forbidden_root in text:
                hits.append(path.relative_to(root).as_posix())
    return hits


def validate(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    checks: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    # Reuse the governed distribution integrity validator first.
    distribution_module = _load_distribution_validator(root)
    distribution = distribution_module.validate(root)
    _check(
        checks,
        errors,
        name="governed_distribution_integrity",
        passed=distribution.get("status") == "PASS",
        detail=f"status={distribution.get('status')}; error_count={distribution.get('error_count', 0)}",
        code="GOVERNED_DISTRIBUTION_INVALID",
    )

    forbidden = [
        ".git",
        ".claude",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "scripts/integration/out",
        "src/astrograph_core.egg-info",
    ]
    present = [item for item in forbidden if (root / item).exists()]
    dynamic_present = sorted({
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir() and path.name in {"__pycache__", "worktrees"}
    })
    _check(
        checks,
        errors,
        name="workspace_state_excluded",
        passed=not present and not dynamic_present,
        detail=json.dumps({"fixed": present, "dynamic": dynamic_present}, ensure_ascii=False),
        code="WORKSPACE_STATE_PRESENT",
    )

    release = json.loads((root / "release.json").read_text(encoding="utf-8"))
    convergence = release.get("architecture_convergence", {})
    prod = release.get("production_paths", {})
    _check(
        checks,
        errors,
        name="architecture_convergence_verified",
        passed=convergence.get("status") == "VERIFIED_SOURCE_ARCHITECTURE",
        detail=str(convergence.get("status", "")),
        code="ARCHITECTURE_CONVERGENCE_NOT_VERIFIED",
    )
    _check(
        checks,
        errors,
        name="canonical_runtime_declared",
        passed=(
            prod.get("runtime") == "astrograph.runtime"
            and prod.get("api") == "astrograph.api"
            and prod.get("cli") == "astrograph.cli"
            and convergence.get("canonical_runtime") == "astrograph.runtime.AstroGraphRuntime"
        ),
        detail=json.dumps({"runtime": prod.get("runtime"), "api": prod.get("api"), "cli": prod.get("cli")}),
        code="CANONICAL_ENTRYPOINTS_INVALID",
    )

    legacy_production = [
        f"{name}={value}"
        for name, value in prod.items()
        if isinstance(value, str)
        and (
            value.startswith("astrograph.diagnosis.sat_sim")
            or value == "astrograph.diagnosis.unified"
            or value.endswith(".joblib")
        )
    ]
    _check(
        checks,
        errors,
        name="compatibility_is_peripheral",
        passed=not legacy_production and bool(release.get("compatibility_paths")),
        detail=json.dumps({"legacy_in_production": legacy_production, "compat_count": len(release.get("compatibility_paths", {}))}),
        code="LEGACY_PRODUCTION_PATH_PRESENT",
    )

    sys.path.insert(0, str(root / "src"))
    from astrograph.architecture import load_stable_layer_policy, scan_stable_layers
    from astrograph.contracts import STABLE_CONTRACTS
    from astrograph.governance.delivery_evidence import validate_delivery_evidence
    from astrograph.runtime import (
        DIAGNOSIS_PIPELINE_ID,
        UNIFIED_DIAGNOSIS_PIPELINE_ID,
        WHOLE_SPACECRAFT_PIPELINE_ID,
    )

    delivery_evidence = validate_delivery_evidence(root, require_bundled=False)
    _check(
        checks,
        errors,
        name="delivery_evidence_consistency",
        passed=delivery_evidence.get("status") != "FAIL",
        detail=json.dumps({
            "status": delivery_evidence.get("status"),
            "bundled_count": delivery_evidence.get("bundled_count"),
            "external_declared_count": delivery_evidence.get("external_declared_count"),
        }, ensure_ascii=False),
        code="DELIVERY_EVIDENCE_INCONSISTENT",
    )

    layer_report = scan_stable_layers(
        root,
        load_stable_layer_policy(root / "configs" / "architecture" / "stable_layers.json"),
    )
    _check(
        checks,
        errors,
        name="stable_layer_dependency_direction",
        passed=layer_report.status == "PASS",
        detail=json.dumps(layer_report.model_dump(mode="json"), ensure_ascii=False),
        code="STABLE_LAYER_DEPENDENCY_VIOLATION",
    )

    contract_ids = [
        STABLE_CONTRACTS.model.contract_id,
        STABLE_CONTRACTS.evidence.contract_id,
        STABLE_CONTRACTS.knowledge.contract_id,
        STABLE_CONTRACTS.policy.contract_id,
    ]
    _check(
        checks,
        errors,
        name="stable_contract_bundle",
        passed=(
            len(contract_ids) == len(set(contract_ids)) == 4
            and STABLE_CONTRACTS.model.artifact_manifest_required
            and not STABLE_CONTRACTS.model.raw_pickle_allowed
            and not STABLE_CONTRACTS.policy.dataset_profile_may_override
        ),
        detail=json.dumps(contract_ids),
        code="STABLE_CONTRACT_BUNDLE_INVALID",
    )

    declared_pipelines = convergence.get("runtime_pipelines", [])
    expected_pipelines = [
        DIAGNOSIS_PIPELINE_ID,
        UNIFIED_DIAGNOSIS_PIPELINE_ID,
        WHOLE_SPACECRAFT_PIPELINE_ID,
    ]
    _check(
        checks,
        errors,
        name="runtime_pipeline_registry_declared",
        passed=declared_pipelines == expected_pipelines,
        detail=json.dumps({"declared": declared_pipelines, "expected": expected_pipelines}),
        code="RUNTIME_PIPELINE_DECLARATION_MISMATCH",
    )

    absolute_hits = _absolute_developer_path_hits(root)
    _check(
        checks,
        errors,
        name="portable_executable_paths",
        passed=not absolute_hits,
        detail=json.dumps(absolute_hits),
        code="DEVELOPER_ABSOLUTE_PATH_PRESENT",
    )

    # Import smoke test in a clean child interpreter with bytecode disabled.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    smoke = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import astrograph; "
                "from astrograph.runtime import AstroGraphRuntime, RuntimeRequest; "
                "from astrograph.contracts import STABLE_CONTRACTS; "
                "print(astrograph.__version__, AstroGraphRuntime.runtime_version, STABLE_CONTRACTS.schema_version)"
            ),
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    _check(
        checks,
        errors,
        name="clean_import_smoke",
        passed=smoke.returncode == 0,
        detail=(smoke.stdout + smoke.stderr).strip(),
        code="CLEAN_IMPORT_SMOKE_FAILED",
    )

    return {
        "schema_version": "astrograph-stable-source-validation.v1",
        "status": "PASS" if not errors else "FAIL",
        "root": str(root),
        "check_count": len(checks),
        "error_count": len(errors),
        "checks": checks,
        "errors": errors,
        "distribution": distribution,
        "delivery_evidence": delivery_evidence,
        "qualification_boundary": (
            "PASS establishes source-package integrity and Stable Architecture Convergence only; "
            "it does not establish G0-G4, flight, causal, hardware, or safety qualification."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
