"""AI-Native Phase 5: governance health checker tests.

Verifies the advisory, read-only checker on the real repository (PASS path) and
on synthetic broken governance trees (FAIL paths), plus determinism.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astrograph.governance.health_check import (
    GovernanceCheckResult,
    GovernanceHealthChecker,
    GovernanceHealthReport,
    HealthStatus,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKED_AT = "2026-08-23T00:00:00+00:00"


def _write_minimal_tree(root: Path, *, version: str = "1.2.3", sha_ok: bool = True) -> Path:
    import hashlib
    import tomllib
    (root / "src/astrograph").mkdir(parents=True)
    (root / "governance/contracts").mkdir(parents=True)
    (root / "governance/evidence/ai_native/feedback_cycles").mkdir(parents=True)
    (root / "release.json").write_text(json.dumps({"version": version}), encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\nname = \"x\"\nversion = \"" + version + "\"\n",
        encoding="utf-8",
    )
    (root / "src/astrograph/__init__.py").write_text(
        "__version__ = \"" + version + "\"\n",
        encoding="utf-8",
    )
    contract = "{}"
    (root / "governance/contracts/1.2.3.yaml").write_text(contract, encoding="utf-8")
    expected = hashlib.sha256(contract.encode()).hexdigest() if sha_ok else "0" * 64
    (root / "governance/contracts/1.2.3.sha256").write_text(expected + "  file", encoding="utf-8")
    return root


def test_health_check_passes_on_current_repo():
    report = GovernanceHealthChecker().run(ROOT, checked_at=CHECKED_AT)
    assert report.overall_status == HealthStatus.PASS
    statuses = {item.check_id: item.status for item in report.checks}
    assert statuses["version_consistency"] == HealthStatus.PASS
    assert statuses["contract_sha_chain"] == HealthStatus.PASS
    assert statuses["governance_artifacts_present"] == HealthStatus.PASS


def test_health_check_is_advisory_and_read_only():
    report = GovernanceHealthChecker().run(ROOT, checked_at=CHECKED_AT)
    assert report.advisory is True
    assert "ReleaseQualificationGate" in report.claim_boundary
    # The checker must not modify the governance tree.
    import hashlib
    before = hashlib.sha256((ROOT / "release.json").read_bytes()).hexdigest()
    GovernanceHealthChecker().run(ROOT, checked_at=CHECKED_AT)
    after = hashlib.sha256((ROOT / "release.json").read_bytes()).hexdigest()
    assert before == after


def test_health_check_is_deterministic():
    first = GovernanceHealthChecker().run(ROOT, checked_at=CHECKED_AT)
    second = GovernanceHealthChecker().run(ROOT, checked_at=CHECKED_AT)
    assert [item.model_dump() for item in first.checks] == [item.model_dump() for item in second.checks]


def test_health_check_detects_version_mismatch(tmp_path: Path):
    root = _write_minimal_tree(tmp_path / "tree", version="1.2.3")
    (root / "release.json").write_text(json.dumps({"version": "9.9.9"}), encoding="utf-8")
    report = GovernanceHealthChecker().run(root, checked_at=CHECKED_AT)
    by_id = {item.check_id: item for item in report.checks}
    assert by_id["version_consistency"].status == HealthStatus.FAIL
    assert report.overall_status == HealthStatus.FAIL


def test_health_check_detects_sha_mismatch(tmp_path: Path):
    root = _write_minimal_tree(tmp_path / "tree", sha_ok=False)
    report = GovernanceHealthChecker().run(root, checked_at=CHECKED_AT)
    by_id = {item.check_id: item for item in report.checks}
    assert by_id["contract_sha_chain"].status == HealthStatus.FAIL
    assert report.overall_status == HealthStatus.FAIL


def test_health_check_failed_property():
    report = GovernanceHealthChecker().run(ROOT, checked_at=CHECKED_AT)
    assert report.failed_checks == []
    assert isinstance(report, GovernanceHealthReport)
