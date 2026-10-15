"""Read-only governance health checker (AI-Native roadmap Phase 5).

Performs an advisory, deterministic health check over the governance evidence
tree: version consistency across release.json / pyproject.toml / __init__.py,
the frozen contract SHA chain (yaml -> .sha256), presence of per-version
governance artifacts, and integrity of the AI-Native evidence tree.

The checker is advisory only: it never modifies governance state and its
findings never gate a formal release by themselves.
"""

from __future__ import annotations

import ast
import hashlib
import json
import tomllib
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class HealthStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class GovernanceCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str = Field(min_length=1)
    status: HealthStatus
    detail: str = Field(min_length=1)


class GovernanceHealthReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-governance-health-report.v1"
    report_id: str = Field(min_length=1)
    project_version: str = Field(min_length=1)
    checked_at: str = Field(min_length=1)
    checks: list[GovernanceCheckResult] = Field(default_factory=list)
    overall_status: HealthStatus
    advisory: bool = True
    claim_boundary: str = Field(
        default="Advisory read-only governance health check; findings never gate or authorize release; formal release remains with ReleaseQualificationGate and human review."
    )

    @property
    def failed_checks(self) -> list[GovernanceCheckResult]:
        return [item for item in self.checks if item.status == HealthStatus.FAIL]

    @property
    def warned_checks(self) -> list[GovernanceCheckResult]:
        return [item for item in self.checks if item.status == HealthStatus.WARNING]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_init_version(init_text: str) -> str:
    tree = ast.parse(init_text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == "__version__":
                    return str(ast.literal_eval(node.value))
    return "?"


class GovernanceHealthChecker:
    """Deterministic, advisory governance health check."""

    checker_id = "governance-health-checker"
    checker_version = "0.1.0"

    def run(
        self,
        project_root: str | Path,
        *,
        checked_at: str | None = None,
    ) -> GovernanceHealthReport:
        root = Path(project_root).resolve()
        checks: list[GovernanceCheckResult] = []
        version = self._check_version_consistency(root, checks)
        self._check_contract_sha_chain(root, checks)
        self._check_governance_artifacts(root, version, checks)
        self._check_ai_native_evidence(root, checks)
        self._check_feedback_ledger_integrity(root, checks)
        self._check_l0_l5_maturity(root, checks)
        severity = {
            HealthStatus.PASS: 0,
            HealthStatus.WARNING: 1,
            HealthStatus.FAIL: 2,
        }
        worst = max(
            (item.status for item in checks),
            default=HealthStatus.PASS,
            key=severity.__getitem__,
        )
        return GovernanceHealthReport(
            report_id="GOV-HLTH",
            project_version=version,
            checked_at=checked_at or _now(),
            checks=checks,
            overall_status=worst,
        )


    def _check_l0_l5_maturity(self, root: Path, checks: list) -> None:
        delivery = root / 'DELIVERY_PROFILE.json'
        if delivery.is_file():
            readiness = root / 'governance/implementation/post49/RELEASE_READINESS.json'
            status = HealthStatus.PASS if readiness.is_file() else HealthStatus.FAIL
            detail = ('current delivery readiness present; historical maturity ledgers intentionally excluded'
                      if readiness.is_file() else 'current delivery readiness missing')
        else:
            evidence = root / 'governance/evidence/ai_native'
            dimensions = {
                'architecture': evidence / 'ai_native_maturity_matrix.json',
                'collaboration': evidence / 'federation',
                'data_injection': evidence / 'feedback_loop_baseline.json',
                'model_lifecycle': evidence / 'lifecycle',
                'operations': evidence / 'governance_health',
            }
            missing = [name for name, path in dimensions.items() if not path.exists()]
            status = HealthStatus.PASS if not missing else HealthStatus.WARNING
            detail = ('five-dimension L0-L5 evidence present' if not missing
                      else 'missing L0-L5 dimension evidence: ' + ', '.join(sorted(missing)))
        checks.append(GovernanceCheckResult(
            check_id='l0_l5_maturity',
            status=status,
            detail=detail,
        ))
    def _check_version_consistency(self, root: Path, checks: list) -> str:
        release = json.loads((root / "release.json").read_text(encoding="utf-8"))
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        init_text = (root / "src/astrograph/__init__.py").read_text(encoding="utf-8")
        values = (
            str(release["version"]),
            str(pyproject["project"]["version"]),
            _read_init_version(init_text),
        )
        consistent = len(set(values)) == 1
        checks.append(
            GovernanceCheckResult(
                check_id="version_consistency",
                status=HealthStatus.PASS if consistent else HealthStatus.FAIL,
                detail=(
                    "release.json / pyproject.toml / __init__.py agree: " + values[0]
                    if consistent
                    else "version mismatch: " + " | ".join(values)
                ),
            )
        )
        return values[0]

    def _check_contract_sha_chain(self, root: Path, checks: list) -> None:
        contract_dir = root / "governance/contracts"
        mismatches: list[str] = []
        if not contract_dir.is_dir():
            mismatches.append("contracts-dir-missing")
        else:
            for yaml_path in sorted(contract_dir.glob("*.yaml")):
                sha_path = contract_dir / (yaml_path.stem + ".sha256")
                if not sha_path.is_file():
                    mismatches.append(yaml_path.name + ":missing-sha-file")
                    continue
                actual = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
                expected = sha_path.read_text(encoding="utf-8").split()[0]
                if actual != expected:
                    mismatches.append(yaml_path.name + ":sha-mismatch")
        checks.append(
            GovernanceCheckResult(
                check_id="contract_sha_chain",
                status=HealthStatus.PASS if not mismatches else HealthStatus.FAIL,
                detail=("all contract yaml hashes match their .sha256 files" if not mismatches
                        else "mismatches: " + ", ".join(mismatches[:5])),
            )
        )

    def _check_governance_artifacts(self, root: Path, version: str, checks: list) -> None:
        missing: list[str] = []
        for relative in (
            "governance/contracts/" + version + ".yaml",
            "governance/contracts/" + version + ".sha256",
            "governance/change_requests/" + version + ".json",
            "governance/task_boards/" + version + ".json",
            "governance/rollback/" + version + ".json",
        ):
            if not (root / relative).is_file():
                missing.append(relative)
        checks.append(
            GovernanceCheckResult(
                check_id="governance_artifacts_present",
                status=HealthStatus.PASS if not missing else HealthStatus.FAIL,
                detail=("per-version governance artifacts present" if not missing
                        else "missing: " + ", ".join(missing)),
            )
        )

    def _check_ai_native_evidence(self, root: Path, checks: list) -> None:
        if (root / 'DELIVERY_PROFILE.json').is_file():
            required = (
                'governance/implementation/post49/RELEASE_READINESS.json',
                'governance/evidence/CURRENT_RUNTIME_ASSETS.json',
                'docs/CURRENT_ENGINEERING_POLICY.md',
            )
            missing = [relative for relative in required if not (root / relative).is_file()]
            checks.append(
                GovernanceCheckResult(
                    check_id='ai_native_evidence_tree',
                    status=HealthStatus.PASS if not missing else HealthStatus.FAIL,
                    detail=('current delivery governance assets present; historical evidence intentionally excluded'
                            if not missing else 'missing current delivery assets: ' + ', '.join(missing)),
                )
            )
            return
        missing: list[str] = []
        for relative in (
            "governance/evidence/ai_native/ai_native_maturity_matrix.json",
            "governance/evidence/ai_native/feedback_loop_baseline.json",
            "governance/evidence/ai_native/feedback_cycles/README.md",
            "docs/CURRENT_ENGINEERING_POLICY.md",
        ):
            if not (root / relative).is_file():
                missing.append(relative)
        checks.append(
            GovernanceCheckResult(
                check_id="ai_native_evidence_tree",
                status=HealthStatus.PASS if not missing else HealthStatus.WARNING,
                detail=("AI-Native evidence tree complete" if not missing
                        else "missing: " + ", ".join(missing)),
            )
        )

    def _check_feedback_ledger_integrity(self, root: Path, checks: list) -> None:
        if (root / 'DELIVERY_PROFILE.json').is_file():
            ledger_dir = root / 'governance/runtime_state/feedback_cycles'
        else:
            ledger_dir = root / "governance/evidence/ai_native/feedback_cycles"
        bad: list[str] = []
        if ledger_dir.is_dir():
            for jsonl in sorted(ledger_dir.glob("*.jsonl")):
                for line in jsonl.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            bad.append(jsonl.name)
                            break
        checks.append(
            GovernanceCheckResult(
                check_id="feedback_ledger_integrity",
                status=HealthStatus.PASS if not bad else HealthStatus.WARNING,
                detail=("feedback ledger lines parse as JSON" if not bad
                        else "bad lines: " + ", ".join(bad)),
            )
        )


__all__ = [
    "GovernanceCheckResult",
    "GovernanceHealthChecker",
    "GovernanceHealthReport",
    "HealthStatus",
]
