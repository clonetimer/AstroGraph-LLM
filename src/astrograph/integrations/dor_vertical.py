from __future__ import annotations

"""DOR vertical-evidence and development-freeze gate bridge."""

from .process_control import run_managed_subprocess
import json
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.evidence.executable import stable_id

from .dor_runtime import DORRuntimeIdentity, inspect_dor_runtime, sha256_file


DORVerticalTask = Literal[
    "DETECTION", "CLASSIFICATION", "LOCALIZATION", "ROOT_CAUSE", "IDENTIFICATION"
]
DORVerticalGrade = Literal[
    "UNIT_TEST",
    "CONTROLLED_SYNTHETIC",
    "SOURCE_ANCHORED_SYNTHETIC_MECHANISM",
    "HIGH_FIDELITY_SYSTEM_SIMULATION",
    "REAL_HARDWARE_TESTBED",
    "REAL_IN_ORBIT",
]
_TASKS = {"DETECTION", "CLASSIFICATION", "LOCALIZATION", "ROOT_CAUSE", "IDENTIFICATION"}
_GRADES = [
    "UNIT_TEST",
    "CONTROLLED_SYNTHETIC",
    "SOURCE_ANCHORED_SYNTHETIC_MECHANISM",
    "HIGH_FIDELITY_SYSTEM_SIMULATION",
    "REAL_HARDWARE_TESTBED",
    "REAL_IN_ORBIT",
]


class DORVerticalFreezeAuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-dor-vertical-freeze-audit-request.v1"
    vertical_registry_path: str = Field(min_length=1)
    expected_vertical_registry_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    freeze_evidence_path: str = Field(min_length=1)
    expected_freeze_evidence_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    expected_target_domain: str = Field(min_length=1)
    minimum_task_grades: dict[DORVerticalTask, DORVerticalGrade]
    executable_path: str = Field(default="dor", min_length=1)
    timeout_s: float = Field(default=30.0, gt=0.0, le=300.0)
    release_gate: bool = True

    @model_validator(mode="after")
    def require_all_task_policies(self) -> "DORVerticalFreezeAuditRequest":
        if set(self.minimum_task_grades) != _TASKS:
            raise ValueError("minimum_task_grades must declare all five DOR diagnostic tasks")
        return self


class DORVerticalFreezeStatus(str, Enum):
    READY_FOR_FREEZE = "READY_FOR_FREEZE"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


class DORVerticalFreezeAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-dor-vertical-freeze-audit.v1"
    audit_id: str = Field(min_length=1)
    status: DORVerticalFreezeStatus
    reason: str = Field(min_length=1)
    release_gate_enabled: bool
    eligible_for_freeze: bool
    target_domain: str | None = None
    vertical_status: str | None = None
    freeze_status: str | None = None
    missing_vertical_gates: list[str] = Field(default_factory=list)
    blocking_freeze_gates: list[str] = Field(default_factory=list)
    vertical_registry_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    freeze_evidence_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    generated_vertical_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    verified_case_artifact_sha256: dict[str, str] = Field(default_factory=dict)
    runtime: DORRuntimeIdentity | None = None
    vertical_result: dict[str, Any] | None = None
    freeze_result: dict[str, Any] | None = None
    claim_boundary: str = (
        "Development-freeze qualification only. Evidence grade is not a performance score, "
        "and READY does not imply flight safety, operational authorization, or model accuracy."
    )


class DORVerticalFreezeAuditService:
    bridge_version = "astrograph-dor-vertical-freeze-bridge@1.0.0"

    @staticmethod
    def _load_bound_json(path_text: str, expected_sha: str, name: str) -> tuple[Path, dict[str, Any], str]:
        path = Path(path_text).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DOR {name} not found: {path}")
        actual = sha256_file(path)
        if actual != expected_sha.lower():
            raise ValueError(f"DOR {name} SHA-256 mismatch")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"DOR {name} must contain a JSON object")
        return path, payload, actual

    @staticmethod
    def _validate_policy(
        registry: dict[str, Any],
        request: DORVerticalFreezeAuditRequest,
    ) -> dict[str, str]:
        if str(registry.get("target_domain") or "") != request.expected_target_domain:
            raise ValueError("DOR vertical target_domain does not match the expected release domain")
        requirements = {
            str(key).upper(): value for key, value in (registry.get("requirements") or {}).items()
        }
        for task, expected_grade in request.minimum_task_grades.items():
            declared = requirements.get(task)
            if not isinstance(declared, dict):
                raise ValueError(f"DOR vertical registry lacks required task policy: {task}")
            declared_grade = str(declared.get("minimum_evidence_grade") or "")
            if declared_grade not in _GRADES or _GRADES.index(declared_grade) < _GRADES.index(
                expected_grade
            ):
                raise ValueError(
                    f"DOR vertical task {task} declares weaker evidence than {expected_grade}"
                )
            if declared.get("require_target_domain") is not True:
                raise ValueError(f"DOR vertical task {task} must require the target domain")

        verified: dict[str, str] = {}
        for case in registry.get("cases") or []:
            if str(case.get("state", "PASS")).upper() not in {"PASS", "PARTIAL"}:
                continue
            case_id = str(case.get("case_id") or "")
            artifact = str(case.get("artifact") or "")
            artifact_sha = str(case.get("artifact_sha256") or "").lower()
            if not case_id or not artifact or len(artifact_sha) != 64:
                raise ValueError(
                    "every supporting vertical case requires case_id, artifact, and artifact_sha256"
                )
            artifact_path = Path(artifact).resolve()
            if not artifact_path.is_file():
                raise FileNotFoundError(f"DOR vertical case artifact not found: {artifact_path}")
            if sha256_file(artifact_path) != artifact_sha:
                raise ValueError(f"DOR vertical case artifact SHA-256 mismatch: {case_id}")
            verified[case_id] = artifact_sha
        return verified

    @staticmethod
    def _validate_safety(vertical: dict[str, Any], freeze: dict[str, Any]) -> None:
        vertical_safety = dict(vertical.get("safety") or {})
        for key in (
            "cross_domain_evidence_cannot_satisfy_target_domain_gate",
            "evidence_grade_is_not_performance_score",
            "source_anchored_fixture_is_not_high_fidelity_system_simulation",
            "real_hardware_testbed_is_not_real_in_orbit",
            "missing_task_evidence_is_not_inferred",
        ):
            if vertical_safety.get(key) is not True:
                raise ValueError(f"DOR vertical safety invariant violated: {key}")
        freeze_safety = dict(freeze.get("safety") or {})
        for key in (
            "pre_rc_may_be_blocked",
            "blocked_pre_rc_must_not_be_renamed_final",
            "missing_satellite_task_evidence_cannot_be_waived_by_cross_domain_success",
            "test_performance_cannot_override_contract_gate",
            "application_stage_requires_core_contract_stability",
        ):
            if freeze_safety.get(key) is not True:
                raise ValueError(f"DOR freeze safety invariant violated: {key}")

    def audit(self, request: DORVerticalFreezeAuditRequest) -> DORVerticalFreezeAudit:
        runtime: DORRuntimeIdentity | None = None
        registry_sha: str | None = None
        evidence_sha: str | None = None
        generated_sha: str | None = None
        verified_cases: dict[str, str] = {}
        vertical: dict[str, Any] | None = None
        freeze: dict[str, Any] | None = None
        try:
            registry_path, registry, registry_sha = self._load_bound_json(
                request.vertical_registry_path,
                request.expected_vertical_registry_sha256,
                "vertical registry",
            )
            evidence_path, _, evidence_sha = self._load_bound_json(
                request.freeze_evidence_path,
                request.expected_freeze_evidence_sha256,
                "freeze evidence",
            )
            verified_cases = self._validate_policy(registry, request)
            runtime = inspect_dor_runtime(request.executable_path, timeout_s=request.timeout_s)
            with tempfile.TemporaryDirectory(prefix="astrograph-dor-vertical-") as directory:
                root = Path(directory)
                vertical_path = root / "vertical.json"
                completed = run_managed_subprocess(
                    [
                        runtime.executable,
                        "vertical-validate",
                        "--registry",
                        str(registry_path),
                        "--output",
                        str(vertical_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=request.timeout_s,
                    check=False,
                )
                if completed.returncode not in {0, 2}:
                    detail = (completed.stderr or completed.stdout or "no DOR output").strip()[-2000:]
                    raise RuntimeError(f"DOR vertical validation exited {completed.returncode}: {detail}")
                vertical = json.loads(vertical_path.read_text(encoding="utf-8"))
                generated_sha = sha256_file(vertical_path)
                freeze_path = root / "freeze.json"
                completed = run_managed_subprocess(
                    [
                        runtime.executable,
                        "freeze-gate",
                        "--evidence",
                        str(evidence_path),
                        "--vertical",
                        str(vertical_path),
                        "--output",
                        str(freeze_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=request.timeout_s,
                    check=False,
                )
                if completed.returncode not in {0, 3}:
                    detail = (completed.stderr or completed.stdout or "no DOR output").strip()[-2000:]
                    raise RuntimeError(f"DOR freeze gate exited {completed.returncode}: {detail}")
                freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

            if vertical.get("contract") != "DOR-VERTICAL-VALIDATION/0.9":
                raise ValueError("unexpected DOR vertical contract")
            if freeze.get("contract") != "DOR-DEVELOPMENT-FREEZE-GATE/1.0":
                raise ValueError("unexpected DOR freeze contract")
            self._validate_safety(vertical, freeze)
            ready = (
                vertical.get("status") == "FULLY_VALIDATED"
                and freeze.get("status") == "READY_FOR_1_0_FREEZE"
            )
            status = (
                DORVerticalFreezeStatus.READY_FOR_FREEZE
                if ready
                else DORVerticalFreezeStatus.BLOCKED
            )
            reason = str(freeze.get("reason") or vertical.get("reason") or "no DOR reason")
            identity = {
                "registry_sha256": registry_sha,
                "freeze_evidence_sha256": evidence_sha,
                "generated_vertical_sha256": generated_sha,
                "minimum_task_grades": request.minimum_task_grades,
                "runtime_fingerprint": runtime.stack_fingerprint_sha256,
                "vertical": vertical,
                "freeze": freeze,
            }
            return DORVerticalFreezeAudit(
                audit_id=stable_id("DOR-VF", identity),
                status=status,
                reason=reason,
                release_gate_enabled=request.release_gate,
                eligible_for_freeze=ready,
                target_domain=str(vertical.get("target_domain") or "") or None,
                vertical_status=str(vertical.get("status") or "") or None,
                freeze_status=str(freeze.get("status") or "") or None,
                missing_vertical_gates=list(map(str, vertical.get("missing_gates") or [])),
                blocking_freeze_gates=list(map(str, freeze.get("blocking_gates") or [])),
                vertical_registry_sha256=registry_sha,
                freeze_evidence_sha256=evidence_sha,
                generated_vertical_sha256=generated_sha,
                verified_case_artifact_sha256=verified_cases,
                runtime=runtime,
                vertical_result=vertical,
                freeze_result=freeze,
            )
        except Exception as exc:
            identity = {
                "registry_sha256": registry_sha,
                "freeze_evidence_sha256": evidence_sha,
                "generated_vertical_sha256": generated_sha,
                "runtime_fingerprint": runtime.stack_fingerprint_sha256 if runtime else None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            return DORVerticalFreezeAudit(
                audit_id=stable_id("DOR-VF", identity),
                status=DORVerticalFreezeStatus.ERROR,
                reason=f"{type(exc).__name__}: {exc}",
                release_gate_enabled=request.release_gate,
                eligible_for_freeze=False,
                vertical_registry_sha256=registry_sha,
                freeze_evidence_sha256=evidence_sha,
                generated_vertical_sha256=generated_sha,
                verified_case_artifact_sha256=verified_cases,
                runtime=runtime,
                vertical_result=vertical,
                freeze_result=freeze,
            )


def vertical_freeze_release_blockers(
    audit: DORVerticalFreezeAudit | None,
) -> list[str]:
    if audit is None or not audit.release_gate_enabled or audit.eligible_for_freeze:
        return []
    return [f"DOR_VERTICAL_FREEZE_{audit.status.value}"]
