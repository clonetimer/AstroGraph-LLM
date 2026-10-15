from __future__ import annotations

"""Fail-closed DOR dataset/task qualification and engine-admission audit."""

from .process_control import run_managed_subprocess
import json
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.evidence.executable import stable_id

from .dor_runtime import DORRuntimeIdentity, inspect_dor_runtime, sha256_file


class DORControlPlaneStatus(str, Enum):
    QUALIFIED = "QUALIFIED"
    CONDITIONALLY_QUALIFIED = "CONDITIONALLY_QUALIFIED"
    UNSUPPORTED = "UNSUPPORTED"
    ERROR = "ERROR"


class DORControlPlaneAuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_manifest_path: str = Field(min_length=1)
    diagnostic_request_path: str = Field(min_length=1)
    data_path: str = Field(min_length=1)
    executable_path: str = Field(default="dor", min_length=1)
    timeout_s: float = Field(default=120.0, gt=0.0, le=600.0)
    release_gate: bool = True
    allow_conditionally_qualified: bool = False
    expected_reference_scope: dict[str, Any] | None = None


class DORControlPlaneAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-dor-control-plane-audit.v2"
    audit_id: str = Field(min_length=1)
    status: DORControlPlaneStatus
    reason: str = Field(min_length=1)
    task: str | None = None
    release_gate_enabled: bool
    allow_conditionally_qualified: bool
    eligible_for_diagnostic_execution: bool
    dataset_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    diagnostic_request_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    data_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    runtime: DORRuntimeIdentity | None = None
    bridge_version: str = Field(min_length=1)
    manifest_contract: str | None = None
    plan_contract: str | None = None
    plan_id: str | None = None
    task_qualification: dict[str, Any] | None = None
    engine_admissions: list[dict[str, Any]] = Field(default_factory=list)
    reference_applicability_summary: dict[str, Any] | None = None
    reference_safety: dict[str, Any] | None = None
    declared_reference_scope: dict[str, Any] | None = None
    claim_boundary: str = (
        "Dataset/task/control-plane qualification only. This audit does not train or select models, "
        "calibrate evidence, infer faults, or authorize operational actions."
    )


class DORControlPlaneAuditService:
    bridge_version = "astrograph-dor-control-plane-bridge@1.0.0"

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object: {path.name}")
        return value

    @staticmethod
    def _optional_hash(path: str) -> str | None:
        candidate = Path(path)
        return sha256_file(candidate) if candidate.is_file() else None

    def audit(self, request: DORControlPlaneAuditRequest) -> DORControlPlaneAudit:
        manifest_hash = self._optional_hash(request.dataset_manifest_path)
        diagnostic_request_hash = self._optional_hash(request.diagnostic_request_path)
        data_hash = self._optional_hash(request.data_path)
        runtime: DORRuntimeIdentity | None = None
        status = DORControlPlaneStatus.ERROR
        reason = "DOR control-plane audit did not run"
        task: str | None = None
        manifest_contract: str | None = None
        plan_contract: str | None = None
        plan_id: str | None = None
        qualification: dict[str, Any] | None = None
        admissions: list[dict[str, Any]] = []
        reference_summary: dict[str, Any] | None = None
        reference_safety: dict[str, Any] | None = None
        declared_reference_scope: dict[str, Any] | None = None
        try:
            manifest_path = Path(request.dataset_manifest_path)
            diagnostic_request_path = Path(request.diagnostic_request_path)
            data_path = Path(request.data_path)
            for name, path in (
                ("dataset manifest", manifest_path),
                ("diagnostic request", diagnostic_request_path),
                ("data", data_path),
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"DOR {name} file not found: {path}")
            runtime = inspect_dor_runtime(request.executable_path, timeout_s=request.timeout_s)
            with tempfile.TemporaryDirectory(prefix="astrograph-dor-control-") as temp_dir:
                output_dir = Path(temp_dir) / "plan"
                completed = run_managed_subprocess(
                    [
                        runtime.executable,
                        "plan",
                        "--manifest",
                        str(manifest_path),
                        "--request",
                        str(diagnostic_request_path),
                        "--data",
                        str(data_path),
                        "--out",
                        str(output_dir),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=request.timeout_s,
                    check=False,
                )
                if completed.returncode not in {0, 2}:
                    detail = (completed.stderr or completed.stdout or "no subprocess output").strip()[-2000:]
                    raise RuntimeError(f"DOR plan exited {completed.returncode}: {detail}")
                effective_manifest = self._load_json(output_dir / "dataset_manifest.json")
                qualification = self._load_json(output_dir / "task_qualification.json")
                admissions_raw = json.loads((output_dir / "engine_admissions.json").read_text(encoding="utf-8"))
                if not isinstance(admissions_raw, list) or not all(isinstance(item, dict) for item in admissions_raw):
                    raise ValueError("DOR engine_admissions.json must contain a list of objects")
                admissions = admissions_raw
                plan = self._load_json(output_dir / "diagnostic_plan.json")
                reference = self._load_json(output_dir / "reference_applicability.json")

            declared_reference_scope = dict(effective_manifest.get("reference_scope") or {})
            if request.expected_reference_scope is not None and (
                declared_reference_scope != request.expected_reference_scope
            ):
                raise ValueError(
                    "DOR manifest reference_scope does not match the resolved Dataset Profile"
                )

            manifest_contract = str(effective_manifest.get("contract") or "")
            plan_contract = str(plan.get("contract") or "")
            if manifest_contract != "DOR-DATASET-MANIFEST/0.4":
                raise ValueError(f"unexpected DOR dataset manifest contract: {manifest_contract}")
            if plan_contract != "DOR-DIAGNOSTIC-PLAN/0.4":
                raise ValueError(f"unexpected DOR diagnostic plan contract: {plan_contract}")
            if plan.get("qualification") != qualification:
                raise ValueError("DOR task qualification artifact disagrees with diagnostic plan")
            if plan.get("engine_admissions") != admissions:
                raise ValueError("DOR engine admission artifact disagrees with diagnostic plan")
            if reference.get("contract") != "DOR-REFERENCE-APPLICABILITY/0.4":
                raise ValueError(f"unexpected DOR reference contract: {reference.get('contract')}")
            reference_safety = dict(reference.get("safety") or {})
            unsafe = {
                "test_labels_used": False,
                "test_scores_used": False,
                "implicit_pooling": False,
            }
            for key, expected in unsafe.items():
                if reference_safety.get(key) is not expected:
                    raise ValueError(f"DOR reference safety invariant violated: {key}")

            raw_status = str(qualification.get("status") or "")
            status = DORControlPlaneStatus(raw_status)
            task = str(plan.get("task") or qualification.get("task") or "") or None
            plan_id = str(plan.get("plan_id") or "") or None
            reference_summary = dict(reference.get("summary") or {})
            reasons = [str(item) for item in qualification.get("reasons") or []]
            warnings = [str(item) for item in qualification.get("warnings") or []]
            if status == DORControlPlaneStatus.QUALIFIED:
                reason = "DOR qualified the declared dataset roles, task, leakage guard, and reference policy."
            elif status == DORControlPlaneStatus.CONDITIONALLY_QUALIFIED:
                reason = "; ".join(warnings) or "DOR returned conditional qualification."
            else:
                reason = "; ".join(reasons) or "DOR rejected the declared diagnostic task."
        except Exception as exc:
            status = DORControlPlaneStatus.ERROR
            reason = f"{type(exc).__name__}: {exc}"

        eligible = status == DORControlPlaneStatus.QUALIFIED or (
            status == DORControlPlaneStatus.CONDITIONALLY_QUALIFIED
            and request.allow_conditionally_qualified
        )
        identity = {
            "manifest_sha256": manifest_hash,
            "manifest_path": request.dataset_manifest_path,
            "diagnostic_request_sha256": diagnostic_request_hash,
            "diagnostic_request_path": request.diagnostic_request_path,
            "data_sha256": data_hash,
            "data_path": request.data_path,
            "runtime_sha256": runtime.executable_sha256 if runtime else None,
            "runtime_source_sha256": runtime.distribution_source_sha256 if runtime else None,
            "dor_version": runtime.dor_version if runtime else None,
            "status": status.value,
            "task": task,
            "plan_id": plan_id,
            "declared_reference_scope": declared_reference_scope,
        }
        return DORControlPlaneAudit(
            audit_id=stable_id("DOR-CP", identity),
            status=status,
            reason=reason,
            task=task,
            release_gate_enabled=request.release_gate,
            allow_conditionally_qualified=request.allow_conditionally_qualified,
            eligible_for_diagnostic_execution=eligible,
            dataset_manifest_sha256=manifest_hash,
            diagnostic_request_sha256=diagnostic_request_hash,
            data_sha256=data_hash,
            runtime=runtime,
            bridge_version=self.bridge_version,
            manifest_contract=manifest_contract,
            plan_contract=plan_contract,
            plan_id=plan_id,
            task_qualification=qualification,
            engine_admissions=admissions,
            reference_applicability_summary=reference_summary,
            reference_safety=reference_safety,
            declared_reference_scope=declared_reference_scope,
        )


def control_plane_release_blockers(audit: DORControlPlaneAudit | None) -> list[str]:
    if audit is None or not audit.release_gate_enabled or audit.eligible_for_diagnostic_execution:
        return []
    return [f"DOR_CONTROL_PLANE_{audit.status.value}"]
