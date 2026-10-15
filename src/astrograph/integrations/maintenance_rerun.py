from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.oag.contracts import OAGInput


READINESS_SCHEMA_VERSION = "astrograph-maintenance-governed-rerun-readiness.v1"
RERUN_RECEIPT_SCHEMA_VERSION = "astrograph-maintenance-governed-rerun-receipt.v1"
RERUN_COMPLETION_SCHEMA_VERSION = "astrograph-maintenance-governed-rerun-completion-receipt.v1"
RERUN_MODE = "OAG_EVIDENCE_BRANCH_REPLAY"


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class MaintenanceGovernedRerunReadiness(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[READINESS_SCHEMA_VERSION] = READINESS_SCHEMA_VERSION
    status: Literal["READY", "BLOCKED"]
    rerun_mode: Literal[RERUN_MODE] = RERUN_MODE
    source_artifact_kind: str
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_oag_input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_report_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_bundle_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    overlay_bundle_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    rerun_input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    supplemental_receipt_sha256s: list[str] = Field(default_factory=list)
    supplemental_provenance_receipt_sha256s: list[str] = Field(default_factory=list)
    raw_telemetry_rerun_performed: Literal[False] = False
    formal_release_eligible: Literal[False] = False
    claim_boundary: str

    @model_validator(mode="after")
    def consistent(self) -> "MaintenanceGovernedRerunReadiness":
        if self.status == "READY" and self.blockers:
            raise ValueError("READY rerun readiness cannot contain blockers")
        if self.status == "BLOCKED" and not self.blockers:
            raise ValueError("BLOCKED rerun readiness requires blockers")
        return self


class MaintenanceGovernedRerunReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[RERUN_RECEIPT_SCHEMA_VERSION] = RERUN_RECEIPT_SCHEMA_VERSION
    status: Literal["PREPARED"] = "PREPARED"
    rerun_mode: Literal[RERUN_MODE] = RERUN_MODE
    source_job_id: str
    child_job_id: str
    actor: str
    created_at: str
    source_artifact_kind: str
    source_oag_input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_report_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_bundle_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    overlay_bundle_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    rerun_input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    supplemental_receipt_sha256s: list[str] = Field(default_factory=list)
    supplemental_provenance_receipt_sha256s: list[str] = Field(default_factory=list)
    child_preflight_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    child_preflight_status: str | None = None
    child_can_start: bool
    original_report_mutated: Literal[False] = False
    raw_telemetry_rerun_performed: Literal[False] = False
    formal_release_eligible: Literal[False] = False
    claim_boundary: str
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_receipt(self) -> "MaintenanceGovernedRerunReceipt":
        if not self.child_can_start:
            raise ValueError("prepared governed rerun receipt requires a startable child preflight")
        for sequence in (self.supplemental_receipt_sha256s, self.supplemental_provenance_receipt_sha256s):
            if len(sequence) != len(set(sequence)):
                raise ValueError("governed rerun lineage hashes must be unique")
            if any(len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value) for value in sequence):
                raise ValueError("invalid governed rerun lineage sha256")
        core = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if canonical_sha256(core) != self.receipt_sha256:
            raise ValueError("governed rerun receipt_sha256 mismatch")
        return self


class MaintenanceGovernedRerunCompletionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[RERUN_COMPLETION_SCHEMA_VERSION] = RERUN_COMPLETION_SCHEMA_VERSION
    status: Literal["COMPLETED"] = "COMPLETED"
    rerun_mode: Literal[RERUN_MODE] = RERUN_MODE
    source_job_id: str
    child_job_id: str
    completed_at: str
    prepared_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_oag_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rerun_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    child_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    child_oag_output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    child_diagnosis_status: str
    child_selected_fault_id: str | None = None
    child_release_status: Literal["FORMAL_RELEASE_BLOCKED"] = "FORMAL_RELEASE_BLOCKED"
    source_report_unchanged: Literal[True] = True
    source_oag_input_unchanged: Literal[True] = True
    actual_rerun_input_matches_prepared: Literal[True] = True
    original_report_mutated: Literal[False] = False
    raw_telemetry_rerun_performed: Literal[False] = False
    formal_release_eligible: Literal[False] = False
    claim_boundary: str
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_receipt(self) -> "MaintenanceGovernedRerunCompletionReceipt":
        core = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if canonical_sha256(core) != self.receipt_sha256:
            raise ValueError("governed rerun completion receipt_sha256 mismatch")
        return self


def _same_frozen_evidence_contract(original: EpisodeEvidenceBundle, overlay: EpisodeEvidenceBundle) -> bool:
    return (
        original.episode.model_dump(mode="json") == overlay.episode.model_dump(mode="json")
        and [item.model_dump(mode="json") for item in original.evidence_definitions]
        == [item.model_dump(mode="json") for item in overlay.evidence_definitions]
        and [item.model_dump(mode="json") for item in original.feature_definitions]
        == [item.model_dump(mode="json") for item in overlay.feature_definitions]
        and original.analysis_component_instance == overlay.analysis_component_instance
        and original.analysis_wheel_index == overlay.analysis_wheel_index
        and original.selected_wheel_index == overlay.selected_wheel_index
    )


def _build_candidate_input(original: OAGInput, overlay: EpisodeEvidenceBundle) -> OAGInput:
    payload = original.model_dump(mode="json")
    payload["bundle"] = overlay.model_dump(mode="json")
    # Do not rewrite a native evidence projection.  It binds the original bundle,
    # feature hashes and raw paired-frame model input; fabricating a new projection
    # from a maintenance capsule would falsely claim a telemetry-level rerun.
    return OAGInput.model_validate(payload)


def assess_governed_oag_rerun(
    *,
    source_job_id: str | None,
    source_artifact_kind: str,
    original_input: OAGInput | None,
    overlay_bundle: EpisodeEvidenceBundle | None,
    source_report: dict[str, Any] | None,
    source_oag_input_sha256: str | None,
    supplemental_receipt_sha256s: list[str],
    supplemental_provenance_receipts: list[dict[str, Any]],
) -> MaintenanceGovernedRerunReadiness:
    blockers: list[str] = []
    warnings: list[str] = []
    rerun_input_sha256: str | None = None
    source_bundle_sha256: str | None = None
    overlay_bundle_sha256: str | None = None

    if source_artifact_kind != "OAG_DIAGNOSTIC":
        blockers.append("SOURCE_ARTIFACT_REQUIRES_RAW_INPUT_RERUN")
    if original_input is None:
        blockers.append("SOURCE_OAG_INPUT_UNAVAILABLE")
    if overlay_bundle is None:
        blockers.append("NO_SUPPLEMENTAL_EVIDENCE_OVERLAY")
    if not supplemental_receipt_sha256s:
        blockers.append("NO_SUPPLEMENTAL_ADMISSION_RECEIPT")
    if not supplemental_provenance_receipts:
        blockers.append("NO_SUPPLEMENTAL_PROVENANCE_RECEIPT")

    verified_provenance: list[dict[str, Any]] = []
    if supplemental_provenance_receipts:
        from astrograph.governance.supplemental_provenance import verify_supplemental_input_provenance_receipt
        for payload in supplemental_provenance_receipts:
            try:
                verified_provenance.append(
                    verify_supplemental_input_provenance_receipt(payload).model_dump(mode="json")
                )
            except Exception:
                blockers.append("INVALID_SUPPLEMENTAL_PROVENANCE_RECEIPT")

    source_report_sha256 = canonical_sha256(source_report) if source_report is not None else None
    original_bundle_sha256 = (
        canonical_sha256(original_input.bundle.model_dump(mode="json")) if original_input is not None else None
    )
    if verified_provenance:
        by_admission: dict[str, dict[str, Any]] = {}
        for receipt in verified_provenance:
            admission_sha = str(receipt.get("admission_receipt_sha256") or "")
            if admission_sha in by_admission:
                blockers.append("DUPLICATE_SUPPLEMENTAL_PROVENANCE_FOR_ADMISSION")
            by_admission[admission_sha] = receipt
            if source_job_id and receipt.get("source_job_id") != source_job_id:
                blockers.append("SUPPLEMENTAL_PROVENANCE_SOURCE_JOB_MISMATCH")
            if source_report_sha256 and receipt.get("source_report_sha256") != source_report_sha256:
                blockers.append("SUPPLEMENTAL_PROVENANCE_SOURCE_REPORT_MISMATCH")
            if original_bundle_sha256 and receipt.get("original_evidence_bundle_sha256") != original_bundle_sha256:
                blockers.append("SUPPLEMENTAL_PROVENANCE_ORIGINAL_BUNDLE_MISMATCH")
        if set(by_admission) != set(supplemental_receipt_sha256s):
            blockers.append("SUPPLEMENTAL_PROVENANCE_CHAIN_COVERAGE_MISMATCH")
        for index, admission_sha in enumerate(supplemental_receipt_sha256s):
            receipt = by_admission.get(admission_sha)
            if receipt is None:
                continue
            if list(receipt.get("parent_admission_receipt_sha256s") or []) != supplemental_receipt_sha256s[:index]:
                blockers.append("SUPPLEMENTAL_PROVENANCE_PARENT_CHAIN_MISMATCH")

    if original_input is not None:
        source_bundle_sha256 = canonical_sha256(original_input.bundle.model_dump(mode="json"))
        three_task = original_input.three_task_result
        if three_task is not None and three_task.evidence_projection is not None:
            blockers.append("NATIVE_EVIDENCE_PROJECTION_BOUND_TO_ORIGINAL_BUNDLE")
            warnings.append(
                "The upstream native projection binds the original paired telemetry/model input. "
                "A maintenance capsule cannot lawfully rewrite that projection; rerun from raw telemetry instead."
            )
    if overlay_bundle is not None:
        overlay_bundle_sha256 = canonical_sha256(overlay_bundle.model_dump(mode="json"))

    if original_input is not None and overlay_bundle is not None:
        if not _same_frozen_evidence_contract(original_input.bundle, overlay_bundle):
            blockers.append("FROZEN_EVIDENCE_CONTRACT_DRIFT")
        if not blockers:
            try:
                candidate = _build_candidate_input(original_input, overlay_bundle)
                rerun_input_sha256 = canonical_sha256(candidate.model_dump(mode="json"))
            except Exception as exc:
                blockers.append("RERUN_OAG_INPUT_CONTRACT_REJECTED")
                warnings.append(f"{type(exc).__name__}:{exc}")

    return MaintenanceGovernedRerunReadiness(
        status="BLOCKED" if blockers else "READY",
        source_artifact_kind=source_artifact_kind or "UNKNOWN",
        blockers=sorted(set(blockers)),
        warnings=warnings,
        source_oag_input_sha256=source_oag_input_sha256,
        source_report_sha256=source_report_sha256,
        source_bundle_sha256=source_bundle_sha256,
        overlay_bundle_sha256=overlay_bundle_sha256,
        rerun_input_sha256=rerun_input_sha256,
        supplemental_receipt_sha256s=list(supplemental_receipt_sha256s),
        supplemental_provenance_receipt_sha256s=[
            str(item.get("receipt_sha256")) for item in verified_provenance if item.get("receipt_sha256")
        ],
        claim_boundary=(
            "READY means AstroGraph can create a new immutable OAG structured-evidence child job whose evidence branch is the admitted maintenance overlay. "
            "It is not a raw-telemetry production rerun, does not rebuild upstream model outputs, and remains review-only/formal_release_eligible=false. "
            "PUBLIC_CASE, PIPELINE_RUNTIME, or native evidence-projection cases must return to their raw governed input path."
        ),
    )


def build_governed_oag_rerun_input(
    *, original_input: OAGInput, overlay_bundle: EpisodeEvidenceBundle,
) -> OAGInput:
    if original_input.three_task_result is not None and original_input.three_task_result.evidence_projection is not None:
        raise ValueError("governed OAG rerun is blocked: NATIVE_EVIDENCE_PROJECTION_BOUND_TO_ORIGINAL_BUNDLE")
    if not _same_frozen_evidence_contract(original_input.bundle, overlay_bundle):
        raise ValueError("governed OAG rerun is blocked: FROZEN_EVIDENCE_CONTRACT_DRIFT")
    return _build_candidate_input(original_input, overlay_bundle)


def build_governed_rerun_receipt(
    *,
    source_job_id: str,
    child_job_id: str,
    actor: str,
    created_at: str,
    readiness: MaintenanceGovernedRerunReadiness,
    child_preflight: dict[str, Any],
    supplemental_provenance_receipt_sha256s: list[str],
) -> dict[str, Any]:
    if readiness.status != "READY":
        raise ValueError("cannot build governed rerun receipt from blocked readiness")
    if list(supplemental_provenance_receipt_sha256s) != list(readiness.supplemental_provenance_receipt_sha256s):
        raise ValueError("governed rerun receipt provenance lineage differs from readiness")
    core = {
        "schema_version": RERUN_RECEIPT_SCHEMA_VERSION,
        "status": "PREPARED",
        "rerun_mode": RERUN_MODE,
        "source_job_id": source_job_id,
        "child_job_id": child_job_id,
        "actor": actor,
        "created_at": created_at,
        "source_artifact_kind": readiness.source_artifact_kind,
        "source_oag_input_sha256": readiness.source_oag_input_sha256,
        "source_report_sha256": readiness.source_report_sha256,
        "source_bundle_sha256": readiness.source_bundle_sha256,
        "overlay_bundle_sha256": readiness.overlay_bundle_sha256,
        "rerun_input_sha256": readiness.rerun_input_sha256,
        "supplemental_receipt_sha256s": readiness.supplemental_receipt_sha256s,
        "supplemental_provenance_receipt_sha256s": list(supplemental_provenance_receipt_sha256s),
        "child_preflight_sha256": canonical_sha256(child_preflight),
        "child_preflight_status": child_preflight.get("status"),
        "child_can_start": bool(child_preflight.get("can_start")),
        "original_report_mutated": False,
        "raw_telemetry_rerun_performed": False,
        "formal_release_eligible": False,
        "claim_boundary": readiness.claim_boundary,
    }
    return MaintenanceGovernedRerunReceipt.model_validate(
        {**core, "receipt_sha256": canonical_sha256(core)}
    ).model_dump(mode="json")


def build_governed_rerun_completion_receipt(
    *,
    prepared_receipt: dict[str, Any],
    source_report: dict[str, Any],
    source_oag_input_sha256: str,
    actual_rerun_input: OAGInput,
    child_report: dict[str, Any],
    completed_at: str,
) -> dict[str, Any]:
    prepared = verify_governed_rerun_receipt(prepared_receipt)
    if canonical_sha256(source_report) != prepared.source_report_sha256:
        raise ValueError("governed rerun source report changed after preparation")
    if source_oag_input_sha256 != prepared.source_oag_input_sha256:
        raise ValueError("governed rerun source OAG input changed after preparation")
    actual_input_sha256 = canonical_sha256(actual_rerun_input.model_dump(mode="json"))
    if actual_input_sha256 != prepared.rerun_input_sha256:
        raise ValueError("governed rerun child input differs from prepared rerun input")
    if str(child_report.get("job_id") or "") != prepared.child_job_id:
        raise ValueError("governed rerun child report job_id differs from prepared receipt")
    if bool(child_report.get("formal_diagnosis_publishable")):
        raise ValueError("governed evidence-branch rerun cannot become formally publishable")
    rerun_link = child_report.get("rerun_provenance") or {}
    if str(rerun_link.get("source_job_id") or "") != prepared.source_job_id:
        raise ValueError("governed rerun child report lost source-job lineage")
    if str(rerun_link.get("receipt_sha256") or "") != prepared.receipt_sha256:
        raise ValueError("governed rerun child report lost prepared-receipt lineage")

    oag_result = ((child_report.get("analysis") or {}).get("oag_result") or {})
    output = oag_result.get("output") or {}
    summary = output.get("summary") or {}
    assessment = output.get("assessment") or {}
    selected_fault_id = (
        summary.get("suspected_fault_id")
        or assessment.get("selected_fault_id")
        or None
    )
    child_oag_output_sha256 = canonical_sha256(output) if output else None
    core = {
        "schema_version": RERUN_COMPLETION_SCHEMA_VERSION,
        "status": "COMPLETED",
        "rerun_mode": RERUN_MODE,
        "source_job_id": prepared.source_job_id,
        "child_job_id": prepared.child_job_id,
        "completed_at": completed_at,
        "prepared_receipt_sha256": prepared.receipt_sha256,
        "source_report_sha256": prepared.source_report_sha256,
        "source_oag_input_sha256": prepared.source_oag_input_sha256,
        "rerun_input_sha256": prepared.rerun_input_sha256,
        "child_report_sha256": canonical_sha256(child_report),
        "child_oag_output_sha256": child_oag_output_sha256,
        "child_diagnosis_status": str(summary.get("status") or assessment.get("conclusion_status") or "UNKNOWN"),
        "child_selected_fault_id": selected_fault_id,
        "child_release_status": "FORMAL_RELEASE_BLOCKED",
        "source_report_unchanged": True,
        "source_oag_input_unchanged": True,
        "actual_rerun_input_matches_prepared": True,
        "original_report_mutated": False,
        "raw_telemetry_rerun_performed": False,
        "formal_release_eligible": False,
        "claim_boundary": (
            "This completion receipt proves that the governed OAG evidence-branch child executed against the exact prepared input while the frozen source report and source OAG input remained unchanged. "
            "The child produced a new review result and release decision, but this branch did not rerun raw telemetry or upstream native model projections and therefore remains formal_release_eligible=false."
        ),
    }
    return MaintenanceGovernedRerunCompletionReceipt.model_validate(
        {**core, "receipt_sha256": canonical_sha256(core)}
    ).model_dump(mode="json")


def verify_governed_rerun_completion_receipt(payload: dict[str, Any]) -> MaintenanceGovernedRerunCompletionReceipt:
    return MaintenanceGovernedRerunCompletionReceipt.model_validate(payload)


def verify_governed_rerun_receipt(payload: dict[str, Any]) -> MaintenanceGovernedRerunReceipt:
    return MaintenanceGovernedRerunReceipt.model_validate(payload)


__all__ = [
    "READINESS_SCHEMA_VERSION",
    "RERUN_RECEIPT_SCHEMA_VERSION",
    "RERUN_MODE",
    "RERUN_COMPLETION_SCHEMA_VERSION",
    "MaintenanceGovernedRerunReadiness",
    "MaintenanceGovernedRerunReceipt",
    "MaintenanceGovernedRerunCompletionReceipt",
    "assess_governed_oag_rerun",
    "build_governed_oag_rerun_input",
    "build_governed_rerun_receipt",
    "build_governed_rerun_completion_receipt",
    "verify_governed_rerun_receipt",
    "verify_governed_rerun_completion_receipt",
    "canonical_sha256",
]
