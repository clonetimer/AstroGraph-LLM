from __future__ import annotations

"""Fail-closed bridge from frozen FEC references to AstroGraph model evidence.

This module only applies an independently commissioned reference.  It never
fits, refreshes, or selects a calibration reference during online diagnosis.
"""

from .process_control import run_managed_subprocess
import csv
import hashlib
import json
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.data_driven import DataDrivenPrediction
from astrograph.dataset_profiles import DatasetReferenceScope
from astrograph.evidence.executable import stable_id
from astrograph.integrations.external_runtime import inspect_python_cli_runtime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_sha256(prediction: DataDrivenPrediction) -> str:
    payload = json.dumps(
        prediction.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class FECEvidenceMode(str, Enum):
    CLASS = "CLASS"
    CONTEXT_CLASS = "CONTEXT_CLASS"
    ANOMALY = "ANOMALY"


class FECEvidenceStatus(str, Enum):
    QUALIFIED = "QUALIFIED"
    NORMAL = "NORMAL"
    ABSTAIN = "ABSTAIN"
    ERROR = "ERROR"


class FECRuntimeIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-fec-runtime-identity.v1"
    executable: str = Field(min_length=1)
    executable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fec_version: str = Field(min_length=1)
    package_source: str
    package_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    package_source_hash_scope: str = "fec-package-python-files"


class FECEvidenceAuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-fec-evidence-audit-request.v2"
    reference_path: str = Field(min_length=1)
    expected_reference_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    expected_source_score_contract_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
    )
    mode: FECEvidenceMode
    executable_path: str = "fec"
    context: dict[str, str] = Field(default_factory=dict)
    modality_mask: dict[str, bool] = Field(default_factory=dict)
    timeout_s: float = Field(default=30.0, gt=0.0, le=300.0)
    release_gate_enabled: bool = True
    expected_reference_scope: DatasetReferenceScope | None = None
    fleet_commissioning_receipt_path: str | None = None
    expected_fleet_commissioning_receipt_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
    )


class FECFleetCommissioningReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-fec-fleet-commissioning-receipt.v1"
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fec_decision: "FECFleetCommissioningDecision"
    claim_boundary: str = Field(min_length=1)


class FECFleetAssetValidation(BaseModel):
    """Serialized FEC FleetReferenceAssetValidation output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str = Field(min_length=1)
    accepted: bool
    state: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    supported_fraction: float = Field(ge=0.0, le=1.0)
    health: dict[str, Any] | None = None
    sentinel_data_ids: list[str] = Field(min_length=1)


class FECFleetCommissioningDecision(BaseModel):
    """Serialized FEC FleetReferenceCommissioningDecision output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fleet_group_id: str = Field(min_length=1)
    bundle_id: str = Field(min_length=1)
    accepted: bool
    state: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    required_asset_ids: list[str] = Field(min_length=2)
    asset_validations: list[FECFleetAssetValidation] = Field(min_length=2)
    power_limited: bool
    promotion_guard_id: str | None = None
    planned_variant_count: int | None = Field(default=None, ge=1)
    promotion_pending: bool = False


class FECEvidenceAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-fec-evidence-audit.v2"
    audit_id: str
    status: FECEvidenceStatus
    mode: FECEvidenceMode
    prediction_id: str
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_path: str
    reference_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reference_contract: str | None = None
    reference_metadata: dict[str, Any] = Field(default_factory=dict)
    declared_reference_scope: DatasetReferenceScope | None = None
    fleet_commissioning_receipt_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    runtime: FECRuntimeIdentity | None = None
    calibration_supported: bool = False
    eligible_for_model_evidence: bool = False
    prediction_set: list[str] = Field(default_factory=list)
    anomaly_decision: str | None = None
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_n_reference: int | None = Field(default=None, ge=0)
    calibration_reason: str | None = None
    context: dict[str, str] = Field(default_factory=dict)
    modality_mask: dict[str, bool] = Field(default_factory=dict)
    release_gate_enabled: bool = True
    errors: list[str] = Field(default_factory=list)
    raw_result: dict[str, Any] = Field(default_factory=dict)
    claim_boundary: str = (
        "FEC establishes reference-backed statistical support or abstention only. "
        "It does not establish physical root cause, propagation, or operational authority."
    )


class FECEvidenceAuditService:
    _CONTRACTS = {
        FECEvidenceMode.CLASS: "FEC-CLASS-CONFORMAL-REFERENCE/1.0",
        FECEvidenceMode.CONTEXT_CLASS: "FEC-CONTEXT-CLASS-CONFORMAL-REFERENCE/1.0",
        FECEvidenceMode.ANOMALY: "FEC-ANOMALY-CONTEXT-REFERENCE/1.0",
    }
    _COMMANDS = {
        FECEvidenceMode.CLASS: "class-infer",
        FECEvidenceMode.CONTEXT_CLASS: "class-context-infer",
        FECEvidenceMode.ANOMALY: "anomaly-infer",
    }

    @staticmethod
    def _runtime(executable: str, timeout_s: float) -> FECRuntimeIdentity:
        identity = inspect_python_cli_runtime(
            executable,
            distribution="fec-toolkit",
            import_name="fec",
            timeout_s=timeout_s,
        )
        return FECRuntimeIdentity(
            executable=identity.executable,
            executable_sha256=identity.executable_sha256,
            fec_version=identity.version,
            package_source=identity.package_source,
            package_source_sha256=identity.package_source_sha256,
        )

    @staticmethod
    def _validate_binding(
        prediction: DataDrivenPrediction,
        request: FECEvidenceAuditRequest,
        reference: dict[str, Any],
        reference_sha256: str,
    ) -> tuple[DatasetReferenceScope | None, str | None]:
        expected_contract = FECEvidenceAuditService._CONTRACTS[request.mode]
        if reference.get("contract") != expected_contract:
            raise ValueError(
                f"reference contract {reference.get('contract')!r} does not match {expected_contract!r}"
            )
        if request.expected_source_score_contract_sha256 is not None:
            actual_score_contract = reference.get("source_score_contract_sha256")
            if str(actual_score_contract or "").lower() != (
                request.expected_source_score_contract_sha256.lower()
            ):
                raise ValueError("FEC source score contract SHA-256 mismatch")
        raw_scope = reference.get("astrograph_reference_scope")
        declared_scope = (
            DatasetReferenceScope.model_validate(raw_scope)
            if raw_scope is not None
            else None
        )
        receipt_sha: str | None = None
        if request.expected_reference_scope is not None:
            if declared_scope is None:
                raise ValueError("FEC reference lacks the Dataset Profile reference scope binding")
            if declared_scope != request.expected_reference_scope:
                raise ValueError("FEC reference scope does not match the resolved Dataset Profile")
            if declared_scope.kind == "FLEET_GROUP":
                if (
                    not request.fleet_commissioning_receipt_path
                    or not request.expected_fleet_commissioning_receipt_sha256
                ):
                    raise ValueError(
                        "FLEET_GROUP reference requires a hash-bound fleet commissioning receipt"
                    )
                receipt_path = Path(request.fleet_commissioning_receipt_path).resolve()
                if not receipt_path.is_file():
                    raise FileNotFoundError(
                        f"FEC fleet commissioning receipt not found: {receipt_path}"
                    )
                receipt_sha = _sha256(receipt_path)
                if receipt_sha != request.expected_fleet_commissioning_receipt_sha256.lower():
                    raise ValueError("FEC fleet commissioning receipt SHA-256 mismatch")
                receipt = FECFleetCommissioningReceipt.model_validate_json(
                    receipt_path.read_text(encoding="utf-8")
                )
                decision = receipt.fec_decision
                expected_members = set(declared_scope.member_asset_ids)
                if not decision.accepted or decision.promotion_pending:
                    raise ValueError(
                        "FEC fleet commissioning decision is not accepted for promotion"
                    )
                if decision.fleet_group_id != declared_scope.fleet_group_id:
                    raise ValueError("FEC fleet commissioning receipt group mismatch")
                if set(decision.required_asset_ids) != expected_members:
                    raise ValueError("FEC fleet commissioning receipt member mismatch")
                if receipt.reference_sha256 != reference_sha256:
                    raise ValueError("FEC fleet commissioning receipt reference SHA-256 mismatch")
                validations = {item.asset_id: item for item in decision.asset_validations}
                if (
                    len(validations) != len(decision.asset_validations)
                    or set(validations) != expected_members
                    or any(
                        not item.accepted or not item.sentinel_data_ids
                        for item in validations.values()
                    )
                ):
                    raise ValueError(
                        "FEC fleet commissioning receipt requires one accepted independent sentinel audit per asset"
                    )
            elif (
                request.fleet_commissioning_receipt_path is not None
                or request.expected_fleet_commissioning_receipt_sha256 is not None
            ):
                raise ValueError("ASSET_LOCAL reference cannot carry a fleet commissioning receipt")
        elif (
            request.fleet_commissioning_receipt_path is not None
            or request.expected_fleet_commissioning_receipt_sha256 is not None
        ):
            raise ValueError(
                "fleet commissioning receipt requires an expected Dataset Profile reference scope"
            )
        expected_context = tuple(map(str, reference.get("context_columns") or ()))
        if set(request.context) != set(expected_context):
            raise ValueError(
                f"context keys {sorted(request.context)} do not match reference {sorted(expected_context)}"
            )
        if request.mode == FECEvidenceMode.ANOMALY:
            if prediction.task_type != "ANOMALY_ATTRIBUTION":
                raise ValueError("ANOMALY mode requires an ANOMALY_ATTRIBUTION prediction")
            if prediction.normal_probability is None:
                raise ValueError("ANOMALY mode requires a normal class probability")
            if request.modality_mask:
                raise ValueError("ANOMALY reference does not accept modality masks")
            return declared_scope, receipt_sha
        if prediction.output_semantics != "PROBABILITIES":
            raise ValueError("class conformal inference requires genuine probability semantics")
        class_frame = tuple(map(str, reference.get("class_frame") or ()))
        prediction_frame = tuple(item.class_id for item in prediction.classes)
        if set(prediction_frame) != set(class_frame):
            raise ValueError(
                f"prediction class frame {sorted(prediction_frame)} does not match reference {sorted(class_frame)}"
            )
        expected_masks = tuple(map(str, reference.get("mask_columns") or ()))
        if set(request.modality_mask) != set(expected_masks):
            raise ValueError(
                f"modality mask keys {sorted(request.modality_mask)} do not match reference {sorted(expected_masks)}"
            )
        return declared_scope, receipt_sha

    @staticmethod
    def _input_row(
        prediction: DataDrivenPrediction,
        request: FECEvidenceAuditRequest,
        reference: dict[str, Any],
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "prediction_id": prediction.prediction_id,
            "episode_id": prediction.episode_id,
            **request.context,
        }
        if request.mode == FECEvidenceMode.ANOMALY:
            row["anomaly_score"] = 1.0 - float(prediction.normal_probability)
            return row
        probabilities = {item.class_id: item.probability for item in prediction.classes}
        for class_id in reference["class_frame"]:
            row[f"score__{class_id}"] = probabilities[str(class_id)]
        for column in reference.get("mask_columns") or ():
            row[str(column)] = int(request.modality_mask[str(column)])
        return row

    @staticmethod
    def _reference_metadata(reference: dict[str, Any]) -> dict[str, Any]:
        groups = reference.get("groups") or []
        return {
            "alpha": reference.get("alpha"),
            "min_group_size": reference.get("min_group_size"),
            "provenance": reference.get("provenance"),
            "calibration_score_sha256": reference.get("calibration_score_sha256"),
            "source_score_contract_sha256": reference.get("source_score_contract_sha256"),
            "class_frame": list(reference.get("class_frame") or []),
            "context_columns": list(reference.get("context_columns") or []),
            "mask_columns": list(reference.get("mask_columns") or []),
            "astrograph_reference_scope": reference.get("astrograph_reference_scope"),
            "group_count": len(groups),
            "supported_group_count": sum(
                1 for group in groups if group.get("supported", True) is True
            ),
        }

    def audit(
        self,
        prediction: DataDrivenPrediction,
        request: FECEvidenceAuditRequest,
    ) -> FECEvidenceAudit:
        prediction_sha = _prediction_sha256(prediction)
        reference_path = Path(request.reference_path).resolve()
        common = {
            "audit_id": stable_id(
                "FEC-AUDIT",
                {
                    "prediction_sha256": prediction_sha,
                    "reference_path": str(reference_path),
                    "expected_reference_sha256": request.expected_reference_sha256.lower(),
                    "expected_source_score_contract_sha256": (
                        request.expected_source_score_contract_sha256.lower()
                        if request.expected_source_score_contract_sha256
                        else None
                    ),
                    "mode": request.mode.value,
                    "context": request.context,
                    "modality_mask": request.modality_mask,
                    "expected_reference_scope": (
                        request.expected_reference_scope.model_dump(mode="json")
                        if request.expected_reference_scope
                        else None
                    ),
                    "fleet_commissioning_receipt_path": (
                        str(Path(request.fleet_commissioning_receipt_path).resolve())
                        if request.fleet_commissioning_receipt_path
                        else None
                    ),
                    "expected_fleet_commissioning_receipt_sha256": (
                        request.expected_fleet_commissioning_receipt_sha256.lower()
                        if request.expected_fleet_commissioning_receipt_sha256
                        else None
                    ),
                },
            ),
            "mode": request.mode,
            "prediction_id": prediction.prediction_id,
            "prediction_sha256": prediction_sha,
            "reference_path": str(reference_path),
            "context": request.context,
            "modality_mask": request.modality_mask,
            "release_gate_enabled": request.release_gate_enabled,
        }
        try:
            if not reference_path.is_file():
                raise FileNotFoundError(f"FEC reference not found: {reference_path}")
            reference_sha = _sha256(reference_path)
            if reference_sha != request.expected_reference_sha256.lower():
                raise ValueError(
                    f"FEC reference SHA-256 mismatch: expected {request.expected_reference_sha256.lower()}, got {reference_sha}"
                )
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            declared_scope, receipt_sha = self._validate_binding(
                prediction,
                request,
                reference,
                reference_sha,
            )
            runtime = self._runtime(request.executable_path, request.timeout_s)
            row = self._input_row(prediction, request, reference)
            with tempfile.TemporaryDirectory(prefix="astrograph-fec-") as directory:
                root = Path(directory)
                score_csv = root / "scores.csv"
                output_csv = root / "result.csv"
                with score_csv.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
                completed = run_managed_subprocess(
                    [
                        runtime.executable,
                        self._COMMANDS[request.mode],
                        "--scores",
                        str(score_csv),
                        "--reference",
                        str(reference_path),
                        "--output",
                        str(output_csv),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=request.timeout_s,
                    check=False,
                )
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout or "no FEC output").strip()[-2000:]
                    raise RuntimeError(f"FEC inference exited {completed.returncode}: {detail}")
                with output_csv.open("r", encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) != 1:
                    raise ValueError(f"FEC inference returned {len(rows)} rows; expected one")
                result = rows[0]
            supported = str(result.get("calibration_supported", "")).lower() == "true"
            n_reference = result.get("calibration_n_reference")
            parsed_n = int(n_reference) if n_reference not in (None, "") else None
            if request.mode == FECEvidenceMode.ANOMALY:
                decision = str(result.get("decision") or "CONTEXT_UNSUPPORTED")
                status = (
                    FECEvidenceStatus.QUALIFIED
                    if supported and decision == "ABNORMAL"
                    else FECEvidenceStatus.NORMAL
                    if supported and decision == "NORMAL"
                    else FECEvidenceStatus.ABSTAIN
                )
                p_value = result.get("p_value")
                return FECEvidenceAudit(
                    **common,
                    status=status,
                    reference_sha256=reference_sha,
                    reference_contract=str(reference["contract"]),
                    reference_metadata=self._reference_metadata(reference),
                    declared_reference_scope=declared_scope,
                    fleet_commissioning_receipt_sha256=receipt_sha,
                    runtime=runtime,
                    calibration_supported=supported,
                    eligible_for_model_evidence=supported and decision == "ABNORMAL",
                    anomaly_decision=decision,
                    p_value=float(p_value) if p_value not in (None, "") else None,
                    calibration_n_reference=parsed_n,
                    calibration_reason=result.get("calibration_reason"),
                    raw_result=result,
                )
            prediction_set = json.loads(result.get("prediction_set") or "[]")
            qualified = supported and bool(prediction_set)
            return FECEvidenceAudit(
                **common,
                status=FECEvidenceStatus.QUALIFIED if qualified else FECEvidenceStatus.ABSTAIN,
                reference_sha256=reference_sha,
                reference_contract=str(reference["contract"]),
                reference_metadata=self._reference_metadata(reference),
                declared_reference_scope=declared_scope,
                fleet_commissioning_receipt_sha256=receipt_sha,
                runtime=runtime,
                calibration_supported=supported,
                eligible_for_model_evidence=qualified,
                prediction_set=list(map(str, prediction_set)),
                calibration_n_reference=parsed_n,
                calibration_reason=result.get("calibration_reason"),
                raw_result=result,
            )
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            IndexError,
            TypeError,
            json.JSONDecodeError,
            subprocess.SubprocessError,
        ) as exc:
            return FECEvidenceAudit(
                **common,
                status=FECEvidenceStatus.ERROR,
                eligible_for_model_evidence=False,
                errors=[f"{type(exc).__name__}: {exc}"],
            )


def fec_evidence_release_blockers(
    audit: FECEvidenceAudit | None,
    accepted_fault_id: str | None,
) -> list[str]:
    if audit is None or not audit.release_gate_enabled:
        return []
    if audit.status == FECEvidenceStatus.ERROR:
        return ["FEC_MODEL_EVIDENCE_ERROR"]
    if not audit.calibration_supported or audit.status == FECEvidenceStatus.ABSTAIN:
        return ["FEC_MODEL_EVIDENCE_UNSUPPORTED"]
    if accepted_fault_id is None:
        return []
    if audit.mode == FECEvidenceMode.ANOMALY:
        return [] if audit.anomaly_decision == "ABNORMAL" else ["FEC_ANOMALY_REFERENCE_REJECTS_FAULT"]
    if accepted_fault_id not in audit.prediction_set:
        return [
            f"FEC_PREDICTION_SET_FAULT_MISMATCH:governor={accepted_fault_id},"
            f"fec={','.join(audit.prediction_set)}"
        ]
    return []
