from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "astrograph-maintenance-supplemental-input-provenance.v1"


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sequence_sha256(values: list[Any] | None) -> str | None:
    if values is None:
        return None
    return canonical_sha256(values)


class SupplementalInputProvenanceReceipt(BaseModel):
    """Content-bound provenance for one admitted maintenance input capsule.

    The receipt records *what bytes/values were admitted* and which frozen
    contract/report they were evaluated against.  It is not a release receipt
    and cannot authorize a production decision by itself.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    provenance_id: str = Field(pattern=r"^SUPP-PROV-[0-9A-F]{20}$")
    source_job_id: str = Field(min_length=1, max_length=80)
    source_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_evidence_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_id: str = Field(min_length=1, max_length=240)
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capsule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    admission_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_admission_receipt_sha256s: list[str] = Field(default_factory=list)
    resolved_source_field: str = Field(min_length=1, max_length=300)
    unit: str = Field(min_length=1, max_length=80)
    window_start_s: float = Field(ge=0.0)
    window_end_s: float = Field(ge=0.0)
    sample_count: int = Field(ge=1)
    sample_times_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_values_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_values_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    actor: str = Field(min_length=1, max_length=200)
    admitted_at: str = Field(min_length=1, max_length=100)
    formal_release_eligible: Literal[False] = False
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identity(self) -> "SupplementalInputProvenanceReceipt":
        if self.window_end_s < self.window_start_s:
            raise ValueError("window_end_s must be >= window_start_s")
        if len(set(self.parent_admission_receipt_sha256s)) != len(self.parent_admission_receipt_sha256s):
            raise ValueError("parent admission receipt hashes must be unique")
        for value in self.parent_admission_receipt_sha256s:
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError("invalid parent admission receipt sha256")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if canonical_sha256(payload) != self.receipt_sha256:
            raise ValueError("supplemental provenance receipt_sha256 mismatch")
        return self


def build_supplemental_input_provenance_receipt(
    *,
    source_job_id: str,
    source_report: dict[str, Any],
    original_evidence_bundle: dict[str, Any],
    contract: dict[str, Any],
    capsule: dict[str, Any],
    admission_receipt_sha256: str,
    parent_admission_receipt_sha256s: list[str],
    actor: str,
    admitted_at: str,
) -> SupplementalInputProvenanceReceipt:
    identity = {
        "source_job_id": source_job_id,
        "contract_id": str(contract.get("contract_id") or ""),
        "capsule_sha256": canonical_sha256(capsule),
        "admission_receipt_sha256": admission_receipt_sha256,
    }
    provenance_id = "SUPP-PROV-" + canonical_sha256(identity)[:20].upper()
    core = {
        "schema_version": SCHEMA_VERSION,
        "provenance_id": provenance_id,
        "source_job_id": source_job_id,
        "source_report_sha256": canonical_sha256(source_report),
        "original_evidence_bundle_sha256": canonical_sha256(original_evidence_bundle),
        "contract_id": str(contract.get("contract_id") or ""),
        "contract_sha256": canonical_sha256(contract),
        "capsule_sha256": canonical_sha256(capsule),
        "admission_receipt_sha256": admission_receipt_sha256,
        "parent_admission_receipt_sha256s": list(parent_admission_receipt_sha256s),
        "resolved_source_field": str(capsule.get("resolved_source_field") or ""),
        "unit": str(capsule.get("unit") or ""),
        "window_start_s": float(capsule.get("window_start_s")),
        "window_end_s": float(capsule.get("window_end_s")),
        "sample_count": len(list(capsule.get("sample_times_s") or [])),
        "sample_times_sha256": _sequence_sha256(list(capsule.get("sample_times_s") or [])),
        "current_values_sha256": _sequence_sha256(list(capsule.get("current_values") or [])),
        "baseline_values_sha256": _sequence_sha256(capsule.get("baseline_values")),
        "actor": actor,
        "admitted_at": admitted_at,
        "formal_release_eligible": False,
        "claim_boundary": (
            "This receipt freezes the admitted maintenance capsule, the frozen input contract, and the original report/evidence identities. "
            "It proves provenance only. It does not convert a maintenance overlay into production release evidence and does not authorize a formal diagnosis."
        ),
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    return SupplementalInputProvenanceReceipt.model_validate(receipt)


def verify_supplemental_input_provenance_receipt(payload: dict[str, Any]) -> SupplementalInputProvenanceReceipt:
    return SupplementalInputProvenanceReceipt.model_validate(payload)


__all__ = [
    "SCHEMA_VERSION",
    "SupplementalInputProvenanceReceipt",
    "build_supplemental_input_provenance_receipt",
    "canonical_sha256",
    "verify_supplemental_input_provenance_receipt",
]
