from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvidenceLevel(str, Enum):
    E0 = "E0"  # simulation assumption / synthetic evidence
    E1 = "E1"  # engineering manual or vendor information
    E2 = "E2"  # peer-reviewed paper, standard, FMEA/FMECA
    E3 = "E3"  # ground/bench test evidence
    E4 = "E4"  # flight-correlated evidence


class EvidenceSourceStatus(str, Enum):
    REGISTERED = "REGISTERED"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    RETIRED = "RETIRED"


class EdgeReviewDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    DEFER = "DEFER"


class EvidenceSourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-evidence-source.v1"
    source_id: str = Field(min_length=8)
    title: str = Field(min_length=1)
    source_type: Literal[
        "simulation_campaign",
        "paper",
        "standard",
        "fmea_fmeca",
        "vendor_manual",
        "engineering_document",
        "ground_test",
        "flight_data",
    ]
    evidence_level: EvidenceLevel
    status: EvidenceSourceStatus = EvidenceSourceStatus.REGISTERED
    scope: list[Literal[
        "methodology",
        "fault_mechanism",
        "parameter_basis",
        "telemetry_definition",
        "observability",
        "validation",
    ]] = Field(default_factory=list)
    original_filename: str = ""
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_reference: str = ""
    applicable_fault_ids: list[str] = Field(default_factory=list)
    applicable_anomaly_ids: list[str] = Field(default_factory=list)
    eligible_for_edge_promotion: bool = False
    registered_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reviewed_at: str | None = None
    reviewer_role: str = ""
    claim_boundary: str = ""
    notes: str = ""

    @field_validator("scope")
    @classmethod
    def unique_scope(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("scope values must be unique")
        return value

    @model_validator(mode="after")
    def approved_requires_review(self) -> "EvidenceSourceRecord":
        if self.status == EvidenceSourceStatus.APPROVED:
            if not self.reviewed_at or not self.reviewer_role:
                raise ValueError("approved evidence source requires reviewed_at and reviewer_role")
        return self

    @classmethod
    def deterministic_id(cls, title: str, document_sha256: str) -> str:
        payload = f"{title}|{document_sha256}".encode("utf-8")
        return "ES-" + hashlib.sha256(payload).hexdigest()[:20].upper()


class EdgeReviewRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-edge-review.v1"
    review_id: str = Field(min_length=8)
    knowledge_version: str = Field(min_length=1)
    fault_id: str = Field(min_length=1)
    anomaly_id: str = Field(min_length=1)
    source_id: str = Field(min_length=8)
    decision: EdgeReviewDecision
    current_evidence_level: EvidenceLevel
    proposed_evidence_level: EvidenceLevel
    reviewer_role: str = Field(min_length=1)
    reviewed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    rationale: str = Field(min_length=1)
    conditions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_promotion(self) -> "EdgeReviewRecord":
        order = {level.value: index for index, level in enumerate(EvidenceLevel)}
        if self.decision == EdgeReviewDecision.APPROVE:
            if order[self.proposed_evidence_level.value] < order[self.current_evidence_level.value]:
                raise ValueError("approved review cannot lower evidence level")
        return self

    @classmethod
    def deterministic_id(
        cls,
        *,
        knowledge_version: str,
        fault_id: str,
        anomaly_id: str,
        source_id: str,
    ) -> str:
        payload = "|".join([knowledge_version, fault_id, anomaly_id, source_id]).encode("utf-8")
        return "ER-" + hashlib.sha256(payload).hexdigest()[:20].upper()
