from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DiagnosisStatus(str, Enum):
    DIAGNOSED = "DIAGNOSED"
    UNKNOWN = "UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class DataGateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class AnomalyEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anomaly_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    signal: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_type: str = "synthetic_residual_strength_not_calibrated"
    observed_at_s: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fault_id: str = Field(min_length=1)
    fault_name: str = Field(min_length=1)
    anomaly_id: str = Field(min_length=1)
    anomaly_name: str = Field(min_length=1)
    weight: float = Field(ge=0.0, le=1.0)
    edge_role: str = "direct"
    evidence_level: str = "E0"
    evidence_source: str = "synthetic_quick_campaign_seed"
    source_ref: str = ""
    review_status: str = "provisional"
    provisional: bool = True
    knowledge_version: str = "astrograph-ontology-v1-e0"
    applicable_condition_ids: list[str] = Field(default_factory=list)
    excluded_condition_ids: list[str] = Field(default_factory=list)

    @field_validator("evidence_level")
    @classmethod
    def validate_evidence_level(cls, value: str) -> str:
        if value not in {"E0", "E1", "E2", "E3", "E4"}:
            raise ValueError("evidence_level must be E0..E4")
        return value

    @field_validator("edge_role")
    @classmethod
    def validate_edge_role(cls, value: str) -> str:
        if value not in {"direct", "indirect", "cascade"}:
            raise ValueError("edge_role must be direct/indirect/cascade")
        return value


class CandidateScore(BaseModel):
    fault_id: str
    fault_name: str
    score: float
    raw_score: float
    evidence_count: int
    direct_evidence_count: int = 0
    evidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    contributions: list[dict[str, Any]] = Field(default_factory=list)
    support_score: float = 0.0
    counter_evidence_penalty: float = 0.0
    counter_evidence: list[dict[str, Any]] = Field(default_factory=list)
    critical_evidence_satisfied: bool = True
    missing_critical_evidence: list[dict[str, Any]] = Field(default_factory=list)
    eligible_for_diagnosis: bool = True
    temporal_constraints_satisfied: bool = True
    temporal_constraint_penalty: float = 0.0
    temporal_constraint_evaluations: list[dict[str, Any]] = Field(default_factory=list)


class DiagnosisResult(BaseModel):
    status: DiagnosisStatus
    conclusion_fault_id: str | None = None
    conclusion_fault_name: str | None = None
    top_score: float = 0.0
    score_margin: float = 0.0
    candidates: list[CandidateScore] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    policy_version: str = ""
    knowledge_version: str = ""
    claim_boundary: str = (
        "Synthetic D0 PHM engineering evidence only; not flight-qualified or calibrated diagnosis."
    )


class WorkflowInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    anomalies: list[AnomalyEvidence]
    operating_condition: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("anomalies")
    @classmethod
    def unique_anomalies(cls, value: list[AnomalyEvidence]) -> list[AnomalyEvidence]:
        ids = [item.anomaly_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("anomaly_id must be unique within one request")
        return value


class DataGateCheck(BaseModel):
    check_id: str
    status: DataGateStatus
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class DataQualityReport(BaseModel):
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: DataGateStatus
    profile: str = ""
    schema_version: str = ""
    checks: list[DataGateCheck] = Field(default_factory=list)
    claim_boundary: str = ""


class DiagnosticAuditRecord(BaseModel):
    audit_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_id: str
    case_id: str
    input_hash: str
    policy_hash: str
    knowledge_hash: str
    report_hash: str
    knowledge_version: str
    policy_version: str
    anomalies: list[dict[str, Any]]
    retrieved_edges: list[dict[str, Any]]
    diagnosis: dict[str, Any]
    report_path: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowState(TypedDict, total=False):
    case_id: str
    request_id: str
    anomalies: list[dict[str, Any]]
    operating_condition: dict[str, Any]
    metadata: dict[str, Any]
    retrieved_edges: list[dict[str, Any]]
    diagnosis: dict[str, Any]
    report_markdown: str
    audit_record: dict[str, Any]
    error: str
