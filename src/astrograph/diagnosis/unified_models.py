from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.evidence.quality import EvidenceQualityAssessment


class UnifiedInputMode(str, Enum):
    PAIRED = "PAIRED"
    SINGLE_TRACE = "SINGLE_TRACE"


class UnifiedBackend(str, Enum):
    AUTO = "auto"
    COMPOSITE = "composite"
    UNIFIED = "unified"


class UnifiedDiagnosisStatus(str, Enum):
    DIAGNOSED = "DIAGNOSED"
    NORMAL = "NORMAL"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN_FAULT = "UNKNOWN_FAULT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    DATA_ERROR = "DATA_ERROR"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


class EngineExecutionStatus(str, Enum):
    EXECUTED = "EXECUTED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


class DataQualityFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: str
    message: str
    fields: list[str] = Field(default_factory=list)


class TelemetryProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: str
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    numeric_signal_count: int = Field(ge=0)
    time_column: str | None = None
    time_monotonic: bool | None = None
    prohibited_columns: list[str] = Field(default_factory=list)
    normalized_columns: list[str] = Field(default_factory=list)
    alias_mapping: dict[str, str] = Field(default_factory=dict)
    findings: list[DataQualityFinding] = Field(default_factory=list)
    status: str = "PASS"


class EngineCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_id: str
    input_modes: list[UnifiedInputMode]
    backends: list[str]
    requires_nominal: bool
    requires_model: bool = False
    known_effect_ids: list[str] = Field(default_factory=list)
    claim_boundary: str


class EngineRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine_id: str
    eligible: bool
    execution_status: EngineExecutionStatus
    status: str
    reason_codes: list[str] = Field(default_factory=list)
    detected_effect_ids: list[str] = Field(default_factory=list)
    primary_effect_id: str | None = None
    score: float | None = None
    raw_result: dict[str, Any] | None = None


class UnifiedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effect_id: str
    support_engines: list[str] = Field(default_factory=list)
    contradicting_engines: list[str] = Field(default_factory=list)
    screening_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_summary: list[str] = Field(default_factory=list)




class DiagnosticEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source_id: str | None = None
    quality: EvidenceQualityAssessment | None = None
    field: str
    statistic: str
    nominal_value: float | None = None
    effect_value: float | None = None
    observed_delta: float | None = None
    threshold: float | None = None
    interpretation: str


class UnifiedDiagnosisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    telemetry_path: Path
    nominal_path: Path | None = None
    backend: UnifiedBackend = UnifiedBackend.AUTO
    catalog_path: Path | None = None
    catalog_model_path: Path | None = None
    open_set_model_path: Path | None = None

    @model_validator(mode="after")
    def validate_paths(self) -> "UnifiedDiagnosisRequest":
        if self.nominal_path is not None and self.nominal_path == self.telemetry_path:
            raise ValueError("nominal_path and telemetry_path must be different files")
        return self


class UnifiedDiagnosisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-unified-diagnosis.v1"
    case_id: str
    input_mode: UnifiedInputMode
    backend: str
    status: UnifiedDiagnosisStatus
    conclusion_effect_ids: list[str] = Field(default_factory=list)
    primary_effect_id: str | None = None
    primary_subsystem: str | None = None
    primary_component: str | None = None
    estimated_onset_time_s: float | None = None
    onset_status: str | None = None
    estimated_severity: str | None = None
    severity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    diagnostic_evidence: list[DiagnosticEvidence] = Field(default_factory=list)
    resolution_basis: str
    data_quality: list[TelemetryProfile]
    engine_runs: list[EngineRunRecord]
    candidates: list[UnifiedCandidate] = Field(default_factory=list)
    routing_decisions: list[str] = Field(default_factory=list)
    labels_used: bool = False
    truth_used: bool = False
    deterministic: bool = True
    claim_boundary: str = (
        "Unified laboratory diagnosis orchestration. Screening scores are deterministic evidence "
        "coverage, not calibrated probabilities, causal proof, flight qualification, or safety approval."
    )


class UnifiedBatchCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    telemetry: Path
    nominal: Path | None = None
    backend: UnifiedBackend = UnifiedBackend.AUTO


class UnifiedBatchManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-unified-batch-manifest.v1"
    cases: list[UnifiedBatchCase]


class UnifiedBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-unified-batch-result.v1"
    results: list[UnifiedDiagnosisResult]
    status_counts: dict[str, int]


__all__ = [
    "UnifiedInputMode",
    "UnifiedBackend",
    "UnifiedDiagnosisStatus",
    "EngineExecutionStatus",
    "DataQualityFinding",
    "TelemetryProfile",
    "EngineCapability",
    "EngineRunRecord",
    "UnifiedCandidate",
    "DiagnosticEvidence",
    "UnifiedDiagnosisRequest",
    "UnifiedDiagnosisResult",
    "UnifiedBatchCase",
    "UnifiedBatchManifest",
    "UnifiedBatchResult",
]
