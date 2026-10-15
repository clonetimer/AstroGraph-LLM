from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConclusionLevel(str, Enum):
    CONFIRMED = "confirmed"
    PROBABLE = "probable"
    POSSIBLE = "possible"
    UNKNOWN = "unknown"
    SYSTEM_ERROR = "system_error"


class DatasetDiagnosisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-dataset-diagnosis-request.v1"
    request_id: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    start: str | float | int | None = None
    end: str | float | int | None = None
    channels: list[str] = Field(default_factory=list)
    baseline_fraction: float = Field(default=0.2, gt=0.0, lt=0.8)
    z_threshold: float = Field(default=6.0, gt=0.0)
    minimum_baseline_points: int = Field(default=10, ge=3)
    signal_mapping: dict[str, dict[str, str]] = Field(default_factory=dict)
    operating_condition: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("channels")
    @classmethod
    def unique_channels(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("channels must be unique")
        return value


class ObservationFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str
    signal: str
    sample_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    baseline_median: float | None = None
    baseline_mad: float | None = None
    maximum_robust_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DetectionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detection_id: str
    anomaly_id: str
    anomaly_name: str
    signal: str
    detector_id: str
    score: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    observed_at: str | float | int | None = None
    missing_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    mapped: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiagnosticConclusion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: ConclusionLevel
    fault_id: str | None = None
    fault_name: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    claim_boundary: str


class LayeredDatasetDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-layered-dataset-diagnosis.v1"
    diagnosis_id: str
    run_id: str
    window_id: str
    request_id: str
    entity_id: str
    generated_at: str = Field(default_factory=utc_now)
    observations: list[ObservationFact] = Field(default_factory=list)
    detections: list[DetectionRecord] = Field(default_factory=list)
    fault_candidates: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    conclusion: DiagnosticConclusion
    provenance: dict[str, Any] = Field(default_factory=dict)
    report_markdown: str = ""
    replayed: bool = False


class DiagnosisRunState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-dataset-diagnosis-run-state.v1"
    run_id: str
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: str = Field(pattern=r"^(CREATED|RUNNING|FAILED|COMPLETED)$")
    completed_stages: list[str] = Field(default_factory=list)
    failed_stage: str = ""
    error: str = ""
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    data_hash: str = ""
    window_id: str = ""
    request: dict[str, Any]
    stage_outputs: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
