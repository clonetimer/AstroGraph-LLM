from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import DataDrivenModelManifest, DataDrivenTaskType


class FaultSubsystem(str, Enum):
    ADCS = "ADCS"
    EPS = "EPS"
    THERMAL = "THERMAL"
    COMM = "COMM"
    DATA = "DATA"
    PAYLOAD = "PAYLOAD"
    PROPULSION = "PROPULSION"
    OTHER = "OTHER"


class TelemetryChannelRole(str, Enum):
    PRIMARY = "PRIMARY"
    SUPPORTING = "SUPPORTING"
    CONTEXT = "CONTEXT"
    SAFETY = "SAFETY"


class TelemetryChannelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channel: str = Field(min_length=1)
    role: TelemetryChannelRole = TelemetryChannelRole.PRIMARY
    expected_effect: str = ""
    unit: str | None = None
    required: bool = True


class FaultValidationRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_independent_runs: int = Field(default=3, ge=1)
    required_splits: list[str] = Field(default_factory=lambda: ["train", "validation", "test"])
    required_activity_axes: list[str] = Field(default_factory=list)
    expert_review_required: bool = True
    real_flight_validation_required_for_product_claim: bool = True

    @field_validator("required_splits", "required_activity_axes")
    @classmethod
    def unique_values(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("validation requirement lists must contain unique values")
        return value


class FaultModeDefinition(BaseModel):
    """Governed fault-mode definition used by model selection and benchmark review.

    This catalog is deliberately narrower than a full ontology. It records the stable
    identifiers and telemetry/model contracts needed by the data-driven branch while
    Neo4j remains the authority for causal and evidence relationships.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-fault-mode-definition.v1"
    fault_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    subsystem: FaultSubsystem
    description: str = Field(min_length=1)
    telemetry_channels: list[TelemetryChannelSpec] = Field(min_length=1)
    candidate_model_ids: list[str] = Field(default_factory=list)
    allowed_task_types: list[DataDrivenTaskType] = Field(
        default_factory=lambda: [
            DataDrivenTaskType.MULTICLASS_FAULT_CLASSIFICATION,
            DataDrivenTaskType.MULTILABEL_FAULT_RANKING,
            DataDrivenTaskType.ANOMALY_ATTRIBUTION,
        ]
    )
    validation: FaultValidationRequirement = Field(default_factory=FaultValidationRequirement)
    neo4j_fault_node_id: str | None = None
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("telemetry_channels")
    @classmethod
    def unique_channels(cls, value: list[TelemetryChannelSpec]) -> list[TelemetryChannelSpec]:
        channels = [item.channel for item in value]
        if len(channels) != len(set(channels)):
            raise ValueError("fault telemetry channels must be unique")
        return value

    @field_validator("candidate_model_ids", "allowed_task_types")
    @classmethod
    def unique_lists(cls, value: list[Any]) -> list[Any]:
        if len(value) != len(set(value)):
            raise ValueError("fault-mode lists must contain unique values")
        return value

    @property
    def required_channels(self) -> list[str]:
        return [item.channel for item in self.telemetry_channels if item.required]


class FaultCatalogValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    fault_id: str | None = None
    model_manifest_id: str | None = None
    message: str


class FaultCatalogValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-fault-catalog-validation-report.v1"
    passed: bool
    catalog_id: str
    issues: list[FaultCatalogValidationIssue] = Field(default_factory=list)


class FaultModeCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-fault-mode-catalog.v1"
    catalog_id: str = Field(min_length=1)
    catalog_version: str = Field(min_length=1)
    fault_modes: list[FaultModeDefinition] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("fault_modes")
    @classmethod
    def unique_fault_ids(cls, value: list[FaultModeDefinition]) -> list[FaultModeDefinition]:
        ids = [item.fault_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("fault catalog fault IDs must be unique")
        return value

    @model_validator(mode="after")
    def validate_neo4j_ids(self) -> "FaultModeCatalog":
        ids = [item.neo4j_fault_node_id for item in self.fault_modes if item.neo4j_fault_node_id]
        if len(ids) != len(set(ids)):
            raise ValueError("neo4j_fault_node_id values must be unique when provided")
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> "FaultModeCatalog":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def get(self, fault_id: str) -> FaultModeDefinition:
        for item in self.fault_modes:
            if item.fault_id == fault_id:
                return item
        raise KeyError(f"unknown fault mode: {fault_id}")

    def fault_ids(self, *, enabled_only: bool = True) -> list[str]:
        return sorted(
            item.fault_id for item in self.fault_modes if item.enabled or not enabled_only
        )

    def validate_manifest(self, manifest: DataDrivenModelManifest) -> FaultCatalogValidationReport:
        issues: list[FaultCatalogValidationIssue] = []
        known = set(self.fault_ids(enabled_only=False))
        for fault_id in manifest.supported_fault_ids:
            if fault_id not in known:
                issues.append(
                    FaultCatalogValidationIssue(
                        code="MODEL_SUPPORTS_UNKNOWN_FAULT",
                        fault_id=fault_id,
                        model_manifest_id=manifest.manifest_id,
                        message=f"model manifest references fault outside catalog: {fault_id}",
                    )
                )
                continue
            definition = self.get(fault_id)
            if not definition.enabled:
                issues.append(
                    FaultCatalogValidationIssue(
                        code="MODEL_SUPPORTS_DISABLED_FAULT",
                        fault_id=fault_id,
                        model_manifest_id=manifest.manifest_id,
                        message=f"model manifest references disabled fault: {fault_id}",
                    )
                )
            if manifest.task_type not in definition.allowed_task_types:
                issues.append(
                    FaultCatalogValidationIssue(
                        code="MODEL_TASK_TYPE_NOT_ALLOWED",
                        fault_id=fault_id,
                        model_manifest_id=manifest.manifest_id,
                        message=(
                            f"task type {manifest.task_type.value} is not allowed for fault {fault_id}"
                        ),
                    )
                )
        return FaultCatalogValidationReport(
            passed=not issues,
            catalog_id=self.catalog_id,
            issues=issues,
        )
