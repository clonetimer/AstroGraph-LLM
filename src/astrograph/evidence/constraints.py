from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CriticalEvidenceRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: str = Field(min_length=1)
    fault_id: str = Field(min_length=1)
    any_of_anomaly_ids: list[str] = Field(min_length=1)
    minimum_matches: int = Field(default=1, ge=1)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    action: Literal["block_diagnosis", "penalize"] = "block_diagnosis"
    penalty: float = Field(default=0.0, ge=0.0)
    rationale: str = Field(min_length=1)
    evidence_level: str = "E0"

    @field_validator("any_of_anomaly_ids")
    @classmethod
    def unique_anomalies(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("any_of_anomaly_ids must be unique")
        return value


class CounterEvidenceRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(min_length=1)
    fault_id: str = Field(min_length=1)
    anomaly_id: str = Field(min_length=1)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    penalty_weight: float = Field(default=0.0, ge=0.0)
    rationale: str = Field(min_length=1)
    evidence_level: str = "E0"


class EvidenceConstraintCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-evidence-constraint-catalog.v1"
    catalog_version: str = Field(min_length=1)
    critical_requirements: list[CriticalEvidenceRequirement] = Field(default_factory=list)
    counter_evidence_rules: list[CounterEvidenceRule] = Field(default_factory=list)
    claim_boundary: str = (
        "Engineering diagnostic constraints. Missing critical evidence blocks deterministic diagnosis; "
        "counter-evidence penalties are not calibrated probabilities."
    )

    @classmethod
    def from_json(cls, path: str | Path) -> "EvidenceConstraintCatalog":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def requirements_for(self, fault_id: str) -> list[CriticalEvidenceRequirement]:
        return [item for item in self.critical_requirements if item.fault_id == fault_id]

    def counter_rules_for(self, fault_id: str) -> list[CounterEvidenceRule]:
        return [item for item in self.counter_evidence_rules if item.fault_id == fault_id]
