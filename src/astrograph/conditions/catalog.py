from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConditionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition_id: str = Field(min_length=1)
    dimension: Literal["orbit_phase", "payload_mode", "communication_mode", "attitude_mode"]
    display_name_zh: str = Field(min_length=1)
    description: str = ""


class ConditionThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_fraction: float = Field(default=0.8, ge=0.5, le=1.0)
    inactive_fraction: float = Field(default=0.2, ge=0.0, le=0.5)
    solar_current_active_a: float = Field(default=0.05, ge=0.0)
    payload_active_threshold: float = Field(default=0.5, ge=0.0)
    downlink_rate_active_bps: float = Field(default=1.0, ge=0.0)
    attitude_torque_maneuver_nm: float = Field(default=0.05, ge=0.0)
    gyro_rate_maneuver_rad_s: float = Field(default=0.01, ge=0.0)

    @model_validator(mode="after")
    def check_fraction_order(self) -> "ConditionThresholds":
        if self.inactive_fraction >= self.active_fraction:
            raise ValueError("inactive_fraction must be below active_fraction")
        return self


class ConditionCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-operating-condition-catalog.v1"
    catalog_version: str = Field(min_length=1)
    evidence_level: str = "E0"
    review_status: str = "provisional"
    thresholds: ConditionThresholds = Field(default_factory=ConditionThresholds)
    conditions: list[ConditionDefinition]

    @model_validator(mode="after")
    def validate_catalog(self) -> "ConditionCatalog":
        ids = [item.condition_id for item in self.conditions]
        if len(ids) != len(set(ids)):
            raise ValueError("condition_id must be unique")
        required_dimensions = {"orbit_phase", "payload_mode", "communication_mode", "attitude_mode"}
        present = {item.dimension for item in self.conditions}
        if present != required_dimensions:
            raise ValueError(f"catalog dimensions must equal {sorted(required_dimensions)}")
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> "ConditionCatalog":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def known_ids(self) -> set[str]:
        return {item.condition_id for item in self.conditions}
