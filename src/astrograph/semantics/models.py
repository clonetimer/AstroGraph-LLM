from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TelemetryFieldSemantics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    chinese_name: str = ""
    component_instance: str = Field(min_length=1)
    observable_property: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    reference_frame: str = Field(min_length=1)
    axis: str = Field(min_length=1)
    sign_convention: str = Field(min_length=1)
    sample_period_s: float = Field(gt=0.0)
    valid_range: str = ""
    status_code_definition: str = ""
    source: str = Field(min_length=1)
    source_class: str = ""
    dataset_role: str = ""
    measurement_status: str = ""
    missing_value_rule: str = ""
    definition_status: str = ""
    source_evidence: str = ""
    limitations: str = ""


class TelemetrySemanticsValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-telemetry-semantics-validation.v1"
    field_count: int = Field(ge=0)
    expected_field_count: int = Field(ge=0)
    field_sets_equal: bool
    sample_periods_equal: bool
    duplicate_fields: list[str] = Field(default_factory=list)
    missing_from_dictionary: list[str] = Field(default_factory=list)
    missing_from_physical_semantics: list[str] = Field(default_factory=list)
    extra_in_dictionary: list[str] = Field(default_factory=list)
    extra_in_physical_semantics: list[str] = Field(default_factory=list)
    mismatched_sample_periods: list[dict[str, Any]] = Field(default_factory=list)
    passed: bool
    protocol_hashes: dict[str, str] = Field(default_factory=dict)
    claim_boundary: str = ""
