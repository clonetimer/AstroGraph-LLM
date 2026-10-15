from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PluginReviewStatus(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    APPROVED = "APPROVED"
    RETIRED = "RETIRED"


class ObjectPluginManifest(BaseModel):
    """Stable contract describing one object-specific deterministic plugin."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-object-plugin-manifest.v1"
    plugin_api_version: str = "1.0"
    plugin_id: str = Field(min_length=3, pattern=r"^[a-z0-9][a-z0-9_.-]+$")
    plugin_version: str = Field(min_length=1)
    object_type: str = Field(min_length=1)
    subsystem: str = Field(min_length=1)
    supported_dataset_profiles: list[str] = Field(default_factory=list)
    required_operator_ids: list[str] = Field(default_factory=list)
    evidence_catalog_schema_version: str = "astrograph-evidence-definition.v1"
    deterministic_runtime: bool = True
    review_status: PluginReviewStatus = PluginReviewStatus.DEVELOPMENT
    source_refs: list[str] = Field(default_factory=list)
    claim_boundary: str = "Engineering diagnostic plugin; not flight-qualified unless separately approved."
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_manifest(self) -> "ObjectPluginManifest":
        if len(self.required_operator_ids) != len(set(self.required_operator_ids)):
            raise ValueError("required_operator_ids must be unique")
        if self.review_status == PluginReviewStatus.APPROVED and not self.source_refs:
            raise ValueError("approved object plugin requires source_refs")
        if not self.deterministic_runtime:
            raise ValueError("object diagnostic plugins must provide a deterministic runtime")
        return self


class ObjectEvidenceBuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delivery_path: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    protocol_root: str = Field(min_length=1)
    evidence_catalog_path: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ObjectEvidenceBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plugin_id: str = Field(min_length=1)
    plugin_version: str = Field(min_length=1)
    builder_id: str = Field(min_length=1)
    builder_version: str = Field(min_length=1)
    bundle: dict[str, Any]
    core_payload: dict[str, Any] | list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)
