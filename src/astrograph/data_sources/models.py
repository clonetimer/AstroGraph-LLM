from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DataSourceType(str, Enum):
    SIMULATION = "simulation"
    SAT_AGENT_PLATFORM = "sat_agent_platform"
    ESA_ADB = "esa_adb"
    LOCAL_FILE = "local_file"
    MEMORY = "memory"


class DatasetSourceConfig(BaseModel):
    """Configuration shared by file-backed dataset sources.

    Paths are supplied at runtime and are intentionally not persisted into probe
    artifacts. ``read_only`` is fail-closed and cannot be disabled for the built-in
    ESA-ADB and simulation adapters.
    """

    model_config = ConfigDict(extra="forbid")

    source_type: DataSourceType
    root: str = ""
    read_only: bool = True
    lazy: bool = True
    recursive: bool = True
    timestamp_column: str | None = None
    label_columns: list[str] = Field(default_factory=list)
    include_globs: list[str] = Field(default_factory=list)
    exclude_globs: list[str] = Field(default_factory=list)
    profile_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("label_columns")
    @classmethod
    def unique_label_columns(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("label_columns must be unique")
        return value


class DataEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    format: str = Field(min_length=1)
    timestamp_column: str | None = None
    row_count: int | None = Field(default=None, ge=0)
    channel_count: int | None = Field(default=None, ge=0)
    label_columns: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataSourceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-data-source-metadata.v1"
    source_type: DataSourceType
    opened: bool
    read_only: bool
    root_token: str = "${ASTROGRAPH_DATA_ROOT}"
    entity_count: int = Field(default=0, ge=0)
    supported_formats: list[str] = Field(default_factory=list)
    adapter_version: str = "0.5.30-r4f3.2"
    metadata: dict[str, Any] = Field(default_factory=dict)
    claim_boundary: str = (
        "Structural data-source compatibility only; no claim of physical, label, "
        "or diagnostic correctness without local dataset acceptance."
    )


class DataQualityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    severity: str = Field(pattern=r"^(INFO|WARNING|ERROR)$")
    entity_id: str = ""
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class DataSourceValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-data-source-validation.v1"
    generated_at: str = Field(default_factory=utc_now)
    source_type: DataSourceType
    status: str = Field(pattern=r"^(PASS|WARNING|FAIL)$")
    entity_count: int = Field(default=0, ge=0)
    inspected_entity_count: int = Field(default=0, ge=0)
    channel_count_min: int = Field(default=0, ge=0)
    channel_count_max: int = Field(default=0, ge=0)
    row_count_total: int = Field(default=0, ge=0)
    issues: list[DataQualityIssue] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    claim_boundary: str = (
        "PASS means the inspected files are readable and satisfy configured structural "
        "checks. It is not an ESA-ADB benchmark score or flight-data qualification."
    )


class ProbeOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_entities: int = Field(default=100, ge=1)
    max_rows_per_entity: int = Field(default=10000, ge=1)
    hash_mode: str = Field(default="metadata", pattern=r"^(none|metadata|full)$")
    include_statistics: bool = False
    include_label_values: bool = False
    root_token: str = "${ASTROGRAPH_ESA_ADB_ROOT}"


class ProbeArtifactSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-dataset-probe-artifacts.v1"
    generated_at: str = Field(default_factory=utc_now)
    output_dir: str
    files: dict[str, str]
