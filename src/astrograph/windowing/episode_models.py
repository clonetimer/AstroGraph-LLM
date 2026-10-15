"""Generic episode models — independent of D0/legacy data formats.

These models define what a diagnosis episode looks like after semantic framing.
They intentionally separate time-series storage (external CSV/Parquet) from the
episode record itself (references only).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TimeAxisQuality(BaseModel):
    """Quality metadata for a time axis, independent of the values."""

    model_config = ConfigDict(extra="forbid")

    is_uniform_frame_counter: bool = Field(
        default=False,
        description="True if the time axis is an integer sequence 0,1,2,… "
        "(a simulation frame counter, not a real epoch timestamp).",
    )
    is_monotonic: bool = Field(default=False)
    is_strictly_monotonic: bool = Field(default=False)
    min_period_s: float | None = Field(default=None, ge=0.0)
    max_period_s: float | None = Field(default=None, ge=0.0)
    median_period_s: float | None = Field(default=None, ge=0.0)
    missing_value_count: int = Field(default=0, ge=0)
    duplicate_count: int = Field(default=0, ge=0)


class DataQualityStatus(BaseModel):
    """Per-component data quality assessment before any diagnosis."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        description="VALID | DEGRADED | INVALID",
    )
    reason_codes: list[str] = Field(default_factory=list)
    empty_trajectory: bool = False
    duplicate_trajectory: bool = False
    column_conflict: bool = False
    time_quality: TimeAxisQuality | None = None
    available_port_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    masked_feature_ids: list[str] = Field(default_factory=list)


class EpisodeComponent(BaseModel):
    """One component's contribution to an episode: time range + semantic channels."""

    model_config = ConfigDict(extra="forbid")

    component_instance_id: str
    component_archetype_id: str
    start_s: float
    end_s: float
    native_sample_period_s: float | None = Field(default=None, ge=0.0)
    semantic_channel_ids: list[str] = Field(default_factory=list)
    row_count: int = Field(default=0, ge=0)
    data_quality: DataQualityStatus | None = None


class EpisodeManifest(BaseModel):
    """A self-contained episode: one contiguous, replayable observation interval.

    An episode groups multiple components that were observed during the same
    time window. Each component keeps its native sampling rate and missing mask
    — no global resampling is performed at the episode level.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-episode-manifest.v1"
    episode_id: str
    source_id: str = "SAT_AGENT_PLATFORM"
    start_s: float
    end_s: float

    # Asset hierarchy
    asset_instance_id: str = "asset-under-test"
    whole_spacecraft_id: str | None = None
    subsystem_ids: list[str] = Field(default_factory=list)

    # Components within this episode
    components: list[EpisodeComponent] = Field(default_factory=list)

    # Context
    operating_condition: str | None = None
    condition_catalog_version: str | None = None

    # Data lineage (no labels in this record)
    source_file_references: dict[str, str] = Field(
        default_factory=dict,
        description="component_instance_id → absolute or relative file path",
    )
    channel_map_catalog_sha256: str | None = None
    semantic_schema_id: str | None = None

    # Quality
    overall_data_quality: DataQualityStatus | None = None

    # Guard: labels must never appear in this record
    labels_used: bool = Field(default=False, frozen=True)
    truth_used: bool = Field(default=False, frozen=True)


class EpisodeManifestCatalog(BaseModel):
    """A collection of episode manifests, typically from one data source run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-episode-manifest-catalog.v1"
    catalog_id: str
    source_id: str = "SAT_AGENT_PLATFORM"
    episodes: list[EpisodeManifest] = Field(default_factory=list)
    generated_at: str | None = None
    claim_boundary: str = (
        "Episodes are unlabeled observation intervals. "
        "Labels, when available, are injected through a separate adapter "
        "and must never enter the model input path."
    )
