from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.conditions.models import OperatingCondition


class StreamSliceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stream: str
    source_file: str
    start_s: float
    end_s: float
    sample_period_s: float = Field(ge=0.0)
    row_count: int = Field(ge=0)
    channel_ids: list[str] = Field(default_factory=list)


class WindowLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    health_state: str
    fault_active: bool
    degradation_active: bool
    fault_ids: list[str] = Field(default_factory=list)


class TelemetryWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-telemetry-window.v1"
    window_id: str
    dataset_id: str
    run_id: str
    paired_nominal_run_id: str | None = None
    split: str
    case_type: str
    case_id: str
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    operating_condition: OperatingCondition
    streams: list[StreamSliceReference]
    labels: WindowLabels
    metadata: dict[str, Any] = Field(default_factory=dict)
