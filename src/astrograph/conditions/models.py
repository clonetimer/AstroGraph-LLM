from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OperatingCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orbit_phase: str = "UNKNOWN"
    payload_mode: str = "UNKNOWN"
    communication_mode: str = "UNKNOWN"
    attitude_mode: str = "UNKNOWN"
    inference_method: str = "deterministic_heuristic_v1"
    confidence: str = "E0_SYNTHETIC_HEURISTIC"
    condition_ids: list[str] = Field(default_factory=list)
    catalog_version: str = ""
    unknown_dimensions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
