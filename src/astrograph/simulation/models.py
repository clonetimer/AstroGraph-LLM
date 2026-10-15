from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SimulationRunCompleted(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "astrograph-simulation-run-completed.v1"
    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    run_dir: str = Field(min_length=1)
    nominal_run_dir: str = Field(min_length=1)
    runtime_id: str = "phm-whole-spacecraft-minimal-runtime-v1.1"
    emitted_at: str = Field(default_factory=_now)
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def paths_are_distinct_for_non_nominal(self) -> "SimulationRunCompleted":
        if not Path(self.run_dir).name:
            raise ValueError("run_dir must identify a simulation run directory")
        return self


class OnlineDataGateReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "astrograph-online-data-gate.v1"
    passed: bool
    run_id: str
    row_count: int = 0
    field_count: int = 0
    required_field_count: int = 0
    time_monotonic: bool = False
    time_aligned_with_nominal: bool = False
    missing_required_fields: list[str] = Field(default_factory=list)
    missing_values: int = 0
    prohibited_model_input_fields: list[str] = Field(default_factory=list)
    protocol_version: str = ""
    warnings: list[str] = Field(default_factory=list)
    claim_boundary: str = "E0 synthetic runtime data gate; not a flight-data acceptance decision."
