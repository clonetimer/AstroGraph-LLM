from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FaultPatternType(str, Enum):
    SPIKE = "spike"
    SHIFT = "shift"
    DRIFT = "drift"
    PERIODIC = "periodic"
    DROPOUT = "dropout"
    NOISE = "noise"
    MULTI_RATE = "multi_rate"
    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    REVERSE_TIMESTAMP = "reverse_timestamp"
    MISSING_CHANNEL = "missing_channel"


class FaultPatternSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: FaultPatternType
    channel: str | None = None
    start_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    end_fraction: float = Field(default=0.8, ge=0.0, le=1.0)
    magnitude: float = 5.0
    period: int = Field(default=10, ge=2)
    keep_every: int = Field(default=2, ge=2)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_interval_and_channel(self) -> "FaultPatternSpec":
        if self.end_fraction < self.start_fraction:
            raise ValueError("end_fraction must be >= start_fraction")
        channel_required = self.pattern not in {
            FaultPatternType.DUPLICATE_TIMESTAMP,
            FaultPatternType.REVERSE_TIMESTAMP,
        }
        if channel_required and not self.channel:
            raise ValueError(f"{self.pattern.value} requires a channel")
        return self
