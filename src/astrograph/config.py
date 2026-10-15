from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DiagnosisPolicy(BaseModel):
    """Deterministic decision policy. Values are engineering defaults, not probabilities."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str = "diagnosis-policy-0.3.3"
    min_score: float = Field(default=0.45, ge=0.0)
    min_margin: float = Field(default=0.08, ge=0.0)
    min_evidence_count: int = Field(default=1, ge=1)
    role_factors: dict[str, float] = Field(
        default_factory=lambda: {"direct": 1.0, "indirect": 0.60, "cascade": 0.35}
    )
    evidence_level_factors: dict[str, float] = Field(
        default_factory=lambda: {"E0": 0.70, "E1": 0.80, "E2": 0.90, "E3": 1.0, "E4": 1.0}
    )
    provisional_factor: float = Field(default=0.85, ge=0.0, le=1.0)
    coverage_floor: float = Field(default=0.50, ge=0.0, le=1.0)
    claim_boundary: str = (
        "Synthetic D0 PHM engineering evidence only; not flight-qualified, not calibrated probability or RUL."
    )

    @classmethod
    def from_json(cls, path: str | Path) -> "DiagnosisPolicy":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
