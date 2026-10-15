"""Three-task plugin results without AAS binary-label normalization."""
from __future__ import annotations
import math
from typing import Literal, Self
from pydantic import Field, model_validator
from astrograph.semantics.online_contracts import FrozenContract, ReleaseRef


class TaskDistribution(FrozenContract):
    status: Literal["AVAILABLE", "ABSTAINED", "UNAVAILABLE"]
    class_ids: tuple[str, ...] = ()
    values: tuple[float, ...] = ()
    score_kind: Literal["PROBABILITY", "CALIBRATED_PROBABILITY", "RANKING_SCORE"]
    reason: str | None = None
    calibration_release: ReleaseRef | None = None

    @model_validator(mode="after")
    def check_distribution(self) -> Self:
        if self.score_kind == "CALIBRATED_PROBABILITY" and self.calibration_release is None:
            raise ValueError("calibrated probabilities require a pinned calibration release")
        if len(self.class_ids) != len(self.values) or len(set(self.class_ids)) != len(self.class_ids):
            raise ValueError("class order and values must align and IDs must be unique")
        if any(not value for value in self.class_ids) or any(not math.isfinite(x) for x in self.values):
            raise ValueError("empty class ID or non-finite score")
        if self.status == "AVAILABLE" and not self.class_ids:
            raise ValueError("available task requires scores")
        if self.status != "AVAILABLE" and not self.reason:
            raise ValueError("abstention/unavailability requires a reason")
        if self.status == "UNAVAILABLE" and self.values:
            raise ValueError("unavailable task must not fabricate scores")
        if self.values and self.score_kind != "RANKING_SCORE":
            if any(x < 0 or x > 1 for x in self.values) or not math.isclose(sum(self.values), 1, abs_tol=1e-6):
                raise ValueError("probabilities must sum to one within tolerance")
        return self


class MultiTaskPluginResult(FrozenContract):
    schema_version: Literal["astrograph-multitask-plugin-result.v1"] = "astrograph-multitask-plugin-result.v1"
    model_release: ReleaseRef
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    anomaly: TaskDistribution
    component: TaskDistribution
    fault: TaskDistribution
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def check_binary_anomaly(self) -> Self:
        if self.anomaly.class_ids and set(self.anomaly.class_ids) != {"NORMAL", "ANOMALY"}:
            raise ValueError("anomaly task must identify NORMAL/ANOMALY explicitly")
        return self
