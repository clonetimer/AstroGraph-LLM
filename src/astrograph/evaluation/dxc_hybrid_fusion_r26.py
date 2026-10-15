"""R26 deterministic fusion of continuous-residual and discrete-state DXC paths."""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .dxc_discrete_consistency import DXCDiscreteConsistencyResult
from .dxc_hybrid_model_based import DXCHybridPrediction


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DXCFusionPolicy(str, Enum):
    CONTINUOUS_ONLY = "CONTINUOUS_ONLY"
    UNION = "UNION"
    DISCRETE_OVERRIDE = "DISCRETE_OVERRIDE"
    DISCRETE_FIRST_SINGLE_CONTINUOUS = "DISCRETE_FIRST_SINGLE_CONTINUOUS"


class DXCR26FusedPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-dxc-r26-fused-prediction.v1"
    policy: DXCFusionPolicy
    predicted_fault_ids: tuple[str, ...]
    first_alarm_time_s: float | None = None
    continuous_fault_ids: tuple[str, ...]
    discrete_fault_ids: tuple[str, ...]
    continuous_model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    discrete_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_semantics: str = "DETERMINISTIC_EVIDENCE_AUTHORITY_FUSION_NOT_PROBABILITY"
    formal_evidence_eligible: bool = False
    claim_boundary: str = (
        "Discrete exact-state consistency has higher engineering authority than the residual heuristic. "
        "The continuous path remains a bounded fallback; this fused result is development diagnosis, not formal EvidenceFact or probability."
    )

    @model_validator(mode="after")
    def integrity(self) -> "DXCR26FusedPrediction":
        payload = {
            "policy": self.policy.value,
            "predicted_fault_ids": self.predicted_fault_ids,
            "first_alarm_time_s": self.first_alarm_time_s,
            "continuous_fault_ids": self.continuous_fault_ids,
            "discrete_fault_ids": self.discrete_fault_ids,
            "continuous_model_sha256": self.continuous_model_sha256,
            "graph_sha256": self.graph_sha256,
            "discrete_result_sha256": self.discrete_result_sha256,
        }
        if _digest(payload) != self.result_sha256:
            raise ValueError("DXC R26 fused result_sha256 mismatch")
        return self


def _continuous_top_one(continuous: DXCHybridPrediction) -> tuple[str, ...]:
    if not continuous.final_component_ids:
        return ()
    target = continuous.final_component_ids[0]
    matches = sorted(
        fault for fault in continuous.predicted_fault_ids
        if fault.split(":", 1)[0] == target
    )
    return tuple(matches[:1])


def fuse_dxc_r26(
    continuous: DXCHybridPrediction,
    discrete: DXCDiscreteConsistencyResult,
    policy: DXCFusionPolicy,
) -> DXCR26FusedPrediction:
    continuous_faults = tuple(sorted(set(continuous.predicted_fault_ids)))
    discrete_faults = tuple(sorted(set(discrete.predicted_fault_ids)))
    if policy == DXCFusionPolicy.CONTINUOUS_ONLY:
        predicted = continuous_faults
        first = continuous.first_alarm_time_s
    elif policy == DXCFusionPolicy.UNION:
        predicted = tuple(sorted(set(continuous_faults) | set(discrete_faults)))
        times = [x for x in (continuous.first_alarm_time_s, discrete.first_event_time_s) if x is not None]
        first = min(times) if times else None
    elif policy == DXCFusionPolicy.DISCRETE_OVERRIDE:
        if discrete_faults:
            predicted = discrete_faults
            first = discrete.first_event_time_s
        else:
            predicted = continuous_faults
            first = continuous.first_alarm_time_s
    elif policy == DXCFusionPolicy.DISCRETE_FIRST_SINGLE_CONTINUOUS:
        if discrete_faults:
            predicted = discrete_faults
            first = discrete.first_event_time_s
        else:
            predicted = _continuous_top_one(continuous)
            first = continuous.first_alarm_time_s if predicted else None
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError("DXC_R26_FUSION_POLICY_UNSUPPORTED")
    payload = {
        "policy": policy.value,
        "predicted_fault_ids": predicted,
        "first_alarm_time_s": first,
        "continuous_fault_ids": continuous_faults,
        "discrete_fault_ids": discrete_faults,
        "continuous_model_sha256": continuous.model_sha256,
        "graph_sha256": continuous.graph_sha256,
        "discrete_result_sha256": discrete.result_sha256,
    }
    return DXCR26FusedPrediction(
        policy=policy,
        predicted_fault_ids=predicted,
        first_alarm_time_s=first,
        continuous_fault_ids=continuous_faults,
        discrete_fault_ids=discrete_faults,
        continuous_model_sha256=continuous.model_sha256,
        graph_sha256=continuous.graph_sha256,
        discrete_result_sha256=discrete.result_sha256,
        result_sha256=_digest(payload),
    )


__all__ = ["DXCFusionPolicy", "DXCR26FusedPrediction", "fuse_dxc_r26"]
