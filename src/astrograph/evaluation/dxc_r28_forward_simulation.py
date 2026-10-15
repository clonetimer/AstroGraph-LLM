"""R28 adapter: R27 core prediction plus qualitative state-transition forward model."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.diagnosis.dxc_forward_response import (
    DXCForwardResponseConfig,
    DXCForwardResponseDecision,
    rank_dxc_forward_response,
)
from .dxc_discrete_consistency import DXCDiscreteConsistencyConfig
from .dxc_hybrid_model_based import DXCHybridResidualConfig, DXCHybridResidualModel
from .dxc_r27_expected_response import DXCR27Prediction, predict_dxc_r27


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DXCR28Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str = "astrograph-dxc-r28-forward-simulation-prediction.v1"
    predicted_fault_ids: tuple[str, ...]
    first_alarm_time_s: float | None = None
    forward_response: DXCForwardResponseDecision
    r27_prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    algorithm_role: str = "PRIMARY_CAPABLE_DEVELOPMENT_CANDIDATE_LEGACY_RETAINED"
    simulation_semantics: str = "QUALITATIVE_COMPONENT_MODE_FORWARD_SIMULATION"
    agent_required: bool = False
    formal_evidence_eligible: bool = False
    qualification: str = "DEVELOPMENT_ONLY_REQUIRES_NEW_HOLDOUT"
    claim_boundary: str = (
        "R28 retains R27/R26 detection timing and adds a truth-blind qualitative component state-transition/forward-response model. "
        "It is not continuous electrical simulation, does not require Agent execution, and is not formal release evidence."
    )

    @model_validator(mode="after")
    def verify(self) -> "DXCR28Prediction":
        if self.agent_required or self.formal_evidence_eligible:
            raise ValueError("R28 core prediction cannot depend on Agent or become formal evidence")
        payload = {
            "predicted_fault_ids": self.predicted_fault_ids,
            "first_alarm_time_s": self.first_alarm_time_s,
            "forward_response_result_sha256": self.forward_response.result_sha256,
            "r27_prediction_sha256": self.r27_prediction_sha256,
        }
        if _digest(payload) != self.result_sha256:
            raise ValueError("DXC R28 prediction result_sha256 mismatch")
        return self


def assemble_dxc_r28_prediction(
    *,
    records: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    r27_prediction: DXCR27Prediction,
    forward_config: DXCForwardResponseConfig | None = None,
) -> DXCR28Prediction:
    decision = rank_dxc_forward_response(
        records=records,
        catalog=catalog,
        base_fault_ids=r27_prediction.predicted_fault_ids,
        first_alarm_time_s=r27_prediction.first_alarm_time_s,
        config=forward_config,
    )
    payload = {
        "predicted_fault_ids": decision.selected_fault_ids,
        "first_alarm_time_s": r27_prediction.first_alarm_time_s,
        "forward_response_result_sha256": decision.result_sha256,
        "r27_prediction_sha256": r27_prediction.result_sha256,
    }
    return DXCR28Prediction(
        predicted_fault_ids=decision.selected_fault_ids,
        first_alarm_time_s=r27_prediction.first_alarm_time_s,
        forward_response=decision,
        r27_prediction_sha256=r27_prediction.result_sha256,
        result_sha256=_digest(payload),
    )


def predict_dxc_r28(
    *,
    records: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    model: DXCHybridResidualModel,
    continuous_config: DXCHybridResidualConfig,
    discrete_config: DXCDiscreteConsistencyConfig,
    forward_config: DXCForwardResponseConfig | None = None,
) -> DXCR28Prediction:
    r27 = predict_dxc_r27(
        records=records,
        catalog=catalog,
        model=model,
        continuous_config=continuous_config,
        discrete_config=discrete_config,
    )
    return assemble_dxc_r28_prediction(
        records=records,
        catalog=catalog,
        r27_prediction=r27,
        forward_config=forward_config,
    )


__all__ = ["DXCR28Prediction", "assemble_dxc_r28_prediction", "predict_dxc_r28"]
