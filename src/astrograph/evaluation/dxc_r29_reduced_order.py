"""R29 adapter: R28 diagnosis plus selective reduced-order physical simulation."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.diagnosis.dxc_reduced_order_physics import (
    DXCReducedOrderPhysicsConfig,
    DXCReducedOrderPhysicsDecision,
    rank_dxc_reduced_order_physics,
)
from .dxc_discrete_consistency import DXCDiscreteConsistencyConfig
from .dxc_hybrid_model_based import DXCHybridResidualConfig, DXCHybridResidualModel
from .dxc_r28_forward_simulation import DXCR28Prediction, predict_dxc_r28


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DXCR29Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str = "astrograph-dxc-r29-reduced-order-simulation-prediction.v1"
    predicted_fault_ids: tuple[str, ...]
    first_alarm_time_s: float | None = None
    reduced_order_physics: DXCReducedOrderPhysicsDecision
    r28_prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    algorithm_role: str = "PRIMARY_CAPABLE_DEVELOPMENT_CANDIDATE_R28_RETAINED"
    simulation_semantics: str = "REDUCED_ORDER_ELECTRICAL_LOAD_TIME_SERIES_FORWARD_SIMULATION"
    agent_required: bool = False
    formal_evidence_eligible: bool = False
    qualification: str = "DEVELOPMENT_ONLY_REQUIRES_NEW_HOLDOUT"
    claim_boundary: str = (
        "R29 preserves R28/R27 detection timing and resolved diagnoses. Only an R28 generic unresolved physical-response disposition "
        "may enter the reduced-order ADAPT light-load simulator. The core runs with Agent OFF and is not formal release evidence."
    )

    @model_validator(mode="after")
    def verify(self) -> "DXCR29Prediction":
        if self.agent_required or self.formal_evidence_eligible:
            raise ValueError("R29 core prediction cannot depend on Agent or become formal evidence")
        payload = {
            "predicted_fault_ids": self.predicted_fault_ids,
            "first_alarm_time_s": self.first_alarm_time_s,
            "reduced_order_result_sha256": self.reduced_order_physics.result_sha256,
            "r28_prediction_sha256": self.r28_prediction_sha256,
        }
        if _digest(payload) != self.result_sha256:
            raise ValueError("DXC R29 prediction result_sha256 mismatch")
        return self


def assemble_dxc_r29_prediction(
    *,
    records: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    r28_prediction: DXCR28Prediction,
    physics_config: DXCReducedOrderPhysicsConfig | None = None,
) -> DXCR29Prediction:
    decision = rank_dxc_reduced_order_physics(
        records=records,
        catalog=catalog,
        base_fault_ids=r28_prediction.predicted_fault_ids,
        base_unresolved_reason_codes=r28_prediction.forward_response.unresolved_reason_codes,
        first_alarm_time_s=r28_prediction.first_alarm_time_s,
        config=physics_config,
    )
    payload = {
        "predicted_fault_ids": decision.selected_fault_ids,
        "first_alarm_time_s": r28_prediction.first_alarm_time_s,
        "reduced_order_result_sha256": decision.result_sha256,
        "r28_prediction_sha256": r28_prediction.result_sha256,
    }
    return DXCR29Prediction(
        predicted_fault_ids=decision.selected_fault_ids,
        first_alarm_time_s=r28_prediction.first_alarm_time_s,
        reduced_order_physics=decision,
        r28_prediction_sha256=r28_prediction.result_sha256,
        result_sha256=_digest(payload),
    )


def predict_dxc_r29(
    *,
    records: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    model: DXCHybridResidualModel,
    continuous_config: DXCHybridResidualConfig,
    discrete_config: DXCDiscreteConsistencyConfig,
    physics_config: DXCReducedOrderPhysicsConfig | None = None,
) -> DXCR29Prediction:
    r28 = predict_dxc_r28(
        records=records,
        catalog=catalog,
        model=model,
        continuous_config=continuous_config,
        discrete_config=discrete_config,
    )
    return assemble_dxc_r29_prediction(
        records=records,
        catalog=catalog,
        r28_prediction=r28,
        physics_config=physics_config,
    )


__all__ = ["DXCR29Prediction", "assemble_dxc_r29_prediction", "predict_dxc_r29"]
