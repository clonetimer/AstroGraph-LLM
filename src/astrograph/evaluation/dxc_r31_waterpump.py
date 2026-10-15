"""R31 integration: R29 diagnosis plus WaterPump physical corroboration.

R31 does not replace or auto-correct the R29 selected fault set.  The added
WaterPump reduced-order model is a deterministic primary-path corroborator that
runs whenever the selected R29 diagnosis is one single WaterPump mode and a
transition time is available.  Agent review is optional and downstream.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.diagnosis.dxc_pump_reduced_order import (
    DXCWaterPumpPhysicsConfig,
    DXCWaterPumpPhysicsDecision,
    evaluate_dxc_water_pump_physics,
)
from astrograph.evaluation.dxc_r29_reduced_order import DXCR29Prediction
from astrograph.simulation.adapt_pump_reduced_order import ADAPTWaterPumpTypeParameters


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DXCR31Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str = "astrograph-dxc-r31-waterpump-physics-prediction.v1"
    predicted_fault_ids: tuple[str, ...]
    first_alarm_time_s: float | None = None
    waterpump_physics: DXCWaterPumpPhysicsDecision
    r29_prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    algorithm_role: str = "PRIMARY_PATH_PHYSICS_CORROBORATOR_NO_AUTO_OVERRIDE"
    agent_required: bool = False
    formal_evidence_eligible: bool = False
    qualification: str = "DEVELOPMENT_CORROBORATOR_REQUIRES_NEW_HOLDOUT"
    claim_boundary: str = (
        "R31 preserves the R29 selected fault set and adds truth-blind WaterPump current/flow physical consistency. "
        "A disagreement is surfaced as a deterministic contradiction for later governed review; it does not silently rewrite the diagnosis."
    )

    @model_validator(mode="after")
    def verify(self) -> "DXCR31Prediction":
        if self.agent_required or self.formal_evidence_eligible:
            raise ValueError("R31_PREDICTION_AGENT_OR_FORMAL_EVIDENCE_FORBIDDEN")
        if self.predicted_fault_ids != self.waterpump_physics.selected_fault_ids:
            raise ValueError("R31_PREDICTION_MUST_PRESERVE_CORE_SELECTED_FAULTS")
        payload = {
            "predicted_fault_ids": self.predicted_fault_ids,
            "first_alarm_time_s": self.first_alarm_time_s,
            "waterpump_result_sha256": self.waterpump_physics.result_sha256,
            "r29_prediction_sha256": self.r29_prediction_sha256,
        }
        if _digest(payload) != self.result_sha256:
            raise ValueError("R31_PREDICTION_SHA256_MISMATCH")
        return self


def assemble_dxc_r31_prediction(
    *,
    records: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    type_parameters: ADAPTWaterPumpTypeParameters,
    r29_prediction: DXCR29Prediction,
    pump_config: DXCWaterPumpPhysicsConfig | None = None,
) -> DXCR31Prediction:
    decision = evaluate_dxc_water_pump_physics(
        records=records,
        catalog=catalog,
        type_parameters=type_parameters,
        base_fault_ids=r29_prediction.predicted_fault_ids,
        transition_time_s=r29_prediction.first_alarm_time_s,
        config=pump_config,
    )
    payload = {
        "predicted_fault_ids": decision.selected_fault_ids,
        "first_alarm_time_s": r29_prediction.first_alarm_time_s,
        "waterpump_result_sha256": decision.result_sha256,
        "r29_prediction_sha256": r29_prediction.result_sha256,
    }
    return DXCR31Prediction(
        predicted_fault_ids=decision.selected_fault_ids,
        first_alarm_time_s=r29_prediction.first_alarm_time_s,
        waterpump_physics=decision,
        r29_prediction_sha256=r29_prediction.result_sha256,
        result_sha256=_digest(payload),
    )


__all__ = ["DXCR31Prediction", "assemble_dxc_r31_prediction"]
