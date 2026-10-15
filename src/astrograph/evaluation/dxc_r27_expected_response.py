"""R27 DXC primary-candidate adapter: R25/R26 detection plus core expected-response ranking."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.diagnosis.dxc_expected_response import (
    DXCExpectedResponseConfig,
    DXCExpectedResponseDecision,
    rank_dxc_expected_response,
)
from .dxc_discrete_consistency import DXCDiscreteConsistencyConfig, DXCDiscreteConsistencyResult, diagnose_discrete_state_consistency
from .dxc_hybrid_fusion_r26 import DXCFusionPolicy, fuse_dxc_r26
from .dxc_hybrid_model_based import (
    DXCHybridResidualConfig,
    DXCHybridResidualModel,
    DXCHybridPrediction,
    predict_hybrid_model_based,
)


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DXCR27Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: str = "astrograph-dxc-r27-expected-response-prediction.v1"
    predicted_fault_ids: tuple[str, ...]
    first_alarm_time_s: float | None = None
    expected_response: DXCExpectedResponseDecision
    continuous_model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    discrete_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    algorithm_role: str = "PRIMARY_CAPABLE_DEVELOPMENT_CANDIDATE_LEGACY_RETAINED"
    agent_required: bool = False
    formal_evidence_eligible: bool = False
    qualification: str = "DEVELOPMENT_ONLY_REQUIRES_NEW_HOLDOUT"
    claim_boundary: str = (
        "R27 is a deterministic diagnosis-core candidate. It keeps R25/R26 timing and adds component-mode expected-response "
        "ranking. The legacy statistical branch remains available; Agent execution is not required for this prediction."
    )

    @model_validator(mode="after")
    def verify(self) -> "DXCR27Prediction":
        if self.agent_required or self.formal_evidence_eligible:
            raise ValueError("R27 deterministic prediction cannot depend on Agent or become formal evidence")
        payload = {
            "predicted_fault_ids": self.predicted_fault_ids,
            "first_alarm_time_s": self.first_alarm_time_s,
            "expected_response_result_sha256": self.expected_response.result_sha256,
            "continuous_model_sha256": self.continuous_model_sha256,
            "graph_sha256": self.graph_sha256,
            "discrete_result_sha256": self.discrete_result_sha256,
        }
        if _digest(payload) != self.result_sha256:
            raise ValueError("DXC R27 prediction result_sha256 mismatch")
        return self


def assemble_dxc_r27_prediction(
    *,
    records: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    continuous: DXCHybridPrediction,
    discrete: DXCDiscreteConsistencyResult,
    expected_response_config: DXCExpectedResponseConfig | None = None,
) -> DXCR27Prediction:
    """Build the R27 diagnosis from already computed deterministic branches.

    This helper prevents repeated residual/discrete execution in evaluation and
    in future runtime orchestration. It does not alter candidate semantics.
    """
    r26_timing = fuse_dxc_r26(
        continuous, discrete, DXCFusionPolicy.DISCRETE_FIRST_SINGLE_CONTINUOUS
    )
    decision = rank_dxc_expected_response(
        records=records,
        catalog=catalog,
        continuous_fault_ids=continuous.predicted_fault_ids,
        discrete_events=[item.model_dump(mode="json") for item in discrete.events],
        discrete_fault_ids=discrete.predicted_fault_ids,
        discrete_first_event_time_s=discrete.first_event_time_s,
        config=expected_response_config,
    )
    payload = {
        "predicted_fault_ids": decision.selected_fault_ids,
        "first_alarm_time_s": r26_timing.first_alarm_time_s,
        "expected_response_result_sha256": decision.result_sha256,
        "continuous_model_sha256": continuous.model_sha256,
        "graph_sha256": continuous.graph_sha256,
        "discrete_result_sha256": discrete.result_sha256,
    }
    return DXCR27Prediction(
        predicted_fault_ids=decision.selected_fault_ids,
        first_alarm_time_s=r26_timing.first_alarm_time_s,
        expected_response=decision,
        continuous_model_sha256=continuous.model_sha256,
        graph_sha256=continuous.graph_sha256,
        discrete_result_sha256=discrete.result_sha256,
        result_sha256=_digest(payload),
    )


def predict_dxc_r27(
    *,
    records: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    model: DXCHybridResidualModel,
    continuous_config: DXCHybridResidualConfig,
    discrete_config: DXCDiscreteConsistencyConfig,
    expected_response_config: DXCExpectedResponseConfig | None = None,
) -> DXCR27Prediction:
    continuous = predict_hybrid_model_based(records, catalog, model, continuous_config)
    discrete = diagnose_discrete_state_consistency(records, catalog, discrete_config)
    return assemble_dxc_r27_prediction(
        records=records,
        catalog=catalog,
        continuous=continuous,
        discrete=discrete,
        expected_response_config=expected_response_config,
    )


__all__ = ["DXCR27Prediction", "assemble_dxc_r27_prediction", "predict_dxc_r27"]
