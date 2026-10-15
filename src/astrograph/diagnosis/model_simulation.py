"""Integrated structural-model diagnosis plus optional simulation review.

The service composes R23 deterministic primitives without granting any new
EvidenceFact or Agent authority.  It is deliberately capable of stopping after
model-based diagnosis when no governed simulator/trace provider is admitted.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.diagnosis.model_based import MinimalDiagnosisResult, enumerate_minimal_diagnoses
from astrograph.knowledge.system_model_graph import (
    ObservationMismatch,
    SystemModelGraph,
    structural_conflicts_from_mismatches,
)
from astrograph.simulation.diagnostic_provider import (
    CounterfactualDiagnosisReview,
    DiagnosticSimulationProvider,
    SimulationTrace,
    compare_counterfactual_diagnoses,
)


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class IntegratedModelSimulationDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-integrated-model-simulation-diagnosis.v1"
    system_model_graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mismatch_count: int = Field(ge=1)
    diagnosis: MinimalDiagnosisResult
    counterfactual_review: CounterfactualDiagnosisReview | None = None
    status: str
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    formal_evidence_eligible: bool = False
    claim_boundary: str = (
        "Engineering diagnosis review only. Structural conflicts constrain candidate sets and optional simulation "
        "distance ranks admitted counterfactual traces. Neither stage is a posterior probability, formal EvidenceFact, "
        "flight-qualified causal proof, or release authority."
    )

    @model_validator(mode="after")
    def verify(self) -> "IntegratedModelSimulationDiagnosis":
        if self.formal_evidence_eligible:
            raise ValueError("R23 integrated model/simulation diagnosis is shadow engineering evidence only")
        payload = {
            "system_model_graph_sha256": self.system_model_graph_sha256,
            "mismatch_count": self.mismatch_count,
            "diagnosis_result_sha256": self.diagnosis.result_sha256,
            "counterfactual_result_sha256": (
                self.counterfactual_review.result_sha256 if self.counterfactual_review else None
            ),
            "status": self.status,
        }
        if _digest(payload) != self.result_sha256:
            raise ValueError("integrated model/simulation diagnosis result hash mismatch")
        return self


def run_model_simulation_diagnosis(
    *,
    graph: SystemModelGraph,
    mismatches: Sequence[ObservationMismatch],
    max_topology_hops: int = 1,
    max_cardinality: int = 4,
    max_candidates: int = 128,
    max_search_nodes: int = 100_000,
    observed_trace: SimulationTrace | None = None,
    simulation_provider: DiagnosticSimulationProvider | None = None,
    channel_scales: Mapping[str, float] | None = None,
    simulation_channels: Sequence[str] | None = None,
) -> IntegratedModelSimulationDiagnosis:
    if not mismatches:
        raise ValueError("integrated diagnosis requires at least one admitted mismatch")
    conflicts = structural_conflicts_from_mismatches(
        graph, list(mismatches), max_topology_hops=max_topology_hops
    )
    diagnosis = enumerate_minimal_diagnoses(
        conflicts,
        max_cardinality=max_cardinality,
        max_candidates=max_candidates,
        max_search_nodes=max_search_nodes,
    )
    review: CounterfactualDiagnosisReview | None = None
    provided_simulation_inputs = [
        observed_trace is not None,
        simulation_provider is not None,
        channel_scales is not None,
    ]
    if any(provided_simulation_inputs) and not all(provided_simulation_inputs):
        raise ValueError("observed_trace, simulation_provider and channel_scales must be admitted together")
    if all(provided_simulation_inputs):
        assert observed_trace is not None and simulation_provider is not None and channel_scales is not None
        if not diagnosis.subset_minimal_diagnoses:
            raise ValueError("counterfactual review requires at least one discovered diagnosis candidate")
        review = compare_counterfactual_diagnoses(
            observed=observed_trace,
            candidates=diagnosis.subset_minimal_diagnoses,
            provider=simulation_provider,
            channel_scales=channel_scales,
            channels=simulation_channels,
        )
    if not diagnosis.search_complete:
        status = "MODEL_DIAGNOSIS_PARTIAL"
    elif review is None:
        status = "MODEL_DIAGNOSIS_COMPLETE_SIMULATION_NOT_ADMITTED"
    else:
        status = "MODEL_DIAGNOSIS_COMPLETE_WITH_COUNTERFACTUAL_REVIEW"
    payload = {
        "system_model_graph_sha256": graph.graph_sha256,
        "mismatch_count": len(mismatches),
        "diagnosis_result_sha256": diagnosis.result_sha256,
        "counterfactual_result_sha256": review.result_sha256 if review else None,
        "status": status,
    }
    return IntegratedModelSimulationDiagnosis(
        system_model_graph_sha256=graph.graph_sha256,
        mismatch_count=len(mismatches),
        diagnosis=diagnosis,
        counterfactual_review=review,
        status=status,
        result_sha256=_digest(payload),
    )


__all__ = ["IntegratedModelSimulationDiagnosis", "run_model_simulation_diagnosis"]
