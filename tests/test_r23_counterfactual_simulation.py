from __future__ import annotations

import pytest

from astrograph.diagnosis.model_based import MinimalDiagnosisCandidate
from astrograph.simulation.diagnostic_provider import (
    PrecomputedTraceSimulationProvider,
    build_simulation_trace,
    compare_counterfactual_diagnoses,
)


def _candidate(name: str, fault: str) -> MinimalDiagnosisCandidate:
    return MinimalDiagnosisCandidate(
        diagnosis_id=name, fault_ids=(fault,), cardinality=1, minimum_cardinality=True,
    )


def test_precomputed_simulation_review_ranks_trace_fit_without_probability_claim():
    observed = build_simulation_trace(
        trace_id="OBS", time_s=[0.0, 1.0, 2.0], channels={"v": [0.0, 1.0, 2.0]},
        source_artifact_sha256="a" * 64,
    )
    a = _candidate("D-A", "CB1:FailedOpen")
    b = _candidate("D-B", "E1:Offset")
    provider = PrecomputedTraceSimulationProvider({
        "D-A": build_simulation_trace(
            trace_id="A", time_s=[0.0, 1.0, 2.0], channels={"v": [0.0, 1.1, 2.1]},
            source_artifact_sha256="b" * 64,
        ),
        "D-B": build_simulation_trace(
            trace_id="B", time_s=[0.0, 1.0, 2.0], channels={"v": [0.0, 4.0, 5.0]},
            source_artifact_sha256="c" * 64,
        ),
    })
    result = compare_counterfactual_diagnoses(
        observed=observed, candidates=[a, b], provider=provider, channel_scales={"v": 1.0}
    )
    assert result.best_diagnosis_id == "D-A"
    assert result.comparisons[0].score_semantics == "DISTANCE_LOWER_IS_BETTER_NOT_PROBABILITY"
    assert result.formal_evidence_eligible is False


def test_simulation_review_rejects_timeline_drift():
    observed = build_simulation_trace(
        trace_id="OBS", time_s=[0.0, 1.0], channels={"v": [0.0, 1.0]}, source_artifact_sha256="a" * 64,
    )
    candidate = _candidate("D-A", "CB1:FailedOpen")
    provider = PrecomputedTraceSimulationProvider({
        "D-A": build_simulation_trace(
            trace_id="A", time_s=[0.0, 2.0], channels={"v": [0.0, 1.0]}, source_artifact_sha256="b" * 64,
        )
    })
    with pytest.raises(ValueError, match="timeline"):
        compare_counterfactual_diagnoses(
            observed=observed, candidates=[candidate], provider=provider, channel_scales={"v": 1.0}
        )
