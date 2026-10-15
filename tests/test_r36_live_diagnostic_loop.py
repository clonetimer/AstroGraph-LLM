"""Reconstructed R36.5 persistence/window contract tests from the source handoff."""
import json

import pytest
from pydantic import ValidationError

from astrograph.diagnosis.generic import (
    AnomalyDecision,
    AnomalyHeadResult,
    GenericDecisionRecord,
    GenericDiagnosisResult,
    GenericDisposition,
    HeadCalibrationStatus,
    HeadExecutionStatus,
    HeadReceipt,
    LocalizationDecision,
    LocalizationHeadResult,
    MechanismDecision,
    MechanismHeadResult,
    ReasoningDecision,
    ReasoningHeadResult,
)
from astrograph.diagnosis.live_loop import (
    AppendOnlyOperatorPacketStore,
    LiveDiagnosticWindow,
    OperatorDiagnosticPacket,
)
from astrograph.state_estimation import (
    ConservationAssessment,
    ModelReceipt,
    StateEstimateResult,
    UncertaintySummary,
)


def state_estimate() -> StateEstimateResult:
    return StateEstimateResult(
        component_archetype_id="TEST_ARCHETYPE",
        timestamps_s=[0.0, 1.0], states=[], parameters=[], residuals=[],
        uncertainty=UncertaintySummary(residual_standard_deviation=0.0),
        ood_score=0.0, constraint_violations=[],
        conservation_assessment=ConservationAssessment(
            method="UNCALIBRATED_RAW_ONLY", raw_physical_tolerance=1.0,
            raw_violation_count=0, raw_violation_fraction=0.0, episode_score=0.0,
            empirical_p_value=1.0, p_value_upper_confidence=1.0,
            false_alarm_rate=0.05, bootstrap_replicates=0,
            statistically_significant=False, calibration_version="NOT_CALIBRATED",
        ),
        receipt=ModelReceipt(
            estimator_id="test", estimator_version="1", dynamics_contract_id="d",
            dynamics_contract_sha256="0" * 64, identification_data_sha256="1" * 64,
        ),
    )


def diagnosis() -> GenericDiagnosisResult:
    receipt = HeadReceipt(
        head_id="h", head_version="1", calibration_status=HeadCalibrationStatus.DETERMINISTIC_CONTRACT,
        input_sha256="2" * 64,
    )
    return GenericDiagnosisResult(
        episode_id="ep",
        anomaly=AnomalyHeadResult(
            anomaly_result_id="a", episode_id="ep", decision=AnomalyDecision.NORMAL,
            anomaly_score=0.0, decision_threshold=0.5, raw_violation_count=0,
            raw_violation_fraction=0.0, evidence=[], receipt=receipt,
        ),
        mechanism=MechanismHeadResult(
            mechanism_result_id="m", episode_id="ep", execution_status=HeadExecutionStatus.COMPLETED,
            decision=MechanismDecision.NO_ANOMALY, effect_patterns=[], candidates=[], evidence=[], receipt=receipt,
        ),
        localization=LocalizationHeadResult(
            localization_result_id="l", episode_id="ep", execution_status=HeadExecutionStatus.COMPLETED,
            decision=LocalizationDecision.NO_ANOMALY, candidates=[], evidence=[], receipt=receipt,
        ),
        reasoning=ReasoningHeadResult(
            reasoning_result_id="r", episode_id="ep", execution_status=HeadExecutionStatus.COMPLETED,
            decision=ReasoningDecision.NO_ANOMALY, hypotheses=[], receipt=receipt,
        ),
        decision=GenericDecisionRecord(
            decision_id="d", episode_id="ep", disposition=GenericDisposition.NORMAL,
            reason_codes=[], unresolved_requirements=[],
        ),
    )


def packet(case_id="C", window_id="W", start=0.0, end=2.0):
    return OperatorDiagnosticPacket(
        case_id=case_id, window_id=window_id, start_s=start, end_s=end,
        semantic_data_sha256="3" * 64,
        state_estimate_sha256="4" * 64,
        diagnosis_sha256="5" * 64,
        state_estimate=state_estimate(), diagnosis=diagnosis(),
    )


def test_packet_store_excludes_computed_hash_and_round_trips(tmp_path):
    store = AppendOnlyOperatorPacketStore(tmp_path / "packets.jsonl")
    original = packet()
    store.append(original)
    raw = json.loads((tmp_path / "packets.jsonl").read_text())
    assert "packet_sha256" not in raw
    restored = store.packets()
    assert len(restored) == 1
    assert restored[0].packet_sha256 == original.packet_sha256


def test_packet_store_rejects_overlapping_windows(tmp_path):
    store = AppendOnlyOperatorPacketStore(tmp_path / "packets.jsonl")
    store.append(packet(window_id="W1", start=0.0, end=2.0))
    with pytest.raises(ValueError, match="OVERLAPPING"):
        store.append(packet(window_id="W2", start=1.0, end=3.0))


def test_live_window_requires_positive_order():
    with pytest.raises(ValidationError):
        LiveDiagnosticWindow(case_id="C", window_id="W", start_s=2.0, end_s=2.0)
