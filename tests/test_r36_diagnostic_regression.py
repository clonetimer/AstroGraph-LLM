"""Reconstructed R36.1/R36.7 executable benchmark contract tests."""
import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from astrograph.evaluation.diagnostic_regression import (
    BenchmarkCase,
    EvidenceClass,
    QualificationAuthorizationReceipt,
    build_regression_board,
    evaluate_diagnostic_benchmark,
)


def cases():
    return [
        BenchmarkCase(
            case_id="TP", true_detected=True, predicted_detected=True, duration_s=100,
            true_onset_s=40, detection_time_s=45,
            true_component_id="RW1", predicted_component_id="RW1",
            true_fault_mode_id="FRICTION", predicted_fault_mode_id="FRICTION",
        ),
        BenchmarkCase(
            case_id="FN", true_detected=True, predicted_detected=False, duration_s=100,
            true_onset_s=50, true_component_id="RW2", true_fault_mode_id="JAM",
            abstained=True,
        ),
        BenchmarkCase(case_id="FP", true_detected=False, predicted_detected=True, duration_s=100, detection_time_s=80),
        BenchmarkCase(case_id="TN", true_detected=False, predicted_detected=False, duration_s=100),
    ]


def test_three_task_metrics_and_development_boundary():
    result = evaluate_diagnostic_benchmark(
        astrograph_version="0.7.1.post49+r36-local", benchmark_id="ADCS",
        dataset_id="D", protocol_id="P", cohort_id="C", source_artifact_sha256="a" * 64,
        evidence_class=EvidenceClass.DEVELOPMENT, cases=cases(),
    )
    assert result.metrics.detection.TP == 1
    assert result.metrics.detection.FP == 1
    assert result.metrics.detection.FN == 1
    assert result.metrics.detection.TN == 1
    assert result.metrics.detection.precision == pytest.approx(0.5)
    assert result.metrics.detection.recall == pytest.approx(0.5)
    assert result.metrics.detection.F1 == pytest.approx(0.5)
    assert result.metrics.detection.latency_mean_s == pytest.approx(5.0)
    assert result.metrics.detection.fault_pre_onset_alarm_count == 0
    assert result.metrics.localization.evaluated_count == 2
    assert result.metrics.localization.accuracy == pytest.approx(0.5)
    assert result.promotion_eligible is False


def test_pre_onset_alarm_is_reported_separately():
    item = BenchmarkCase(
        case_id="early", true_detected=True, predicted_detected=True,
        duration_s=100, true_onset_s=40, first_alarm_time_s=20,
        detection_time_s=50,
    )
    result = evaluate_diagnostic_benchmark(
        astrograph_version="v", benchmark_id="B", dataset_id="D",
        protocol_id="P", cohort_id="C", source_artifact_sha256="a" * 64,
        evidence_class=EvidenceClass.DEVELOPMENT, cases=[item],
    )
    assert result.metrics.detection.fault_pre_onset_alarm_count == 1
    assert result.metrics.detection.latency_mean_s == pytest.approx(10.0)


def test_qualification_requires_verified_identity_bound_receipt():
    kwargs = dict(
        astrograph_version="v", benchmark_id="B", dataset_id="D",
        protocol_id="P", cohort_id="C", source_artifact_sha256="a" * 64,
        evidence_class=EvidenceClass.QUALIFICATION, cases=cases(),
    )
    with pytest.raises(ValueError, match="authorization receipt"):
        evaluate_diagnostic_benchmark(**kwargs)
    receipt = QualificationAuthorizationReceipt(
        receipt_id="QR-1", issuer="governance-gate",
        issued_at=datetime.now(timezone.utc), dataset_id="D", protocol_id="P",
        cohort_id="C", source_artifact_sha256="a" * 64, signature="external-signature",
    )
    result = evaluate_diagnostic_benchmark(
        **kwargs, qualification_receipt=receipt,
        qualification_receipt_validator=lambda value: value.signature == "external-signature",
    )
    assert result.promotion_eligible is True
    assert result.qualification_receipt_id == "QR-1"


def test_board_keeps_missing_and_development_not_promotable():
    snapshot = evaluate_diagnostic_benchmark(
        astrograph_version="v", benchmark_id="ADCS", dataset_id="D", protocol_id="P", cohort_id="C",
        source_artifact_sha256="a" * 64, evidence_class=EvidenceClass.DEVELOPMENT, cases=cases(),
    )
    board = build_regression_board(
        astrograph_version="v", benchmark_ids=["ADCS", "ESA_MISSION1"], snapshots=[snapshot]
    )
    assert board.rows[0].status == "COMPLETE"
    assert board.rows[0].promotion_eligible is False
    assert board.rows[1].status == "MISSING"
    assert board.promotion_eligible is False


@pytest.mark.parametrize("payload", [
    dict(case_id="bad", true_detected=True, predicted_detected=True, duration_s=10, true_onset_s=11, detection_time_s=11),
    dict(case_id="bad", true_detected=True, predicted_detected=True, duration_s=10, true_onset_s=5, detection_time_s=11),
    dict(case_id="bad", true_detected=True, predicted_detected=True, duration_s=10, true_onset_s=5, detection_time_s=4),
    dict(case_id="bad", true_detected=False, predicted_detected=False, duration_s=10, true_fault_mode_id="X"),
    dict(case_id="bad", true_detected=False, predicted_detected=False, duration_s=10, detection_time_s=1),
])
def test_invalid_case_contracts_fail_closed(payload):
    with pytest.raises(ValidationError):
        BenchmarkCase(**payload)
