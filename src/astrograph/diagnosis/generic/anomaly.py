from __future__ import annotations

from astrograph.state_estimation import StateEstimateResult

from .contracts import (
    AnomalyDecision,
    AnomalyHeadResult,
    DataQualityContext,
    DataQualityStatus,
    HeadCalibrationStatus,
    HeadEvidence,
    HeadReceipt,
    stable_id,
)


class AnomalyHead:
    head_id = "astrograph.generic.anomaly.physics-adequacy"
    head_version = "1.0.0"

    def evaluate(
        self,
        *,
        episode_id: str,
        state_estimate: StateEstimateResult,
        data_quality: DataQualityContext,
    ) -> AnomalyHeadResult:
        assessment = state_estimate.conservation_assessment
        calibrated = (
            assessment.method != "UNCALIBRATED_RAW_ONLY"
            and state_estimate.receipt.conservation_calibration_version
            != "NOT_CALIBRATED"
        )
        receipt = HeadReceipt(
            head_id=self.head_id,
            head_version=self.head_version,
            calibration_status=(
                HeadCalibrationStatus.CALIBRATED
                if calibrated
                else HeadCalibrationStatus.NOT_CALIBRATED
            ),
            calibration_id=(
                state_estimate.receipt.estimator_id if calibrated else "NONE"
            ),
            calibration_version=(
                state_estimate.receipt.conservation_calibration_version
                if calibrated
                else "NONE"
            ),
            calibration_sha256=(
                state_estimate.receipt.uncertainty_calibration_sha256
                if calibrated
                else "NONE"
            ),
            input_sha256=state_estimate.receipt.identification_data_sha256,
        )
        raw_evidence_id = stable_id(
            "EV",
            {
                "episode_id": episode_id,
                "type": "raw_conservation_receipt",
                "count": assessment.raw_violation_count,
                "fraction": assessment.raw_violation_fraction,
            },
        )
        statistical_evidence_id = stable_id(
            "EV",
            {
                "episode_id": episode_id,
                "type": "noise_aware_conservation",
                "score": assessment.episode_score,
                "p_upper": assessment.p_value_upper_confidence,
            },
        )
        evidence = [
            HeadEvidence(
                evidence_id=raw_evidence_id,
                evidence_type="RAW_CONSERVATION_RECEIPT",
                source_reference="state_estimate.constraint_violations",
                summary=(
                    f"Preserved {assessment.raw_violation_count} raw pointwise violations "
                    f"(fraction={assessment.raw_violation_fraction:.6f})."
                ),
            ),
            HeadEvidence(
                evidence_id=statistical_evidence_id,
                evidence_type="NOISE_AWARE_CONSERVATION_ASSESSMENT",
                source_reference="state_estimate.conservation_assessment",
                summary=(
                    f"Episode score={assessment.episode_score:.6g}; Monte Carlo p-value upper "
                    f"confidence={assessment.p_value_upper_confidence:.6g}."
                ),
            ),
        ]
        threshold = 1.0 - assessment.false_alarm_rate
        anomaly_score = 1.0 - assessment.p_value_upper_confidence
        if data_quality.status == DataQualityStatus.INVALID:
            decision = AnomalyDecision.DATA_INVALID
        elif not calibrated:
            decision = AnomalyDecision.INSUFFICIENT_EVIDENCE
        elif assessment.statistically_significant:
            decision = AnomalyDecision.ANOMALOUS
        else:
            decision = AnomalyDecision.NORMAL
        result_id = stable_id(
            "AH",
            {
                "episode_id": episode_id,
                "decision": decision.value,
                "input_sha256": receipt.input_sha256,
            },
        )
        return AnomalyHeadResult(
            anomaly_result_id=result_id,
            episode_id=episode_id,
            decision=decision,
            anomaly_score=anomaly_score,
            decision_threshold=threshold,
            raw_violation_count=assessment.raw_violation_count,
            raw_violation_fraction=assessment.raw_violation_fraction,
            evidence=evidence,
            receipt=receipt,
        )


__all__ = ["AnomalyHead"]
