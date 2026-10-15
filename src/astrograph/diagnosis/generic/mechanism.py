from __future__ import annotations

from astrograph.state_estimation import StateEstimateResult

from .contracts import (
    AnomalyDecision,
    AnomalyHeadResult,
    EffectPattern,
    FailureMechanism,
    HeadCalibrationStatus,
    HeadEvidence,
    HeadExecutionStatus,
    HeadReceipt,
    MechanismCandidate,
    MechanismDecision,
    MechanismHeadResult,
    stable_id,
)


class MechanismHead:
    """Produce generic mechanism candidates without promoting an uncalibrated exact cause."""

    head_id = "astrograph.generic.mechanism.balance-pattern"
    head_version = "1.1.0"

    def evaluate(
        self,
        *,
        episode_id: str,
        state_estimate: StateEstimateResult,
        anomaly: AnomalyHeadResult,
    ) -> MechanismHeadResult:
        receipt = HeadReceipt(
            head_id=self.head_id,
            head_version=self.head_version,
            calibration_status=HeadCalibrationStatus.NOT_CALIBRATED,
            input_sha256=state_estimate.receipt.identification_data_sha256,
        )
        result_id = stable_id(
            "MH",
            {
                "episode_id": episode_id,
                "anomaly_result_id": anomaly.anomaly_result_id,
                "input_sha256": receipt.input_sha256,
            },
        )
        if anomaly.decision in {
            AnomalyDecision.DATA_INVALID,
            AnomalyDecision.INSUFFICIENT_EVIDENCE,
        }:
            return MechanismHeadResult(
                mechanism_result_id=result_id,
                episode_id=episode_id,
                execution_status=HeadExecutionStatus.NOT_EVALUATED,
                decision=MechanismDecision.NOT_EVALUATED,
                effect_patterns=[],
                candidates=[],
                evidence=[],
                receipt=receipt,
            )
        if anomaly.decision == AnomalyDecision.NORMAL:
            return MechanismHeadResult(
                mechanism_result_id=result_id,
                episode_id=episode_id,
                execution_status=HeadExecutionStatus.COMPLETED,
                decision=MechanismDecision.NO_ANOMALY,
                effect_patterns=[EffectPattern.CALIBRATED_NO_DEVIATION],
                candidates=[],
                evidence=[],
                receipt=receipt,
            )

        balance = next(
            (
                item
                for item in state_estimate.residuals
                if item.residual_id == "balance_equation"
            ),
            None,
        )
        if balance is None:
            return MechanismHeadResult(
                mechanism_result_id=result_id,
                episode_id=episode_id,
                execution_status=HeadExecutionStatus.COMPLETED,
                decision=MechanismDecision.INSUFFICIENT_EVIDENCE,
                effect_patterns=[],
                candidates=[],
                evidence=[],
                receipt=receipt,
            )
        pattern_evidence_id = stable_id(
            "EV",
            {
                "episode_id": episode_id,
                "type": "persistent_balance_deviation",
                "mean": balance.mean,
                "rms": balance.root_mean_square,
            },
        )
        evidence = [
            HeadEvidence(
                evidence_id=pattern_evidence_id,
                evidence_type="EFFECT_PATTERN",
                source_reference="state_estimate.residuals.balance_equation",
                summary=(
                    f"Noise-aware balance deviation is significant; raw residual mean={balance.mean:.6g} "
                    f"and RMS={balance.root_mean_square:.6g}."
                ),
            )
        ]
        patterns = [EffectPattern.PERSISTENT_BALANCE_DEVIATION]
        if anomaly.raw_violation_count:
            patterns.append(EffectPattern.RAW_POINTWISE_TOLERANCE_EXCEEDANCE)
        candidate = MechanismCandidate(
            mechanism_id=FailureMechanism.UNMODELED_BALANCE_DISTURBANCE,
            screening_score=anomaly.anomaly_score,
            calibrated=False,
            evidence_ids=[pattern_evidence_id],
        )
        return MechanismHeadResult(
            mechanism_result_id=result_id,
            episode_id=episode_id,
            execution_status=HeadExecutionStatus.COMPLETED,
            decision=MechanismDecision.CANDIDATES,
            effect_patterns=patterns,
            candidates=[candidate],
            evidence=evidence,
            receipt=receipt,
        )


__all__ = ["MechanismHead"]
