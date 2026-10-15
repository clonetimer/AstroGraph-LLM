from __future__ import annotations

from .contracts import (
    AnomalyDecision,
    AnomalyHeadResult,
    DataQualityContext,
    DataQualityStatus,
    GenericDecisionRecord,
    GenericDisposition,
    HypothesisStatus,
    LocalizationHeadResult,
    MechanismHeadResult,
    OpenSetDecision,
    ReasoningHeadResult,
    stable_id,
)


class GenericDecisionGovernor:
    governor_id = "astrograph.generic.decision-governor"
    governor_version = "1.0.0"

    def decide(
        self,
        *,
        episode_id: str,
        data_quality: DataQualityContext,
        anomaly: AnomalyHeadResult,
        mechanism: MechanismHeadResult,
        localization: LocalizationHeadResult,
        reasoning: ReasoningHeadResult,
    ) -> GenericDecisionRecord:
        reasons: list[str] = []
        unresolved: list[str] = []
        accepted_hypothesis_id: str | None = None

        if data_quality.status == DataQualityStatus.INVALID:
            disposition = GenericDisposition.DATA_INVALID
            reasons.append("DATA_QUALITY_GATE_FAILED")
            unresolved.extend(data_quality.reason_codes)
        elif anomaly.decision == AnomalyDecision.INSUFFICIENT_EVIDENCE:
            disposition = GenericDisposition.INSUFFICIENT_EVIDENCE
            reasons.append("ANOMALY_HEAD_NOT_CALIBRATED")
            unresolved.append("ANOMALY_CALIBRATION")
        elif anomaly.decision == AnomalyDecision.NORMAL:
            disposition = GenericDisposition.NORMAL
            reasons.append("CALIBRATED_ANOMALY_HEAD_NORMAL")
        elif anomaly.decision == AnomalyDecision.DATA_INVALID:
            disposition = GenericDisposition.DATA_INVALID
            reasons.append("ANOMALY_HEAD_REJECTED_INVALID_DATA")
        else:
            supported = [
                item
                for item in reasoning.hypotheses
                if item.status == HypothesisStatus.SUPPORTED
            ]
            unresolved.extend(
                requirement
                for item in reasoning.hypotheses
                for requirement in item.missing_requirements
            )
            if mechanism.open_set_decision == OpenSetDecision.NOT_EVALUATED:
                disposition = GenericDisposition.INSUFFICIENT_EVIDENCE
                reasons.append("OPEN_SET_OOD_NOT_EVALUATED")
                unresolved.append("OPEN_SET_OOD_CALIBRATION")
            elif mechanism.open_set_decision == OpenSetDecision.OOD:
                disposition = GenericDisposition.OOD
                reasons.append("CALIBRATED_OPEN_SET_REJECTION")
            elif len(supported) > 1:
                disposition = GenericDisposition.AMBIGUOUS
                reasons.append("MULTIPLE_SUPPORTED_HYPOTHESES")
            elif len(supported) == 1:
                disposition = GenericDisposition.DIAGNOSED
                accepted_hypothesis_id = supported[0].hypothesis_id
                reasons.append("ONE_CALIBRATED_SUPPORTED_HYPOTHESIS")
            else:
                disposition = GenericDisposition.INSUFFICIENT_EVIDENCE
                reasons.append("NO_CALIBRATED_ROOT_CAUSE_HYPOTHESIS")
                if mechanism.candidates:
                    unresolved.append("MECHANISM_CALIBRATION")
                if localization.candidates:
                    unresolved.append("ROOT_CAUSE_LOCALIZATION_EVIDENCE")

        decision_id = stable_id(
            "GD",
            {
                "episode_id": episode_id,
                "disposition": disposition.value,
                "accepted_hypothesis_id": accepted_hypothesis_id,
                "anomaly_result_id": anomaly.anomaly_result_id,
            },
        )
        return GenericDecisionRecord(
            decision_id=decision_id,
            episode_id=episode_id,
            disposition=disposition,
            accepted_hypothesis_id=accepted_hypothesis_id,
            reason_codes=reasons,
            unresolved_requirements=sorted(set(unresolved)),
        )


__all__ = ["GenericDecisionGovernor"]
