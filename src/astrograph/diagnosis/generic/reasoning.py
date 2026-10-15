from __future__ import annotations

from .contracts import (
    DiagnosticHypothesis,
    HeadCalibrationStatus,
    HeadExecutionStatus,
    HeadReceipt,
    HypothesisStatus,
    LocalizationDecision,
    LocalizationHeadResult,
    MechanismDecision,
    MechanismHeadResult,
    ReasoningDecision,
    ReasoningHeadResult,
    stable_id,
)


class ReasoningHead:
    """Assemble auditable hypotheses; it does not use an LLM as formal evidence."""

    head_id = "astrograph.generic.reasoning.evidence-composition"
    head_version = "1.0.0"

    def evaluate(
        self,
        *,
        episode_id: str,
        mechanism: MechanismHeadResult,
        localization: LocalizationHeadResult,
    ) -> ReasoningHeadResult:
        receipt = HeadReceipt(
            head_id=self.head_id,
            head_version=self.head_version,
            calibration_status=HeadCalibrationStatus.DETERMINISTIC_CONTRACT,
            calibration_id="generic-reasoning-contract",
            calibration_version="1.0.0",
            input_sha256=mechanism.receipt.input_sha256,
        )
        result_id = stable_id(
            "RH",
            {
                "episode_id": episode_id,
                "mechanism_result_id": mechanism.mechanism_result_id,
                "localization_result_id": localization.localization_result_id,
            },
        )
        if (
            mechanism.decision == MechanismDecision.NOT_EVALUATED
            or localization.decision == LocalizationDecision.NOT_EVALUATED
        ):
            return ReasoningHeadResult(
                reasoning_result_id=result_id,
                episode_id=episode_id,
                execution_status=HeadExecutionStatus.NOT_EVALUATED,
                decision=ReasoningDecision.NOT_EVALUATED,
                hypotheses=[],
                receipt=receipt,
            )
        if mechanism.decision == MechanismDecision.NO_ANOMALY:
            return ReasoningHeadResult(
                reasoning_result_id=result_id,
                episode_id=episode_id,
                execution_status=HeadExecutionStatus.COMPLETED,
                decision=ReasoningDecision.NO_ANOMALY,
                hypotheses=[],
                receipt=receipt,
            )

        hypotheses: list[DiagnosticHypothesis] = []
        for mechanism_candidate in mechanism.candidates:
            for localization_candidate in localization.candidates:
                missing: list[str] = []
                if not mechanism_candidate.calibrated:
                    missing.append("MECHANISM_CALIBRATION")
                if not localization_candidate.root_cause_supported:
                    missing.append("ROOT_CAUSE_LOCALIZATION_EVIDENCE")
                status = (
                    HypothesisStatus.SUPPORTED
                    if not missing
                    else HypothesisStatus.INSUFFICIENT_EVIDENCE
                )
                support = sorted(
                    set(
                        mechanism_candidate.evidence_ids
                        + localization_candidate.evidence_ids
                    )
                )
                hypothesis_id = stable_id(
                    "DH",
                    {
                        "episode_id": episode_id,
                        "mechanism": mechanism_candidate.mechanism_id.value,
                        "path": localization_candidate.asset_path,
                    },
                )
                hypotheses.append(
                    DiagnosticHypothesis(
                        hypothesis_id=hypothesis_id,
                        mechanism_id=mechanism_candidate.mechanism_id,
                        localization_path=localization_candidate.asset_path,
                        status=status,
                        supporting_evidence_ids=support,
                        contradicting_evidence_ids=[],
                        missing_requirements=missing,
                        falsification_checks=[
                            "Acquire an independent observation that separates component effect from upstream cause.",
                            "Repeat the declared excitation and test whether the balance deviation persists.",
                        ],
                    )
                )
        supported = [
            item for item in hypotheses if item.status == HypothesisStatus.SUPPORTED
        ]
        decision = (
            ReasoningDecision.HYPOTHESES
            if supported
            else ReasoningDecision.INSUFFICIENT_EVIDENCE
        )
        return ReasoningHeadResult(
            reasoning_result_id=result_id,
            episode_id=episode_id,
            execution_status=HeadExecutionStatus.COMPLETED,
            decision=decision,
            hypotheses=hypotheses,
            receipt=receipt,
        )


__all__ = ["ReasoningHead"]
