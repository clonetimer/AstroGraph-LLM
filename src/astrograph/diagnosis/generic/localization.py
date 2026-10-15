from __future__ import annotations

from .contracts import (
    AnomalyHeadResult,
    AssetTopologyContext,
    HeadCalibrationStatus,
    HeadEvidence,
    HeadExecutionStatus,
    HeadReceipt,
    LocalizationCandidate,
    LocalizationDecision,
    LocalizationHeadResult,
    LocalizationRole,
    MechanismDecision,
    MechanismHeadResult,
    RootCauseEvidenceContext,
    stable_id,
)


class LocalizationHead:
    """Localize observed model mismatch, without relabeling it as a root cause."""

    head_id = "astrograph.generic.localization.topology-path"
    head_version = "1.1.0"

    def evaluate(
        self,
        *,
        topology: AssetTopologyContext,
        anomaly: AnomalyHeadResult,
        mechanism: MechanismHeadResult,
        root_cause_evidence: RootCauseEvidenceContext | None = None,
    ) -> LocalizationHeadResult:
        receipt = HeadReceipt(
            head_id=self.head_id,
            head_version=self.head_version,
            calibration_status=HeadCalibrationStatus.DETERMINISTIC_CONTRACT,
            calibration_id="asset-topology-contract",
            calibration_version="1.0.0",
            input_sha256=anomaly.receipt.input_sha256,
        )
        result_id = stable_id(
            "LH",
            {
                "episode_id": topology.episode_id,
                "component_instance_id": topology.component_instance_id,
                "mechanism_result_id": mechanism.mechanism_result_id,
            },
        )
        if mechanism.decision == MechanismDecision.NOT_EVALUATED:
            return LocalizationHeadResult(
                localization_result_id=result_id,
                episode_id=topology.episode_id,
                execution_status=HeadExecutionStatus.NOT_EVALUATED,
                decision=LocalizationDecision.NOT_EVALUATED,
                candidates=[],
                evidence=[],
                receipt=receipt,
            )
        if mechanism.decision == MechanismDecision.NO_ANOMALY:
            return LocalizationHeadResult(
                localization_result_id=result_id,
                episode_id=topology.episode_id,
                execution_status=HeadExecutionStatus.COMPLETED,
                decision=LocalizationDecision.NO_ANOMALY,
                candidates=[],
                evidence=[],
                receipt=receipt,
            )
        if (
            root_cause_evidence is not None
            and root_cause_evidence.input_sha256 != anomaly.receipt.input_sha256
        ):
            raise ValueError("root-cause evidence does not match the diagnosis input")
        mechanism_is_calibrated = bool(mechanism.candidates) and all(
            item.calibrated for item in mechanism.candidates
        )
        root_cause_supported = bool(
            mechanism_is_calibrated
            and root_cause_evidence is not None
            and root_cause_evidence.supports_root_cause
        )
        evidence_id = stable_id(
            "EV",
            {
                "episode_id": topology.episode_id,
                "type": (
                    "boundary_causal_isolation"
                    if root_cause_supported
                    else "affected_component_binding"
                ),
                "component": topology.component_instance_id,
                "archetype": topology.component_archetype_id,
            },
        )
        evidence = HeadEvidence(
            evidence_id=evidence_id,
            evidence_type=(
                "BOUNDARY_CAUSAL_ISOLATION"
                if root_cause_supported
                else "AFFECTED_COMPONENT_BINDING"
            ),
            source_reference=(
                "root_cause_evidence_context"
                if root_cause_supported
                else "asset_topology_context.asset_path"
            ),
            summary=(
                "Observed component-boundary inputs, channel-integrity checks, time alignment and "
                "independent repetitions isolate the calibrated mechanism to this component."
                if root_cause_supported
                else "The state estimator and significant balance deviation are bound to the declared "
                "component instance; this identifies an affected component, not a root cause."
            ),
        )
        candidate = LocalizationCandidate(
            asset_path=topology.asset_path,
            component_instance_id=topology.component_instance_id,
            role=(
                LocalizationRole.ROOT_CAUSE
                if root_cause_supported
                else LocalizationRole.AFFECTED_COMPONENT
            ),
            score=anomaly.anomaly_score,
            root_cause_supported=root_cause_supported,
            evidence_ids=sorted(
                {
                    evidence_id,
                    *(
                        root_cause_evidence.evidence_ids
                        if root_cause_supported and root_cause_evidence is not None
                        else []
                    ),
                }
            ),
        )
        return LocalizationHeadResult(
            localization_result_id=result_id,
            episode_id=topology.episode_id,
            execution_status=HeadExecutionStatus.COMPLETED,
            decision=(
                LocalizationDecision.ROOT_CAUSE_IDENTIFIED
                if root_cause_supported
                else LocalizationDecision.AFFECTED_COMPONENT_IDENTIFIED
            ),
            candidates=[candidate],
            evidence=[evidence],
            receipt=receipt,
        )


__all__ = ["LocalizationHead"]
