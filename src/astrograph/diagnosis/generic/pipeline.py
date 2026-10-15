from __future__ import annotations

from astrograph.state_estimation import StateEstimateResult

from .anomaly import AnomalyHead
from .contracts import (
    AssetTopologyContext,
    DataQualityContext,
    GenericDiagnosisResult,
    RootCauseEvidenceContext,
)
from .governor import GenericDecisionGovernor
from .localization import LocalizationHead
from .mechanism import MechanismHead
from .reasoning import ReasoningHead


class GenericDiagnosisPipeline:
    """Deterministic four-head sequence with a fail-closed generic governor."""

    def __init__(
        self,
        *,
        anomaly_head: AnomalyHead | None = None,
        mechanism_head: MechanismHead | None = None,
        localization_head: LocalizationHead | None = None,
        reasoning_head: ReasoningHead | None = None,
        governor: GenericDecisionGovernor | None = None,
    ) -> None:
        self.anomaly_head = anomaly_head or AnomalyHead()
        self.mechanism_head = mechanism_head or MechanismHead()
        self.localization_head = localization_head or LocalizationHead()
        self.reasoning_head = reasoning_head or ReasoningHead()
        self.governor = governor or GenericDecisionGovernor()

    def diagnose(
        self,
        *,
        state_estimate: StateEstimateResult,
        topology: AssetTopologyContext,
        data_quality: DataQualityContext,
        root_cause_evidence: RootCauseEvidenceContext | None = None,
    ) -> GenericDiagnosisResult:
        if topology.component_archetype_id != state_estimate.component_archetype_id:
            raise ValueError(
                "asset topology component archetype does not match the state estimate"
            )
        anomaly = self.anomaly_head.evaluate(
            episode_id=topology.episode_id,
            state_estimate=state_estimate,
            data_quality=data_quality,
        )
        mechanism = self.mechanism_head.evaluate(
            episode_id=topology.episode_id,
            state_estimate=state_estimate,
            anomaly=anomaly,
        )
        localization = self.localization_head.evaluate(
            topology=topology,
            anomaly=anomaly,
            mechanism=mechanism,
            root_cause_evidence=root_cause_evidence,
        )
        reasoning = self.reasoning_head.evaluate(
            episode_id=topology.episode_id,
            mechanism=mechanism,
            localization=localization,
        )
        decision = self.governor.decide(
            episode_id=topology.episode_id,
            data_quality=data_quality,
            anomaly=anomaly,
            mechanism=mechanism,
            localization=localization,
            reasoning=reasoning,
        )
        return GenericDiagnosisResult(
            episode_id=topology.episode_id,
            anomaly=anomaly,
            mechanism=mechanism,
            localization=localization,
            reasoning=reasoning,
            decision=decision,
        )


__all__ = ["GenericDiagnosisPipeline"]
