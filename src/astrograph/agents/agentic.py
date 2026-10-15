from __future__ import annotations

from collections import defaultdict
from typing import Any

from astrograph.agents.base import ControlledAgent
from astrograph.evidence import (
    EpisodeEvidenceBundle,
    EvidenceDefinition,
)
from astrograph.investigation import (
    CritiqueFinding,
    CritiqueResult,
    CritiqueSeverity,
    DiagnosticHypothesis,
    GraphQueryPlan,
    HypothesisSet,
    HypothesisStatus,
    HypothesisType,
    InvestigationBudget,
    InvestigationPlan,
    InvestigationStep,
    VerificationPlan,
    VerificationTask,
)
from astrograph.services import (
    DeterministicReferenceReasoner,
    EvidenceNavigationPlanningService,
    InvestigationPlanningService,
    SafetyValidationService,
    StructuredHypothesisProposalService,
    StructuredReasoningBackend,
    VerificationPlanningService,
)
from astrograph.retrieval import PreReasoningRetrievalPackage


class InvestigationOrchestratorAgent(ControlledAgent):
    """Compatibility role wrapper around InvestigationPlanningService.

    This class does not call an LLM in V0.5.12.  It is retained so historical
    workflows and traces remain readable while deterministic responsibilities
    are moved out of the Agent layer.
    """

    agent_id = "investigation-orchestrator-agent"
    agent_version = "0.5.12-compat"

    def __init__(self, *args: Any, backend: StructuredReasoningBackend | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.service = InvestigationPlanningService()

    def plan(self, *, task_id: str, run_id: str, candidate_fault_ids: list[str]) -> InvestigationPlan:
        return self.service.plan(
            task_id=task_id,
            run_id=run_id,
            candidate_fault_ids=candidate_fault_ids,
            object_label="D3反作用轮",
            evidence_tool_id="d3_rw.evidence.build",
            core_tool_id="diagnosis.invoke_core",
        )


class EvidenceNavigatorAgent(ControlledAgent):
    """Compatibility role wrapper around deterministic graph-plan compilation."""

    agent_id = "evidence-navigator-agent"
    agent_version = "0.5.12-compat"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.service = EvidenceNavigationPlanningService()

    def navigate(
        self,
        *,
        candidate_fault_ids: list[str],
        definitions: list[EvidenceDefinition],
    ) -> GraphQueryPlan:
        return self.service.navigate(
            candidate_fault_ids=candidate_fault_ids,
            definitions=definitions,
            object_label="反作用轮",
        )

    def navigate_episode(
        self,
        bundle: EpisodeEvidenceBundle,
        *,
        candidate_fault_ids: list[str] | None = None,
    ) -> EvidenceNavigationResult:
        """Walk the episode evidence graph in the whole-spacecraft domain.

        Delegates to the deterministic navigation service; used by the online
        workflow to expose evidence paths to the reasoning layer without
        changing hypothesis logic.
        """
        return self.service.navigate_episode(
            bundle,
            candidate_fault_ids=candidate_fault_ids,
        )


class VerificationScientistAgent(ControlledAgent):
    """Deprecated name retained for the deterministic verification-plan service."""

    agent_id = "verification-scientist-agent"
    agent_version = "0.5.12-compat"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.service = VerificationPlanningService()

    def build_plan(self, bundle: EpisodeEvidenceBundle) -> VerificationPlan:
        return self.service.build_plan(bundle, generated_by_id=self.agent_id)


class DiagnosticReasonerAgent(ControlledAgent):
    """Compatibility façade over deterministic reference and structured proposal services."""

    agent_id = "diagnostic-reasoner-agent"
    agent_version = "0.5.10-compat"

    def __init__(self, *args: Any, backend: StructuredReasoningBackend | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.reference_service = DeterministicReferenceReasoner()
        self.proposal_service = StructuredHypothesisProposalService(agent_id=self.agent_id)

    @staticmethod
    def _reference_hypotheses(bundle: EpisodeEvidenceBundle) -> HypothesisSet:
        return DeterministicReferenceReasoner.build(bundle)

    @staticmethod
    def diagnostic_protocol(bundle: EpisodeEvidenceBundle) -> dict[str, Any]:
        return DeterministicReferenceReasoner.diagnostic_protocol(bundle)

    @classmethod
    def reference_hypotheses(cls, bundle: EpisodeEvidenceBundle) -> HypothesisSet:
        return DeterministicReferenceReasoner.build(bundle)

    @staticmethod
    def _normalize_no_registered_fault(
        candidate: HypothesisSet,
        reference: HypothesisSet,
        *,
        validation_metadata: dict[str, Any] | None = None,
    ) -> HypothesisSet:
        return DeterministicReferenceReasoner.normalize_no_registered_fault(
            candidate, reference, validation_metadata=validation_metadata
        )

    def reason(
        self,
        bundle: EpisodeEvidenceBundle,
        *,
        candidate_fusion: dict[str, Any] | None = None,
        retrieval_context: PreReasoningRetrievalPackage | dict[str, Any] | None = None,
        deliberation_context: dict[str, Any] | None = None,
        capture_shadow_candidate_revision: bool = False,
    ) -> HypothesisSet:
        return self.proposal_service.reason(
            bundle,
            backend=self.backend,
            candidate_fusion=candidate_fusion,
            retrieval_context=retrieval_context,
            deliberation_context=deliberation_context,
            capture_shadow_candidate_revision=capture_shadow_candidate_revision,
        )


class SafetyCriticAgent(ControlledAgent):
    """Compatibility façade over split evidence-integrity and hypothesis-safety services."""

    agent_id = "safety-critic-agent"
    agent_version = "0.5.2-compat"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.service = SafetyValidationService()

    def critique(self, hypothesis_set: HypothesisSet, bundle: EpisodeEvidenceBundle) -> CritiqueResult:
        return self.service.critique(
            hypothesis_set, bundle, generated_by_agent_id=self.agent_id
        )
