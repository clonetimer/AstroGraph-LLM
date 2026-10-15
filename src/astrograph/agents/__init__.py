from __future__ import annotations

from .base import AgentResult, ControlledAgent
from .knowledge_curation import (
    ControlledKnowledgeCurationWorkflow,
    KnowledgeCandidateCriticAgent,
    KnowledgeCurationAgent,
    KnowledgeSourceProfilerAgent,
    VllmKnowledgeExtractionBackend,
)
from .operator_assistant import OperatorAssistantAgent
from .diagnosis_planner import (
    DiagnosisExecutionPlan,
    DiagnosisPlannerAgent,
    DiagnosisPlanningContext,
    DiagnosisPlanStep,
    DiagnosisStage,
)

__all__ = [
    "AgentResult",
    "ControlledAgent",
    "KnowledgeCurationAgent",
    "ControlledKnowledgeCurationWorkflow",
    "KnowledgeCandidateCriticAgent",
    "KnowledgeSourceProfilerAgent",
    "VllmKnowledgeExtractionBackend",
    "OperatorAssistantAgent",
    "DiagnosisExecutionPlan",
    "DiagnosisPlannerAgent",
    "DiagnosisPlanningContext",
    "DiagnosisPlanStep",
    "DiagnosisStage",
    "ReportAgent",
    "ReportPayload",
    "DiagnosticReasonerAgent",
    "EvidenceNavigatorAgent",
    "InvestigationOrchestratorAgent",
    "SafetyCriticAgent",
    "StructuredReasoningBackend",
    "VerificationScientistAgent",
]


def __getattr__(name: str):
    if name in {"ReportAgent", "ReportPayload"}:
        from .report_agent import ReportAgent, ReportPayload

        return {"ReportAgent": ReportAgent, "ReportPayload": ReportPayload}[name]
    if name in {
        "DiagnosticReasonerAgent",
        "EvidenceNavigatorAgent",
        "InvestigationOrchestratorAgent",
        "SafetyCriticAgent",
        "StructuredReasoningBackend",
        "VerificationScientistAgent",
    }:
        from .agentic import (
            DiagnosticReasonerAgent,
            EvidenceNavigatorAgent,
            InvestigationOrchestratorAgent,
            SafetyCriticAgent,
            StructuredReasoningBackend,
            VerificationScientistAgent,
        )

        return {
            "DiagnosticReasonerAgent": DiagnosticReasonerAgent,
            "EvidenceNavigatorAgent": EvidenceNavigatorAgent,
            "InvestigationOrchestratorAgent": InvestigationOrchestratorAgent,
            "SafetyCriticAgent": SafetyCriticAgent,
            "StructuredReasoningBackend": StructuredReasoningBackend,
            "VerificationScientistAgent": VerificationScientistAgent,
        }[name]
    raise AttributeError(name)

from .object_plugin_factory import ObjectPluginDesignAgent, ObjectPluginDesignBackend
from .dataset_onboarding import (
    ControlledDatasetOnboardingWorkflow,
    DatasetProfilerAgent,
    MappingContractCriticAgent,
    SchemaMappingProposalAgent,
)

__all__ += [
    "ControlledDatasetOnboardingWorkflow",
    "DatasetProfilerAgent",
    "MappingContractCriticAgent",
    "ObjectPluginDesignAgent",
    "ObjectPluginDesignBackend",
    "SchemaMappingProposalAgent",
]
