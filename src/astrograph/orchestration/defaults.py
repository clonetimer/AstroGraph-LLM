from __future__ import annotations

from astrograph.orchestration.agentic_investigation_workflow import AgenticInvestigationWorkflow
from astrograph.orchestration.dataset_ingestion_workflow import DatasetIngestionWorkflow
from astrograph.orchestration.diagnosis_workflow import DiagnosisWorkflow
from astrograph.orchestration.experiment_workflow import ExperimentWorkflow
from astrograph.orchestration.knowledge_governance_workflow import KnowledgeGovernanceWorkflow
from astrograph.orchestration.recovery_workflow import RecoveryWorkflow
from astrograph.orchestration.registry import WorkflowDefinition, WorkflowRegistry
from astrograph.service.core import AstroGraphService


def build_default_workflow_registry(
    service: AstroGraphService,
    *,
    agentic_workflow: AgenticInvestigationWorkflow | None = None,
) -> WorkflowRegistry:
    """Register V0.4.4 compatibility workflows and the V0.5.0 shadow workflow."""

    resolved_agentic = agentic_workflow or AgenticInvestigationWorkflow()
    workflows = [
        DiagnosisWorkflow(
            service.repository,
            policy=service.policy,
            audit_store=service.audit_store,
            constraints=service.evidence_constraints,
        ),
        DatasetIngestionWorkflow(),
        KnowledgeGovernanceWorkflow(service),
        ExperimentWorkflow(service),
        RecoveryWorkflow(service),
        resolved_agentic,
    ]
    registry = WorkflowRegistry()
    for workflow in workflows:
        is_agentic = workflow.workflow_id == AgenticInvestigationWorkflow.workflow_id
        registry.register(
            WorkflowDefinition(
                workflow_id=workflow.workflow_id,
                workflow_version=workflow.workflow_version,
                description=(
                    "Agentic D3 reaction-wheel investigation in shadow mode."
                    if is_agentic
                    else f"AstroGraph compatibility workflow {workflow.workflow_id}."
                ),
                handler=workflow.invoke,
                deterministic=(not is_agentic or resolved_agentic.reasoning_backend is None),
            )
        )
    return registry
