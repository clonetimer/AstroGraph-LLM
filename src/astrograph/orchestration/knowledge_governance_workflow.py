from __future__ import annotations

from astrograph.orchestration.models import ExecutionResult, PermissionScope, TaskEnvelope, WorkflowContext
from astrograph.service.core import AstroGraphService


class KnowledgeGovernanceWorkflow:
    workflow_id = "knowledge.governance.workflow.v0.4.4"
    workflow_version = "0.4.4"

    def __init__(self, service: AstroGraphService) -> None:
        self.service = service

    def invoke(self, envelope: TaskEnvelope, context: WorkflowContext) -> ExecutionResult:
        if PermissionScope.READ_KNOWLEDGE not in set(context.permissions):
            return ExecutionResult.failed(
                task_id=envelope.task_id,
                workflow_id=self.workflow_id,
                code="ASTROGRAPH_WORKFLOW_PERMISSION_DENIED",
                message="Knowledge governance read workflow requires READ_KNOWLEDGE permission.",
            )
        return ExecutionResult.succeeded(
            task_id=envelope.task_id,
            workflow_id=self.workflow_id,
            output_payload={"changes": self.service.knowledge_changes()},
            tool_versions={"knowledge.governance": self.workflow_version},
            warnings=[
                "V0.4.4 exposes governance state only; it does not autonomously publish knowledge or add fault capability."
            ],
        )
