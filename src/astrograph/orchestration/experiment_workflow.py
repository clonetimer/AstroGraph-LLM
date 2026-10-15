from __future__ import annotations

from astrograph.orchestration.models import ExecutionResult, PermissionScope, TaskEnvelope, WorkflowContext
from astrograph.service.core import AstroGraphService


class ExperimentWorkflow:
    workflow_id = "experiment.workflow.v0.4.4"
    workflow_version = "0.4.4"

    def __init__(self, service: AstroGraphService) -> None:
        self.service = service

    def invoke(self, envelope: TaskEnvelope, context: WorkflowContext) -> ExecutionResult:
        if PermissionScope.RUN_EXPERIMENT not in set(context.permissions):
            return ExecutionResult.failed(
                task_id=envelope.task_id,
                workflow_id=self.workflow_id,
                code="ASTROGRAPH_WORKFLOW_PERMISSION_DENIED",
                message="Experiment workflow requires RUN_EXPERIMENT permission.",
            )
        return ExecutionResult.succeeded(
            task_id=envelope.task_id,
            workflow_id=self.workflow_id,
            output_payload={"experiments": self.service.experiments()},
            tool_versions={"experiment.registry": self.workflow_version},
            warnings=["V0.4.4 lists existing experiment records only; it does not infer new model performance claims."],
        )
