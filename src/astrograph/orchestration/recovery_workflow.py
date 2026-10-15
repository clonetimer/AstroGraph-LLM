from __future__ import annotations

from pathlib import Path

from astrograph.orchestration.models import ExecutionResult, PermissionScope, TaskEnvelope, WorkflowContext
from astrograph.service.core import AstroGraphService


class RecoveryWorkflow:
    workflow_id = "recovery.workflow.v0.4.4"
    workflow_version = "0.4.4"

    def __init__(self, service: AstroGraphService) -> None:
        self.service = service

    def invoke(self, envelope: TaskEnvelope, context: WorkflowContext) -> ExecutionResult:
        action = str(envelope.input_payload.get("action", "status"))
        permissions = set(context.permissions)
        if action == "status":
            if PermissionScope.READ_STATE not in permissions:
                return ExecutionResult.failed(
                    task_id=envelope.task_id,
                    workflow_id=self.workflow_id,
                    code="ASTROGRAPH_WORKFLOW_PERMISSION_DENIED",
                    message="Recovery status requires READ_STATE permission.",
                )
            output = self.service.persistence_status()
        elif action == "snapshot":
            if not {PermissionScope.READ_STATE, PermissionScope.WRITE_ARTIFACTS}.issubset(permissions):
                return ExecutionResult.failed(
                    task_id=envelope.task_id,
                    workflow_id=self.workflow_id,
                    code="ASTROGRAPH_WORKFLOW_PERMISSION_DENIED",
                    message="Recovery snapshot requires READ_STATE and WRITE_ARTIFACTS permissions.",
                )
            output_dir = envelope.input_payload.get("output_dir")
            if not output_dir:
                return ExecutionResult.failed(
                    task_id=envelope.task_id,
                    workflow_id=self.workflow_id,
                    code="ASTROGRAPH_OUTPUT_DIR_REQUIRED",
                    message="output_dir is required for snapshot action.",
                )
            output = self.service.snapshot_state(Path(output_dir))
        elif action == "recover":
            if PermissionScope.WRITE_STATE not in permissions:
                return ExecutionResult.failed(
                    task_id=envelope.task_id,
                    workflow_id=self.workflow_id,
                    code="ASTROGRAPH_WORKFLOW_PERMISSION_DENIED",
                    message="Recovery action requires WRITE_STATE permission.",
                )
            output = self.service.recover_state()
        else:
            return ExecutionResult.failed(
                task_id=envelope.task_id,
                workflow_id=self.workflow_id,
                code="ASTROGRAPH_UNSUPPORTED_RECOVERY_ACTION",
                message=f"Unsupported recovery action: {action}",
            )
        return ExecutionResult.succeeded(
            task_id=envelope.task_id,
            workflow_id=self.workflow_id,
            output_payload={"recovery": output},
            tool_versions={"persistence.recovery": self.workflow_version},
        )
