from __future__ import annotations

from pathlib import Path

from astrograph.data.gate import validate_dataset
from astrograph.orchestration.migration import legacy_protocol_warnings
from astrograph.orchestration.models import ExecutionResult, PermissionScope, TaskEnvelope, WorkflowContext


class DatasetIngestionWorkflow:
    workflow_id = "dataset.ingestion.workflow.v0.4.4"
    workflow_version = "0.4.4"

    def invoke(self, envelope: TaskEnvelope, context: WorkflowContext) -> ExecutionResult:
        if PermissionScope.READ_DATA not in set(context.permissions):
            return ExecutionResult.failed(
                task_id=envelope.task_id,
                workflow_id=self.workflow_id,
                code="ASTROGRAPH_WORKFLOW_PERMISSION_DENIED",
                message="Dataset ingestion validation requires READ_DATA permission.",
            )
        data_dir = envelope.input_payload.get("data_dir")
        if not data_dir:
            return ExecutionResult.failed(
                task_id=envelope.task_id,
                workflow_id=self.workflow_id,
                code="ASTROGRAPH_DATA_DIR_REQUIRED",
                message="data_dir is required for dataset ingestion validation.",
            )
        report = validate_dataset(Path(data_dir))
        return ExecutionResult.succeeded(
            task_id=envelope.task_id,
            workflow_id=self.workflow_id,
            output_payload={"data_quality_report": report.model_dump(mode="json")},
            tool_versions={"data.gate.validate_dataset": self.workflow_version},
            warnings=legacy_protocol_warnings(envelope.model_dump(mode="json")),
        )
