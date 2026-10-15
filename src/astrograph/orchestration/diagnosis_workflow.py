from __future__ import annotations

from astrograph.audit.store import AuditStore
from astrograph.config import DiagnosisPolicy
from astrograph.diagnosis.workflow import invoke_workflow
from astrograph.evidence import EvidenceConstraintCatalog
from astrograph.knowledge.repository import KnowledgeRepository
from astrograph.orchestration.migration import legacy_protocol_warnings
from astrograph.orchestration.models import ExecutionResult, PermissionScope, TaskEnvelope, WorkflowContext
from astrograph.schemas import WorkflowInput


class DiagnosisWorkflow:
    workflow_id = "diagnosis.workflow.v0.4.4"
    workflow_version = "0.4.4"

    def __init__(
        self,
        repository: KnowledgeRepository,
        *,
        policy: DiagnosisPolicy | None = None,
        audit_store: AuditStore | None = None,
        constraints: EvidenceConstraintCatalog | None = None,
    ) -> None:
        self.repository = repository
        self.policy = policy
        self.audit_store = audit_store
        self.constraints = constraints

    def invoke(self, envelope: TaskEnvelope, context: WorkflowContext) -> ExecutionResult:
        required = {PermissionScope.RUN_DIAGNOSIS, PermissionScope.WRITE_AUDIT}
        missing = sorted(required - set(context.permissions))
        if missing:
            return ExecutionResult.failed(
                task_id=envelope.task_id,
                workflow_id=self.workflow_id,
                code="ASTROGRAPH_WORKFLOW_PERMISSION_DENIED",
                message="Diagnosis workflow requires RUN_DIAGNOSIS and WRITE_AUDIT permissions.",
                details={"missing_permissions": [item.value for item in missing]},
            )
        try:
            payload = envelope.input_payload.get("workflow_input", envelope.input_payload)
            request = WorkflowInput.model_validate(payload)
            state = invoke_workflow(
                self.repository,
                request,
                policy=self.policy,
                audit_store=self.audit_store,
                constraints=self.constraints,
            )
            return ExecutionResult.succeeded(
                task_id=envelope.task_id,
                workflow_id=self.workflow_id,
                output_payload={"workflow_state": dict(state)},
                tool_versions={"diagnosis.workflow": self.workflow_version},
                warnings=legacy_protocol_warnings(envelope.model_dump(mode="json")),
            )
        except Exception as exc:
            return ExecutionResult.failed(
                task_id=envelope.task_id,
                workflow_id=self.workflow_id,
                code="ASTROGRAPH_DIAGNOSIS_WORKFLOW_FAILED",
                message=f"Diagnosis workflow failed: {type(exc).__name__}: {exc}",
            )
