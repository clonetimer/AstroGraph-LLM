from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from astrograph.agents.object_plugin_factory import ObjectPluginDesignAgent, ObjectPluginDesignBackend
from astrograph.orchestration.models import (
    ExecutionResult,
    PermissionScope,
    TaskEnvelope,
    WorkflowContext,
)
from astrograph.plugin_factory import ObjectPluginFactoryService


class ObjectPluginFactoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_root: str = Field(min_length=1)
    object_context: dict = Field(default_factory=dict)
    draft_build_spec: dict | None = None


class ObjectPluginFactoryWorkflow:
    """Design-time workflow: Agent spec -> deterministic scaffold -> validation."""

    workflow_id = "object.plugin.factory.workflow.v0.5.13"
    workflow_version = "0.5.13"

    def __init__(self, *, design_backend: ObjectPluginDesignBackend | None = None) -> None:
        self.design_backend = design_backend
        self.design_agent = ObjectPluginDesignAgent(backend=design_backend)
        self.factory = ObjectPluginFactoryService()

    def invoke(self, envelope: TaskEnvelope, context: WorkflowContext) -> ExecutionResult:
        required = {PermissionScope.READ_KNOWLEDGE, PermissionScope.WRITE_ARTIFACTS}
        missing = sorted(required - set(context.permissions), key=lambda item: item.value)
        if missing:
            return ExecutionResult.failed(
                task_id=envelope.task_id,
                workflow_id=self.workflow_id,
                code="ASTROGRAPH_OBJECT_PLUGIN_FACTORY_PERMISSION_DENIED",
                message="Object plugin generation requires read-knowledge and write-artifacts permissions.",
                details={"missing_permissions": [item.value for item in missing]},
            )
        if self.design_backend is not None:
            if envelope.execution_policy.deterministic_only:
                return ExecutionResult.failed(
                    task_id=envelope.task_id,
                    workflow_id=self.workflow_id,
                    code="ASTROGRAPH_OBJECT_PLUGIN_DESIGN_LLM_BLOCKED",
                    message="LLM-backed plugin design requires deterministic_only=false.",
                )
            if not envelope.execution_policy.allow_llm:
                return ExecutionResult.failed(
                    task_id=envelope.task_id,
                    workflow_id=self.workflow_id,
                    code="ASTROGRAPH_OBJECT_PLUGIN_DESIGN_LLM_NOT_ALLOWED",
                    message="LLM-backed plugin design requires allow_llm=true.",
                )
        try:
            request = ObjectPluginFactoryInput.model_validate(envelope.input_payload)
            spec = self.design_agent.design(
                object_context=request.object_context,
                draft_build_spec=request.draft_build_spec,
            )
            result = self.factory.generate(spec, request.output_root)
            plan = result["plan"]
            warnings = [
                "Generated object plugins are design-time artifacts and are not automatically registered online."
            ]
            if bool(plan["requires_human_review"]):
                warnings.append(
                    "Custom-code or blocked features require formula, unit, source, code, and test review."
                )
            return ExecutionResult.succeeded(
                task_id=envelope.task_id,
                workflow_id=self.workflow_id,
                output_payload={
                    "build_spec": spec.model_dump(mode="json"),
                    **result,
                    "online_registration_allowed": False,
                },
                warnings=warnings,
                metadata={
                    "design_backend": (
                        "deterministic-draft"
                        if self.design_backend is None
                        else f"{self.design_backend.model_id}:{self.design_backend.model_version}"
                    ),
                    "generated_code_executed": False,
                },
            )
        except Exception as exc:
            return ExecutionResult.failed(
                task_id=envelope.task_id,
                workflow_id=self.workflow_id,
                code="ASTROGRAPH_OBJECT_PLUGIN_FACTORY_FAILED",
                message=f"Object plugin factory failed: {type(exc).__name__}: {exc}",
            )
