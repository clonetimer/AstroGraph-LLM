from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from astrograph.orchestration.models import ExecutionResult, TaskEnvelope, WorkflowContext

WorkflowHandler = Callable[[TaskEnvelope, WorkflowContext], ExecutionResult]


@dataclass(frozen=True)
class WorkflowDefinition:
    workflow_id: str
    workflow_version: str
    description: str
    handler: WorkflowHandler
    deterministic: bool = True


class WorkflowRegistry:
    """Registry for deterministic workflows. It is not an LLM router."""

    def __init__(self) -> None:
        self._definitions: dict[str, WorkflowDefinition] = {}

    def register(self, definition: WorkflowDefinition) -> None:
        if definition.workflow_id in self._definitions:
            raise ValueError(f"workflow already registered: {definition.workflow_id}")
        self._definitions[definition.workflow_id] = definition

    def get(self, workflow_id: str) -> WorkflowDefinition:
        try:
            return self._definitions[workflow_id]
        except KeyError as exc:
            raise KeyError(f"unknown workflow: {workflow_id}") from exc

    def list_definitions(self) -> list[WorkflowDefinition]:
        return [self._definitions[key] for key in sorted(self._definitions)]

    def invoke(self, workflow_id: str, envelope: TaskEnvelope, context: WorkflowContext) -> ExecutionResult:
        definition = self.get(workflow_id)
        if envelope.execution_policy.deterministic_only and not definition.deterministic:
            return ExecutionResult.failed(
                task_id=envelope.task_id,
                workflow_id=workflow_id,
                code="ASTROGRAPH_NON_DETERMINISTIC_WORKFLOW_BLOCKED",
                message="Non-deterministic workflows are blocked by the execution policy.",
            )
        return definition.handler(envelope, context)
