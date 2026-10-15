from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from astrograph.orchestration.models import PermissionScope, SideEffectLevel, WorkflowContext


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_id: str = Field(min_length=1)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    tool_version: str = "0.5.0"
    deterministic: bool = True
    side_effect_level: SideEffectLevel = SideEffectLevel.NONE
    required_permissions: list[PermissionScope] = Field(default_factory=list)
    timeout_s: float = Field(default=30.0, gt=0.0)
    retry_count: int = Field(default=0, ge=0)
    execution_mode: Literal["TRUSTED_IN_PROCESS", "ISOLATED_PROCESS"] = "TRUSTED_IN_PROCESS"
    max_payload_bytes: int = Field(default=2_000_000, gt=0)
    max_output_bytes: int = Field(default=2_000_000, gt=0)


@dataclass(frozen=True)
class WorkerTarget:
    module_file: str
    function_name: str
    artifact_sha256: str


@dataclass(frozen=True)
class ToolDefinition:
    spec: ToolSpec
    handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    worker_target: WorkerTarget | None = None


class ToolRegistry:
    """White-list registry for all tools callable by controlled agents/workflows."""

    def __init__(self, *, executor=None) -> None:
        from .executor import ToolExecutor
        self._definitions: dict[str, ToolDefinition] = {}
        self.executor = executor or ToolExecutor()
        self._sealed = False

    def seal(self) -> None:
        """Prevent hot mutation of the task-bound registry snapshot."""
        self._sealed = True

    def register(self, definition: ToolDefinition) -> None:
        if self._sealed:
            raise RuntimeError("registry snapshot is sealed")
        from .executor import validate_tool_definition
        validate_tool_definition(definition)
        tool_id = definition.spec.tool_id
        if tool_id in self._definitions:
            raise ValueError(f"tool already registered: {tool_id}")
        self._definitions[tool_id] = ToolDefinition(
            definition.spec.model_copy(deep=True), definition.handler, definition.worker_target)

    def get(self, tool_id: str) -> ToolDefinition:
        try:
            value = self._definitions[tool_id]
            return ToolDefinition(value.spec.model_copy(deep=True), value.handler, value.worker_target)
        except KeyError as exc:
            raise KeyError(f"unknown tool: {tool_id}") from exc

    def list_specs(self) -> list[ToolSpec]:
        return [self._definitions[key].spec.model_copy(deep=True) for key in sorted(self._definitions)]

    def require_permissions(self, tool_id: str, context: WorkflowContext) -> None:
        spec = self.get(tool_id).spec
        if not context.has_permissions(spec.required_permissions):
            missing = sorted(set(spec.required_permissions) - set(context.permissions))
            raise PermissionError(f"tool {tool_id} missing permissions: {', '.join(item.value for item in missing)}")

    def invoke(self, tool_id: str, payload: dict[str, Any], context: WorkflowContext) -> dict[str, Any]:
        definition = self.get(tool_id)
        return self.executor.invoke(definition, payload, context)
