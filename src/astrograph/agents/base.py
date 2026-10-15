from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.orchestration.models import WorkflowContext
from astrograph.tools import ToolRegistry


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-agent-result.v1"
    agent_id: str
    status: str
    output: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class ControlledAgent:
    """Base class for agents restricted to a ToolRegistry allow-list."""

    agent_id = "controlled-agent"
    agent_version = "0.5.0"

    def __init__(self, tool_registry: ToolRegistry | None = None) -> None:
        self.tool_registry = tool_registry

    def invoke_tool(self, tool_id: str, payload: dict[str, Any], context: WorkflowContext) -> dict[str, Any]:
        if self.tool_registry is None:
            raise PermissionError("agent has no ToolRegistry")
        return self.tool_registry.invoke(tool_id, payload, context)
