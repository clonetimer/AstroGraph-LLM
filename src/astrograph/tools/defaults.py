from __future__ import annotations

from typing import Any

from astrograph.orchestration.models import PermissionScope, SideEffectLevel
from astrograph.schemas import WorkflowInput
from astrograph.service.core import AstroGraphService
from astrograph.tools.registry import ToolDefinition, ToolRegistry, ToolSpec


def build_default_tool_registry(service: AstroGraphService) -> ToolRegistry:
    """Register existing deterministic capabilities as controlled tools.

    The registry only exposes capabilities that already exist in V0.4.3. It does
    not add new faults, degradations, RUL, probability calibration, or compound
    fault reasoning.
    """

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            spec=ToolSpec(
                tool_id="diagnosis.invoke",
                input_schema=WorkflowInput.model_json_schema(),
                output_schema={"type": "object"},
                tool_version="0.4.4",
                deterministic=True,
                side_effect_level=SideEffectLevel.MUTATE_STATE,
                required_permissions=[PermissionScope.RUN_DIAGNOSIS, PermissionScope.WRITE_AUDIT],
            ),
            handler=lambda payload: service.diagnose(WorkflowInput.model_validate(payload)),
        )
    )

    registry.register(
        ToolDefinition(
            spec=ToolSpec(
                tool_id="episode.evaluate_temporal_constraints",
                input_schema={"type": "object", "required": ["episodes"]},
                output_schema={"type": "object"},
                tool_version="0.4.4",
                deterministic=True,
                side_effect_level=SideEffectLevel.READ_ONLY,
                required_permissions=[PermissionScope.READ_DATA, PermissionScope.RUN_DIAGNOSIS],
            ),
            handler=lambda payload: service.evaluate_episode_temporal(payload.get("episodes", [])),
        )
    )

    registry.register(
        ToolDefinition(
            spec=ToolSpec(
                tool_id="state.persistence_status",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                tool_version="0.4.4",
                deterministic=True,
                side_effect_level=SideEffectLevel.READ_ONLY,
                required_permissions=[PermissionScope.READ_STATE],
            ),
            handler=lambda payload: service.persistence_status(),
        )
    )

    registry.register(
        ToolDefinition(
            spec=ToolSpec(
                tool_id="knowledge.list_changes",
                input_schema={"type": "object"},
                output_schema={"type": "array"},
                tool_version="0.4.4",
                deterministic=True,
                side_effect_level=SideEffectLevel.READ_ONLY,
                required_permissions=[PermissionScope.READ_KNOWLEDGE],
            ),
            handler=lambda payload: {"changes": service.knowledge_changes()},
        )
    )

    registry.register(
        ToolDefinition(
            spec=ToolSpec(
                tool_id="experiment.list",
                input_schema={"type": "object"},
                output_schema={"type": "array"},
                tool_version="0.4.4",
                deterministic=True,
                side_effect_level=SideEffectLevel.READ_ONLY,
                required_permissions=[PermissionScope.RUN_EXPERIMENT],
            ),
            handler=lambda payload: {"experiments": service.experiments()},
        )
    )

    return registry
