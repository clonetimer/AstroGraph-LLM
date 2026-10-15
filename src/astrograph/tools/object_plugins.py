from __future__ import annotations

from typing import Any

from astrograph.object_plugins import (
    ObjectEvidenceBuildRequest,
    ObjectPluginRegistry,
    build_builtin_object_plugin_registry,
)
from astrograph.orchestration.models import PermissionScope, SideEffectLevel
from astrograph.tools.registry import ToolDefinition, ToolRegistry, ToolSpec


def build_object_plugin_tool_registry(
    *,
    object_plugins: ObjectPluginRegistry | None = None,
) -> ToolRegistry:
    """Generic tools for approved object plugins.

    This registry is separate from the legacy V0.5.x allow-list so historical
    tool inventories remain stable during migration.
    """

    plugins = object_plugins or build_builtin_object_plugin_registry()
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            spec=ToolSpec(
                tool_id="object_plugin.list",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                tool_version="0.5.13",
                deterministic=True,
                side_effect_level=SideEffectLevel.READ_ONLY,
                required_permissions=[PermissionScope.READ_KNOWLEDGE],
            ),
            handler=lambda payload: {
                "plugins": [item.model_dump(mode="json") for item in plugins.list_manifests()]
            },
        )
    )
    registry.register(
        ToolDefinition(
            spec=ToolSpec(
                tool_id="object_plugin.evidence.build",
                input_schema={
                    "type": "object",
                    "required": [
                        "plugin_id",
                        "delivery_path",
                        "run_id",
                        "protocol_root",
                        "evidence_catalog_path",
                    ],
                },
                output_schema={"type": "object"},
                tool_version="0.5.13",
                deterministic=True,
                side_effect_level=SideEffectLevel.READ_ONLY,
                required_permissions=[PermissionScope.READ_DATA, PermissionScope.READ_KNOWLEDGE],
                timeout_s=120.0,
            ),
            handler=_ObjectEvidenceBuildTool(plugins),
        )
    )
    return registry


class _ObjectEvidenceBuildTool:
    def __init__(self, plugins: ObjectPluginRegistry) -> None:
        self.plugins = plugins

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        plugin = self.plugins.get(str(payload["plugin_id"]))
        result = plugin.build_evidence(
            ObjectEvidenceBuildRequest(
                delivery_path=str(payload["delivery_path"]),
                run_id=str(payload["run_id"]),
                protocol_root=str(payload["protocol_root"]),
                evidence_catalog_path=str(payload["evidence_catalog_path"]),
                parameters=dict(payload.get("parameters", {})),
            )
        )
        return result.model_dump(mode="json")
