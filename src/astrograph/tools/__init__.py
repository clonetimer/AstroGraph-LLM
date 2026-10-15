from .registry import ToolDefinition, ToolRegistry, ToolSpec

__all__ = ["ToolDefinition", "ToolRegistry", "ToolSpec", "AgentToolPlugin", "AgentToolPluginManifest", "TrustedAgentToolPluginManager", "get_trusted_agent_tool_plugin_manager", "build_online_tool_registry", "build_object_plugin_tool_registry"]


def __getattr__(name: str):
    if name in {"AgentToolPlugin", "AgentToolPluginManifest", "TrustedAgentToolPluginManager", "get_trusted_agent_tool_plugin_manager"}:
        from . import plugins as _plugins
        return getattr(_plugins, name)
    if name == "build_online_tool_registry":
        from .online import build_online_tool_registry
        return build_online_tool_registry
    if name == "build_object_plugin_tool_registry":
        from .object_plugins import build_object_plugin_tool_registry
        return build_object_plugin_tool_registry
    raise AttributeError(name)

