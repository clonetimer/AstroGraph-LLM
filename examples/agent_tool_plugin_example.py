"""Example trusted Agent tool plugin.

Copy this file into the administrator-owned ASTROGRAPH_AGENT_TOOL_PLUGIN_DIR only after
review.  It is DEVELOPMENT by default and therefore will not be registered into tasks.
"""
from astrograph.orchestration.models import PermissionScope, SideEffectLevel
from astrograph.tools.plugins import AgentToolExport, AgentToolPlugin, AgentToolPluginManifest
from astrograph.tools.registry import ToolSpec


def summarize_channels(payload):
    values = [str(item) for item in payload.get("channels", [])]
    return {"status": "SUCCEEDED", "channel_count": len(values), "channels": values[:32]}


class ExamplePlugin(AgentToolPlugin):
    @property
    def manifest(self):
        return AgentToolPluginManifest(
            plugin_id="example.telemetry",
            plugin_version="0.1.0",
            display_name="示例遥测工具",
            description="演示受信目录、importlib 发现与隔离进程执行。",
            review_status="DEVELOPMENT",
            source_refs=["examples/agent_tool_plugin_example.py"],
        )

    def exports(self):
        return [AgentToolExport(
            display_name="通道摘要",
            purpose="返回输入通道数量与名称摘要。",
            function_name="summarize_channels",
            spec=ToolSpec(
                tool_id="example.telemetry.summarize_channels",
                tool_version="0.1.0",
                input_schema={
                    "type": "object",
                    "properties": {"channels": {"type": "array", "items": {"type": "string"}, "maxItems": 128}},
                    "required": ["channels"],
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "required": ["status", "channel_count", "channels"],
                    "additionalProperties": True,
                },
                deterministic=True,
                side_effect_level=SideEffectLevel.NONE,
                required_permissions=[PermissionScope.READ_DATA],
                execution_mode="ISOLATED_PROCESS",
                timeout_s=10,
            ),
        )]


PLUGIN = ExamplePlugin()
