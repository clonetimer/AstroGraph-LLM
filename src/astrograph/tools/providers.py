from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from astrograph.integrations.platform_state import PlatformStateRepository


class ProviderTool(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_id: str
    provider_type: str
    tool_id: str
    display_name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    read_only: bool = True
    source: str = "provider"


class ToolProvider(ABC):
    provider_id: str
    provider_type: str

    @abstractmethod
    def health(self) -> dict[str, Any]: ...

    @abstractmethod
    def list_tools(self) -> list[ProviderTool]: ...

    @abstractmethod
    def invoke(self, tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class MCPProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]+$")
    name: str = Field(min_length=1, max_length=120)
    transport: Literal["streamable_http"] = "streamable_http"
    url: str = Field(min_length=8, max_length=2000)
    enabled: bool = True
    read_only: bool = True
    tool_allowlist: list[str] = Field(default_factory=list)
    timeout_s: float = Field(default=30.0, gt=0, le=300)


class MCPToolProvider(ToolProvider):
    provider_type = "mcp"

    def __init__(self, config: MCPProviderConfig):
        self.config = config
        self.provider_id = config.provider_id

    async def _list_async(self):
        try:
            from mcp import Client
        except ImportError as exc:
            raise RuntimeError("MCP Python SDK is not installed; install astrograph-core[mcp]") from exc
        async with Client(self.config.url) as client:
            return await client.list_tools()

    async def _call_async(self, tool_id: str, arguments: dict[str, Any]):
        try:
            from mcp import Client
        except ImportError as exc:
            raise RuntimeError("MCP Python SDK is not installed; install astrograph-core[mcp]") from exc
        async with Client(self.config.url) as client:
            # List first so the SDK can validate structured output against the advertised schema.
            await client.list_tools()
            return await client.call_tool(tool_id, arguments)

    def health(self) -> dict[str, Any]:
        try:
            listed = asyncio.run(asyncio.wait_for(self._list_async(), timeout=self.config.timeout_s))
            tools = list(getattr(listed, "tools", []) or [])
            return {"status": "PASS", "provider_id": self.provider_id, "tool_count": len(tools), "transport": self.config.transport, "url": self.config.url}
        except Exception as exc:
            return {"status": "FAIL", "provider_id": self.provider_id, "error": f"{type(exc).__name__}: {exc}", "transport": self.config.transport, "url": self.config.url}

    def list_tools(self) -> list[ProviderTool]:
        listed = asyncio.run(asyncio.wait_for(self._list_async(), timeout=self.config.timeout_s))
        allow = set(self.config.tool_allowlist)
        result: list[ProviderTool] = []
        for tool in list(getattr(listed, "tools", []) or []):
            name = str(tool.name)
            if allow and name not in allow:
                continue
            result.append(ProviderTool(
                provider_id=self.provider_id,
                provider_type=self.provider_type,
                tool_id=name,
                display_name=str(getattr(tool, "title", None) or name),
                description=str(getattr(tool, "description", None) or ""),
                input_schema=dict(getattr(tool, "input_schema", None) or {}),
                output_schema=(dict(tool.output_schema) if getattr(tool, "output_schema", None) else None),
                read_only=self.config.read_only,
                source="mcp",
            ))
        return result

    def invoke(self, tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.config.tool_allowlist and tool_id not in set(self.config.tool_allowlist):
            raise PermissionError("MCP tool is not allowlisted by AstroGraph provider policy")
        result = asyncio.run(asyncio.wait_for(self._call_async(tool_id, arguments), timeout=self.config.timeout_s))
        if bool(getattr(result, "is_error", False)):
            messages = []
            for block in getattr(result, "content", []) or []:
                text = getattr(block, "text", None)
                if text:
                    messages.append(str(text))
            raise RuntimeError("MCP tool returned error: " + " | ".join(messages[:4]))
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return structured if isinstance(structured, dict) else {"result": structured}
        content = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text is not None:
                content.append(str(text))
        return {"status": "SUCCEEDED", "content": content}


class ToolProviderManager:
    """Load external provider registrations from the SQLite control plane."""

    def __init__(self, state: PlatformStateRepository):
        self.state = state

    def providers(self) -> list[ToolProvider]:
        result: list[ToolProvider] = []
        for item in self.state.list_tool_providers(enabled_only=True):
            if item.get("provider_type") == "mcp":
                config = MCPProviderConfig.model_validate({**item.get("config", {}), "provider_id": item["provider_id"], "name": item["name"], "enabled": item["enabled"]})
                result.append(MCPToolProvider(config))
        return result

    def catalog(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for provider in self.providers():
            health = provider.health()
            if health.get("status") != "PASS":
                rows.append({"provider_id": provider.provider_id, "provider_type": provider.provider_type, "status": "UNAVAILABLE", "health": health, "tools": []})
                continue
            try:
                tools = [tool.model_dump(mode="json") for tool in provider.list_tools()]
            except Exception as exc:
                rows.append({"provider_id": provider.provider_id, "provider_type": provider.provider_type, "status": "UNAVAILABLE", "health": {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, "tools": []})
            else:
                rows.append({"provider_id": provider.provider_id, "provider_type": provider.provider_type, "status": "AVAILABLE", "health": health, "tools": tools})
        return rows
    def register_read_only_for_task(self, registry) -> tuple[list[str], list[str]]:
        """Snapshot enabled read-only MCP tools into one task-bound ToolRegistry.

        External providers are namespaced and treated as external I/O.  Write-capable
        MCP providers remain visible/manageable in the control plane but are not
        admitted to MissionAgent tasks until an explicit write-approval policy exists.
        """
        from astrograph.orchestration.models import PermissionScope, SideEffectLevel
        from astrograph.tools.registry import ToolDefinition, ToolSpec

        registered: list[str] = []
        limitations: list[str] = []
        for provider in self.providers():
            config = getattr(provider, "config", None)
            if config is not None and not bool(getattr(config, "read_only", True)):
                limitations.append(f"mcp_provider_write_blocked:{provider.provider_id}")
                continue
            try:
                tools = provider.list_tools()
            except Exception as exc:
                limitations.append(f"mcp_provider_unavailable:{provider.provider_id}:{type(exc).__name__}:{str(exc)[:200]}")
                continue
            for remote in tools:
                local_id = f"mcp.{provider.provider_id}.{remote.tool_id}"
                try:
                    timeout_s = float(getattr(config, "timeout_s", 30.0) if config is not None else 30.0)
                    registry.register(ToolDefinition(
                        spec=ToolSpec(
                            tool_id=local_id,
                            tool_version="mcp-2026-07-28",
                            input_schema=remote.input_schema or {"type": "object"},
                            output_schema=remote.output_schema or {"type": "object"},
                            deterministic=False,
                            side_effect_level=SideEffectLevel.EXTERNAL_IO,
                            required_permissions=[PermissionScope.READ_KNOWLEDGE],
                            timeout_s=timeout_s,
                            retry_count=0,
                            execution_mode="TRUSTED_IN_PROCESS",
                            max_payload_bytes=2_000_000,
                            max_output_bytes=8_000_000,
                        ),
                        handler=(lambda payload, p=provider, tid=remote.tool_id: p.invoke(tid, payload)),
                    ))
                except Exception as exc:
                    limitations.append(f"mcp_tool_registration_failed:{local_id}:{type(exc).__name__}:{str(exc)[:200]}")
                    continue
                registered.append(local_id)
        return registered, limitations

