"""Tool catalog projected for the operator UI and headless clients."""
from __future__ import annotations

from typing import Any

from astrograph.agents.mission.modes import allowed_tools, TASK_TOOLS, COMMON_TOOLS
from astrograph.agents.mission.service import CATALOG
from astrograph.orchestration.models import PermissionScope, SideEffectLevel
from astrograph.tools.plugins import get_trusted_agent_tool_plugin_manager


DISPLAY: dict[str, tuple[str, str]] = {
    "model.infer": ("模型运行（工程）", "工程"),
    "model.capabilities": ("模型适配检查（工程）", "工程"),
    "model.execute_compatible": ("兼容模型执行（工程）", "工程"),
    "dataset.replay": ("数据集回放", "实验"),
    "graph.reason": ("证据关系检查（工程）", "工程"),
    "knowledge.graph.mechanism": ("故障机理追溯", "维护"),
    "knowledge.graph.match_observations": ("按现象查找相关故障", "维护"),
    "diagnosis.fuse": ("综合诊断", "维护"),
    "document.convert": ("文档解析（工程）", "工程"),
    "document.clean": ("文档整理（工程）", "工程"),
    "rag.retrieve": ("查找本任务技术资料", "维护"),
    "literature.search": ("查找外部技术资料", "维护"),
    "graph.build": ("候选知识图构建", "知识"),
    "model.train": ("离线模型训练", "实验"),
    "model.infer_candidate": ("候选模型推理", "实验"),
    "exploration.propose": ("可证伪探索提案", "探索"),
    "output.structured": ("生成诊断记录", "维护"),
}

PURPOSE_ZH: dict[str, str] = {
    "model.infer": "准备受治理的原生 ADCS/PHM 推理，或对通用时序执行带语义边界的检测。",
    "model.capabilities": "检查当前通道与已登记模型合同是否兼容；发现兼容性不等于授权执行。",
    "model.execute_compatible": "只通过主机绑定的兼容提供方执行已准入模型。",
    "dataset.replay": "离线回放带真值数据集，用于实验评估，不作为在线诊断。",
    "graph.reason": "检查诊断证据之间的已登记关系，主要用于工程审计。",
    "knowledge.graph.mechanism": "沿设备、现象和故障关系查看可能的传播路径，帮助解释为什么当前现象与候选故障相关。",
    "knowledge.graph.match_observations": "根据已观测到的异常现象，在本机知识图谱中查找相关故障候选，作为复核线索。",
    "diagnosis.fuse": "综合遥测、知识与规则证据，形成维护人员可复核的故障候选；不会绕过人工复核门槛。",
    "document.convert": "把受支持文档转换为可检索文本，同时保留来源工件。",
    "document.clean": "保守清洗文本，保留数字、单位、否定词和来源。",
    "rag.retrieve": "只在当前任务已转换的本地文档中检索来源片段。",
    "literature.search": "通过主机配置的 PaperSeek 查询外部文献；不接触遥测真值。",
    "graph.build": "把来源约束的信息整理为待审核图候选，不直接发布到 Active KG。",
    "model.train": "仅在明确审批的离线三任务训练清单上训练候选模型。",
    "model.infer_candidate": "用离线候选模型和精确通道合同执行无真值推理。",
    "exploration.propose": "仅在创造模式生成隔离的、可证伪的离线假设；不进入诊断投票。",
    "output.structured": "把已冻结的诊断结果整理成可保存、可审计的诊断记录。",
}

IMPLEMENTATION: dict[str, tuple[str, str]] = {
    "model.infer": ("astrograph.agents.mission.service", "MissionTask._infer_native"),
    "model.capabilities": ("astrograph.agents.mission.service", "MissionTask._model_capabilities"),
    "model.execute_compatible": ("astrograph.agents.mission.service", "MissionTask._execute_compatible"),
    "dataset.replay": ("astrograph.agents.mission.service", "MissionTask._replay_dataset"),
    "graph.reason": ("astrograph.agents.mission.service", "MissionTask._graph_reason"),
    "knowledge.graph.mechanism": ("astrograph.tools.neo4j_maintenance", "get_fault_mechanism"),
    "knowledge.graph.match_observations": ("astrograph.tools.neo4j_maintenance", "match_observations"),
    "diagnosis.fuse": ("astrograph.agents.mission.service", "MissionTask._diagnose"),
    "document.convert": ("astrograph.agents.mission.documents", "DocumentTools.convert"),
    "document.clean": ("astrograph.agents.mission.documents", "DocumentTools.clean"),
    "rag.retrieve": ("astrograph.agents.mission.documents", "DocumentTools.retrieve"),
    "literature.search": ("astrograph.agents.mission.service", "MissionTask._literature"),
    "graph.build": ("astrograph.agents.mission.service", "MissionTask._build_graph"),
    "model.train": ("astrograph.agents.mission.service", "MissionTask._train"),
    "model.infer_candidate": ("astrograph.agents.mission.service", "MissionTask._candidate_infer"),
    "exploration.propose": ("astrograph.agents.mission.service", "MissionTask._explore"),
    "output.structured": ("astrograph.agents.mission.service", "MissionTask._structured_output"),
}

PERMISSIONS: dict[str, list[PermissionScope]] = {
    "model.infer": [PermissionScope.READ_DATA, PermissionScope.RUN_DIAGNOSIS],
    "model.capabilities": [PermissionScope.READ_DATA],
    "model.execute_compatible": [PermissionScope.READ_DATA, PermissionScope.RUN_DIAGNOSIS],
    "dataset.replay": [PermissionScope.READ_DATA, PermissionScope.RUN_EXPERIMENT],
    "graph.reason": [PermissionScope.READ_KNOWLEDGE],
    "knowledge.graph.mechanism": [PermissionScope.READ_KNOWLEDGE],
    "knowledge.graph.match_observations": [PermissionScope.READ_KNOWLEDGE],
    "diagnosis.fuse": [PermissionScope.READ_DATA, PermissionScope.READ_KNOWLEDGE, PermissionScope.RUN_DIAGNOSIS],
    "document.convert": [PermissionScope.READ_ARTIFACTS],
    "document.clean": [PermissionScope.READ_ARTIFACTS],
    "rag.retrieve": [PermissionScope.READ_ARTIFACTS],
    "literature.search": [PermissionScope.READ_ARTIFACTS],
    "graph.build": [PermissionScope.READ_ARTIFACTS, PermissionScope.WRITE_ARTIFACTS],
    "model.train": [PermissionScope.RUN_EXPERIMENT],
    "model.infer_candidate": [PermissionScope.READ_DATA],
    "exploration.propose": [PermissionScope.READ_ARTIFACTS, PermissionScope.WRITE_ARTIFACTS],
    "output.structured": [PermissionScope.RENDER_REPORT],
}

PLATFORM_BOUND = {"diagnosis.fuse", "rag.retrieve", "output.structured", "exploration.propose"}


def _builtin_tool(tool_id: str) -> dict[str, Any]:
    display_name, category = DISPLAY.get(tool_id, (tool_id, "其他"))
    modes = [mode for mode in ("normal", "thinking", "creative") if tool_id in allowed_tools("diagnose", mode)]
    module, symbol = IMPLEMENTATION.get(tool_id, ("", ""))
    query_input = tool_id in {"rag.retrieve", "literature.search"}
    if tool_id == "knowledge.graph.mechanism":
        custom_input = {"type":"object","properties":{"fault_id":{"type":"string"},"max_nodes":{"type":"integer","minimum":1,"maximum":100}},"required":["fault_id"],"additionalProperties":False}
    elif tool_id == "knowledge.graph.match_observations":
        custom_input = {"type":"object","properties":{"anomaly_ids":{"type":"array","items":{"type":"string"},"minItems":1,"maxItems":20}},"required":["anomaly_ids"],"additionalProperties":False}
    else:
        custom_input = None
    side_effect = SideEffectLevel.EXTERNAL_IO if tool_id == "literature.search" else SideEffectLevel.WRITE_ARTIFACT
    deterministic = tool_id not in {"graph.build", "literature.search"}
    tasks = sorted(task for task, tool_ids in TASK_TOOLS.items() if tool_id in tool_ids or tool_id in COMMON_TOOLS)
    return {
        "source": "builtin",
        "tool_id": tool_id,
        "display_name": display_name,
        "category": category,
        "operator_facing": category in {"维护", "检索"},
        "purpose": PURPOSE_ZH.get(tool_id, CATALOG.get(tool_id, "")),
        "allowed_tasks": tasks,
        "allowed_work_modes": modes,
        "platform_upload_bound": tool_id in PLATFORM_BOUND,
        "spec": {
            "input_schema": (custom_input or ({
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 4000}},
                "required": ["query"],
                "additionalProperties": False,
            } if query_input else {"type": "object", "additionalProperties": False})),
            "output_schema": {"type": "object", "required": ["status"]},
            "deterministic": deterministic,
            "side_effect_level": side_effect.value,
            "required_permissions": [item.value for item in PERMISSIONS.get(tool_id, [PermissionScope.READ_ARTIFACTS])],
            "execution_mode": "TRUSTED_IN_PROCESS",
        },
        "implementation": {
            "module": module,
            "symbol": symbol,
            "registry": "astrograph.tools.registry.ToolRegistry",
            "task_snapshot": "sealed after registration",
        },
    }


def platform_tool_catalog() -> dict[str, Any]:
    builtin = [_builtin_tool(tool_id) for tool_id in sorted(CATALOG)]
    manager = get_trusted_agent_tool_plugin_manager()
    plugins = manager.catalog()
    for item in plugins:
        item.setdefault("category", "插件")
        item["platform_upload_bound"] = False
    plugin_status = manager.status()
    # Viewer-facing catalog exposes operational state but not the administrator's absolute filesystem path.
    plugin_status = {**plugin_status, "trusted_root": None}
    return {
        "schema_version": "astrograph-platform-tool-catalog.v1",
        "tools": builtin + plugins,
        "plugin_status": plugin_status,
        "boundaries": {
            "running_registry_hot_mutation_allowed": False,
            "hot_reload_affects": "future registry snapshots only",
            "dynamic_plugin_execution": "isolated process with SHA-256 pin",
        },
    }
