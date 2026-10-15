from __future__ import annotations

from pathlib import Path
from typing import Any

from astrograph.data.thermal_star_adapter import materialize_d3_thermal_star_delivery
from astrograph.diagnosis.reaction_wheel import classify_rw_residual_evidence
from astrograph.evidence import D3RWExecutableEvidenceBuilder, D3RWExecutableEvidenceCatalog, EpisodeEvidenceBundle
from astrograph.evidence.thermal_star_executable import D3ThermalStarExecutableEvidenceBuilder
from astrograph.diagnosis.thermal_star import classify_thermal_star_evidence
from astrograph.investigation import DecisionRecord, HypothesisSet
from astrograph.knowledge import build_episode_evidence_graph
from astrograph.orchestration.models import PermissionScope, SideEffectLevel
from astrograph.object_plugins import (
    ObjectEvidenceBuildRequest,
    ObjectPluginRegistry,
    build_builtin_object_plugin_registry,
)
from astrograph.semantics import TelemetrySemanticsRegistry
from astrograph.tools.registry import ToolDefinition, ToolRegistry, ToolSpec


def build_agentic_tool_registry(
    *,
    include_thermal_star: bool = False,
    object_plugin_registry: ObjectPluginRegistry | None = None,
) -> ToolRegistry:
    """Build the V0.5.0 Agentic investigation tool allow-list.

    Every handler is deterministic and read-only. The registry does not expose
    arbitrary Python, arbitrary Cypher, knowledge publication, or automatic
    control actions.
    """

    registry = ToolRegistry()
    object_plugins = object_plugin_registry or build_builtin_object_plugin_registry()

    registry.register(
        ToolDefinition(
            spec=ToolSpec(
                tool_id="semantics.validate",
                input_schema={"type": "object", "required": ["protocol_root"]},
                output_schema={"type": "object"},
                tool_version=TelemetrySemanticsRegistry.registry_version,
                deterministic=True,
                side_effect_level=SideEffectLevel.READ_ONLY,
                required_permissions=[PermissionScope.READ_DATA],
            ),
            handler=_validate_semantics,
        )
    )

    registry.register(
        ToolDefinition(
            spec=ToolSpec(
                tool_id="evidence.catalog.read",
                input_schema={"type": "object", "required": ["evidence_catalog_path"]},
                output_schema={"type": "object"},
                tool_version="0.5.0",
                deterministic=True,
                side_effect_level=SideEffectLevel.READ_ONLY,
                required_permissions=[PermissionScope.READ_KNOWLEDGE],
            ),
            handler=_read_evidence_catalog,
        )
    )

    registry.register(
        ToolDefinition(
            spec=ToolSpec(
                tool_id="d3_rw.evidence.build",
                input_schema={
                    "type": "object",
                    "required": [
                        "d3_delivery_path",
                        "run_id",
                        "protocol_root",
                        "evidence_catalog_path",
                    ],
                },
                output_schema={"type": "object"},
                tool_version=D3RWExecutableEvidenceBuilder.builder_version,
                deterministic=True,
                side_effect_level=SideEffectLevel.READ_ONLY,
                required_permissions=[PermissionScope.READ_DATA, PermissionScope.READ_KNOWLEDGE],
                timeout_s=120.0,
            ),
            handler=_D3RWEvidenceBuildTool(object_plugins),
        )
    )

    registry.register(
        ToolDefinition(
            spec=ToolSpec(
                tool_id="diagnosis.invoke_core",
                input_schema={"type": "object", "required": ["wheel_evidence"]},
                output_schema={"type": "object"},
                tool_version="0.4.4",
                deterministic=True,
                side_effect_level=SideEffectLevel.READ_ONLY,
                required_permissions=[PermissionScope.RUN_DIAGNOSIS],
            ),
            handler=_invoke_d3_rw_core,
        )
    )

    if include_thermal_star:
        registry.register(
            ToolDefinition(
                spec=ToolSpec(
                    tool_id="d3_thermal_star.evidence.build",
                    input_schema={"type":"object","required":["delivery_path","run_id","protocol_root","evidence_catalog_path"]},
                    output_schema={"type":"object"},
                    tool_version=D3ThermalStarExecutableEvidenceBuilder.builder_version,
                    deterministic=True,
                    side_effect_level=SideEffectLevel.READ_ONLY,
                    required_permissions=[PermissionScope.READ_DATA, PermissionScope.READ_KNOWLEDGE],
                    timeout_s=120.0,
                ),
                handler=_D3ThermalStarEvidenceBuildTool(),
            )
        )

        registry.register(
            ToolDefinition(
                spec=ToolSpec(
                    tool_id="diagnosis.invoke_thermal_star_core",
                    input_schema={"type":"object","required":["evidence_facts"]},
                    output_schema={"type":"object"},
                    tool_version="0.5.3",
                    deterministic=True,
                    side_effect_level=SideEffectLevel.READ_ONLY,
                    required_permissions=[PermissionScope.RUN_DIAGNOSIS],
                ),
                handler=_invoke_thermal_star_core,
            )
        )

    registry.register(
        ToolDefinition(
            spec=ToolSpec(
                tool_id="evidence_graph.build",
                input_schema={
                    "type": "object",
                    "required": ["bundle", "hypothesis_set", "decision_record"],
                },
                output_schema={"type": "object"},
                tool_version="0.5.0",
                deterministic=True,
                side_effect_level=SideEffectLevel.READ_ONLY,
                required_permissions=[PermissionScope.READ_DATA, PermissionScope.READ_KNOWLEDGE],
            ),
            handler=_build_evidence_graph,
        )
    )

    return registry


def _validate_semantics(payload: dict[str, Any]) -> dict[str, Any]:
    semantics = TelemetrySemanticsRegistry(str(payload["protocol_root"]))
    return {
        "validation": semantics.validation.model_dump(mode="json"),
        "event_window_semantics": semantics.event_window_semantics,
        "rw_installation": semantics.rw_installation,
        "registry_version": semantics.registry_version,
    }


def _read_evidence_catalog(payload: dict[str, Any]) -> dict[str, Any]:
    catalog = D3RWExecutableEvidenceCatalog(str(payload["evidence_catalog_path"]))
    return {
        "catalog_version": catalog.catalog_version,
        "catalog_sha256": catalog.sha256,
        "registered_fault_ids": sorted(catalog.registered_faults),
        "feature_definitions": [item.model_dump(mode="json") for item in catalog.feature_definitions],
        "evidence_definitions": [item.model_dump(mode="json") for item in catalog.evidence_definitions],
    }


class _D3RWEvidenceBuildTool:
    """Legacy tool alias backed by the registered reaction-wheel object plugin."""

    plugin_id = "builtin.d3.reaction_wheel"

    def __init__(self, plugins: ObjectPluginRegistry) -> None:
        self.plugins = plugins

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.plugins.get(self.plugin_id).build_evidence(
            ObjectEvidenceBuildRequest(
                delivery_path=str(payload["d3_delivery_path"]),
                run_id=str(payload["run_id"]),
                protocol_root=str(payload["protocol_root"]),
                evidence_catalog_path=str(payload["evidence_catalog_path"]),
            )
        )
        return {
            "bundle": result.bundle,
            "core_evidence": result.core_payload,
            "builder_version": result.builder_version,
            "object_plugin_id": result.plugin_id,
            "object_plugin_version": result.plugin_version,
        }


def _invoke_d3_rw_core(payload: dict[str, Any]) -> dict[str, Any]:
    status, prediction, wheel_index, evidence, explanation = classify_rw_residual_evidence(
        list(payload["wheel_evidence"]),
        nominal_case=bool(payload.get("nominal_case", False)),
    )
    return {
        "diagnosis_status": status,
        "predicted_case_id": prediction,
        "inferred_wheel_index": wheel_index,
        "selected_evidence": evidence,
        "explanation": explanation,
        "core_version": "v0.4.4-d3-rw-deterministic-e0-v1",
        "nominal_case_switch_is_evaluation_only": bool(payload.get("nominal_case", False)),
    }


class _D3ThermalStarEvidenceBuildTool:
    def __init__(self) -> None:
        self._directory_cache: dict[tuple[str,str,str], D3ThermalStarExecutableEvidenceBuilder] = {}
    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        delivery=Path(str(payload["delivery_path"])).resolve(); protocol=str(Path(str(payload["protocol_root"])).resolve()); catalog=str(Path(str(payload["evidence_catalog_path"])).resolve())
        if delivery.is_dir():
            key=(str(delivery),protocol,catalog); builder=self._directory_cache.get(key)
            if builder is None:
                builder=D3ThermalStarExecutableEvidenceBuilder(delivery,semantics_registry=TelemetrySemanticsRegistry(protocol),catalog=D3RWExecutableEvidenceCatalog(catalog)); self._directory_cache[key]=builder
            bundle=builder.build_run(str(payload["run_id"]))
        else:
            with materialize_d3_thermal_star_delivery(delivery) as root:
                builder=D3ThermalStarExecutableEvidenceBuilder(root,semantics_registry=TelemetrySemanticsRegistry(protocol),catalog=D3RWExecutableEvidenceCatalog(catalog)); bundle=builder.build_run(str(payload["run_id"]))
        return {"bundle":bundle.model_dump(mode="json"),"core_payload":builder.core_payload(bundle),"builder_version":builder.builder_version}

def _invoke_thermal_star_core(payload: dict[str, Any]) -> dict[str, Any]:
    return classify_thermal_star_evidence(list(payload["evidence_facts"]),nominal_case=bool(payload.get("nominal_case",False)))

def _build_evidence_graph(payload: dict[str, Any]) -> dict[str, Any]:
    graph = build_episode_evidence_graph(
        EpisodeEvidenceBundle.model_validate(payload["bundle"]),
        HypothesisSet.model_validate(payload["hypothesis_set"]),
        DecisionRecord.model_validate(payload["decision_record"]),
    )
    return graph.model_dump(mode="json")
