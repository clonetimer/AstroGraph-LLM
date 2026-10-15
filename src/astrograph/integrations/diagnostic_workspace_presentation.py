"""Read-only production projection for the R34 Diagnostic Workspace.

The projection consumes frozen platform authority, a validated SystemModelGraph,
and exact SimulationTrace bytes retained when an existing R29/R31 provider ran.
It never runs a predictor, graph query, Probe, or simulator.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.audit.store import sha256_json
from astrograph.diagnosis.dxc_pump_reduced_order import DXCWaterPumpPhysicsDecision
from astrograph.diagnosis.dxc_reduced_order_physics import DXCReducedOrderPhysicsDecision
from astrograph.integrations.maintenance_presentation import extract_formal_evidence_bundle
from astrograph.knowledge.system_model_graph import SystemEdgeKind, SystemModelGraph
from astrograph.simulation.diagnostic_provider import SimulationTrace


WORKSPACE_PRESENTATION_SCHEMA_VERSION = "astrograph-diagnostic-workspace-presentation.v1"
WORKSPACE_SOURCE_SCHEMA_VERSION = "astrograph-diagnostic-workspace-source-bundle.v2"
SIMULATION_SEMANTICS = "SIMULATED_TRAJECTORY_NOT_PROBABILITY"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class WorkspaceAuthoritySnapshot(_Strict):
    diagnosis_disposition: str
    release_status: str
    evidence_bundle_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class WorkspaceSimulationSource(_Strict):
    provider_id: Literal[
        "adapt-resistive-lightbulb-reduced-order-v1",
        "adapt-waterpump-reduced-order-v1",
    ]
    diagnosis_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    channel_units: dict[str, str]
    trace: SimulationTrace
    decision: dict[str, Any]

    @model_validator(mode="after")
    def bind_existing_decision_to_trace(self) -> "WorkspaceSimulationSource":
        if set(self.channel_units) != set(self.trace.channels):
            raise ValueError("workspace simulation units must exactly cover trace channels")
        if self.provider_id == "adapt-waterpump-reduced-order-v1":
            decision = DXCWaterPumpPhysicsDecision.model_validate(self.decision)
            fits = [item for item in decision.candidate_fits if item.diagnosis_id == self.diagnosis_id]
            if len(fits) != 1 or fits[0].fault_id != self.candidate_id:
                raise ValueError("workspace pump trace is not bound to one decision candidate")
            expected_provenance = "ASTROGRAPH_ADAPT_WATERPUMP_REDUCED_ORDER_V1"
            expected_channels = {fits[0].current_sensor_id, fits[0].flow_sensor_id}
        else:
            decision = DXCReducedOrderPhysicsDecision.model_validate(self.decision)
            fits = [item for item in decision.candidate_fits if item.diagnosis_id == self.diagnosis_id]
            if len(fits) != 1 or "|".join(fits[0].fault_ids) != self.candidate_id:
                raise ValueError("workspace load trace is not bound to one decision candidate")
            expected_provenance = "ASTROGRAPH_ADAPT_RESISTIVE_LOAD_REDUCED_ORDER_V1"
            expected_channels = {fits[0].current_sensor_id}
        if set(self.trace.channels) != expected_channels:
            raise ValueError("workspace simulation trace channels do not match decision receipt")
        if fits[0].simulation_trace_sha256 != self.trace.trace_sha256:
            raise ValueError("workspace simulation trace hash does not match decision receipt")
        if self.trace.provenance_kind != expected_provenance:
            raise ValueError("workspace simulation trace provenance does not match provider")
        return self


class DiagnosticWorkspaceSourceBundle(_Strict):
    schema_version: Literal[WORKSPACE_SOURCE_SCHEMA_VERSION] = WORKSPACE_SOURCE_SCHEMA_VERSION
    job_id: str = Field(pattern=r"^JOB-[0-9A-F]{24}$")
    authority_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    system_model_graph: SystemModelGraph | None = None
    simulation_sources: tuple[WorkspaceSimulationSource, ...] = ()
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def verify_bundle(self) -> "DiagnosticWorkspaceSourceBundle":
        payload = self.model_dump(mode="json", exclude={"bundle_sha256"})
        if sha256_json(payload) != self.bundle_sha256:
            raise ValueError("diagnostic workspace source bundle hash mismatch")
        identities = [(item.provider_id, item.diagnosis_id, item.trace.trace_sha256) for item in self.simulation_sources]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate diagnostic workspace simulation source")
        return self


def authority_snapshot(report: dict[str, Any], maintenance: dict[str, Any]) -> WorkspaceAuthoritySnapshot:
    pipeline = ((report.get("analysis") or {}).get("pipeline_result") or {})
    release = (
        maintenance.get("release_status")
        or report.get("release_status")
        or pipeline.get("release_status")
        or ("FORMAL_RELEASE_ALLOWED" if bool(report.get("formal_diagnosis_publishable") or pipeline.get("formal_diagnosis_publishable")) else "FORMAL_RELEASE_BLOCKED")
    )
    bundle = extract_formal_evidence_bundle(report)
    bundle_sha = sha256_json(bundle.model_dump(mode="json")) if bundle is not None else None
    return WorkspaceAuthoritySnapshot(
        diagnosis_disposition=str(maintenance.get("diagnosis_disposition") or maintenance.get("diagnosis_status") or "UNAVAILABLE"),
        release_status=str(release),
        evidence_bundle_sha256=bundle_sha,
    )


def build_source_bundle(
    *, job_id: str, authority: WorkspaceAuthoritySnapshot,
    candidate_context_sha256: str,
    system_model_graph: SystemModelGraph | None = None,
    simulation_sources: tuple[WorkspaceSimulationSource, ...] = (),
) -> DiagnosticWorkspaceSourceBundle:
    payload = {
        "schema_version": WORKSPACE_SOURCE_SCHEMA_VERSION,
        "job_id": job_id,
        "authority_snapshot_sha256": sha256_json(authority.model_dump(mode="json")),
        "candidate_context_sha256": candidate_context_sha256,
        "system_model_graph": system_model_graph.model_dump(mode="json") if system_model_graph else None,
        "simulation_sources": [item.model_dump(mode="json") for item in simulation_sources],
    }
    return DiagnosticWorkspaceSourceBundle(**payload, bundle_sha256=sha256_json(payload))


def candidate_context_identity(maintenance: dict[str, Any]) -> dict[str, str | None]:
    fault_id = maintenance.get("fault_id") or maintenance.get("candidate")
    component = maintenance.get("component_id") or maintenance.get("component")
    if fault_id and ":" in str(fault_id):
        component = str(fault_id).split(":", 1)[0]
    return {
        "selected_fault_id": str(fault_id) if fault_id else None,
        "component_id": str(component) if component else None,
    }


def _candidate_context(maintenance: dict[str, Any]) -> dict[str, Any]:
    identity = candidate_context_identity(maintenance)
    alternatives = []
    for item in maintenance.get("alternative_hypotheses") or []:
        if isinstance(item, dict):
            alternatives.append({
                "candidate_id": str(item.get("fault_id") or item.get("id") or ""),
                "label": str(item.get("label") or item.get("fault_id") or item.get("id") or ""),
            })
    return {
        **identity,
        "subsystem_id": maintenance.get("subsystem_id") or maintenance.get("subsystem"),
        "alternatives": [item for item in alternatives if item["candidate_id"]],
    }


def _observed_traces(visualization: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for channel in visualization.get("channels") or []:
        samples = []
        for point in channel.get("points") or []:
            try:
                samples.append({"time_s": float(point["t"]), "value": float(point["v"])})
            except (KeyError, TypeError, ValueError):
                continue
        if len(samples) >= 2:
            result.append({
                "trace_id": f"observed:{channel.get('name')}",
                "channel": str(channel.get("name")),
                "label": str(channel.get("label") or channel.get("name")),
                "unit": str(channel.get("unit") or ""),
                "samples": samples,
                "semantics": "OBSERVED_TELEMETRY_DISPLAY_ONLY",
            })
    return result


def _mechanism_graph(
    graph: SystemModelGraph | None, *, component_id: str | None,
    observed_channels: set[str], max_nodes: int = 80,
) -> dict[str, Any]:
    if graph is None:
        return {"status": "UNAVAILABLE", "reason_code": "SYSTEM_MODEL_GRAPH_NOT_RETAINED", "source": None, "nodes": [], "edges": [], "provenance_sha256": None}
    component_node = f"component:{component_id}" if component_id else None
    known = {item.node_id: item for item in graph.nodes}
    if not component_node or component_node not in known:
        return {"status": "UNAVAILABLE", "reason_code": "EXACT_DIAGNOSIS_COMPONENT_NOT_IN_SYSTEM_MODEL_GRAPH", "source": "SYSTEM_MODEL_GRAPH", "nodes": [], "edges": [], "provenance_sha256": graph.graph_sha256}
    selected: set[str] = {component_node}
    allowed = {SystemEdgeKind.CONNECTED_TO, SystemEdgeKind.COMMANDS, SystemEdgeKind.HAS_FAULT_MODE, SystemEdgeKind.INSTANCE_OF}
    for edge in graph.edges:
        if edge.relation not in allowed:
            continue
        if edge.source_id == component_node:
            selected.add(edge.target_id)
        if edge.target_id == component_node:
            selected.add(edge.source_id)
    fault_prefix = f"fault:{component_id}:"
    selected.update(node_id for node_id in known if node_id.startswith(fault_prefix))
    ordered = [component_node, *[node_id for node_id in sorted(selected) if node_id != component_node][: max_nodes - 1]]
    kept = set(ordered)
    nodes = []
    for node_id in ordered:
        item = known[node_id]
        props = item.properties
        label = props.get("fault_id") or props.get("component_id") or props.get("command_id") or props.get("type_id") or props.get("name") or node_id
        component_name = str(props.get("component_id") or node_id.removeprefix("component:"))
        related = sorted(name for name in observed_channels if name in {component_name, node_id})
        nodes.append({"id": node_id, "label": str(label), "kind": item.kind.value, "properties": props, "related_channels": related})
    edges = [
        {"from": edge.source_id, "to": edge.target_id, "relation": edge.relation.value, "label": edge.relation.value}
        for edge in graph.edges if edge.source_id in kept and edge.target_id in kept
    ]
    return {"status": "AVAILABLE", "reason_code": None, "source": "SYSTEM_MODEL_GRAPH", "nodes": nodes, "edges": edges, "provenance_sha256": graph.graph_sha256}


def _simulation_traces(bundle: DiagnosticWorkspaceSourceBundle | None) -> list[dict[str, Any]]:
    if bundle is None:
        return []
    result = []
    for source in bundle.simulation_sources:
        for channel, values in source.trace.channels.items():
            result.append({
                "trace_id": f"{source.trace.trace_id}:{channel}",
                "provider_id": source.provider_id,
                "diagnosis_id": source.diagnosis_id,
                "candidate_id": source.candidate_id,
                "channel": channel,
                "unit": source.channel_units[channel],
                "samples": [
                    {"time_s": timestamp, "value": value}
                    for timestamp, value in zip(source.trace.time_s, values)
                ],
                "source_trace_sha256": source.trace.trace_sha256,
                "semantics": SIMULATION_SEMANTICS,
            })
    return result


def _evidence_links(maintenance: dict[str, Any]) -> list[dict[str, Any]]:
    summary = maintenance.get("evidence_summary") or {}
    result = []
    for role, keys in (("SUPPORTING", ("supporting_items", "supporting")), ("CONTRADICTING", ("contradicting_items", "contradicting")), ("MISSING", ("missing_items", "missing"))):
        items = next((summary.get(key) for key in keys if summary.get(key) is not None), [])
        for item in items or []:
            if isinstance(item, dict):
                result.append({
                    "evidence_id": str(item.get("evidence_fact_id") or item.get("evidence_definition_id") or ""),
                    "label": str(item.get("label") or item.get("evidence_definition_id") or ""),
                    "role": role,
                    "channel": item.get("channel"),
                    "component_id": item.get("component_id"),
                })
    return [item for item in result if item["evidence_id"]]


def _agent_links(report: dict[str, Any]) -> list[dict[str, Any]]:
    agent = report.get("agent_execution") or ((report.get("analysis") or {}).get("agent_execution") or {})
    investigation = agent.get("investigation") or {}
    result = []
    for status, items in (("COMPLETED", investigation.get("findings") or []), ("REQUIRED", investigation.get("required_observations") or [])):
        for index, item in enumerate(items):
            value = item if isinstance(item, dict) else {"probe_id": str(item)}
            result.append({
                "link_id": str(value.get("id") or value.get("result_hash") or f"{status.lower()}-{index}"),
                "probe_id": str(value.get("probe_id") or value.get("kind") or ""),
                "status": status,
                "channel": value.get("channel") or value.get("trace_id"),
                "component_id": value.get("component_id") or value.get("graph_node_id"),
            })
    return result


def build_diagnostic_workspace_presentation(
    *, report: dict[str, Any], maintenance: dict[str, Any], visualization: dict[str, Any],
    source_bundle: DiagnosticWorkspaceSourceBundle | None = None,
) -> dict[str, Any]:
    authority = authority_snapshot(report, maintenance)
    source_status = "NOT_RETAINED"
    graph = None
    if source_bundle is not None:
        if source_bundle.job_id != str(report.get("job_id")):
            raise ValueError("diagnostic workspace source job mismatch")
        if source_bundle.authority_snapshot_sha256 != sha256_json(authority.model_dump(mode="json")):
            raise ValueError("diagnostic workspace source authority snapshot drift")
        current_candidate_sha256 = sha256_json(candidate_context_identity(maintenance))
        if source_bundle.candidate_context_sha256 != current_candidate_sha256:
            raise ValueError("diagnostic workspace source candidate context drift")
        graph = source_bundle.system_model_graph
        source_status = "VERIFIED"
    context = _candidate_context(maintenance)
    observed = _observed_traces(visualization)
    graph_projection = _mechanism_graph(
        graph, component_id=context["component_id"],
        observed_channels={item["channel"] for item in observed},
    )
    payload = {
        "schema_version": WORKSPACE_PRESENTATION_SCHEMA_VERSION,
        "job_id": str(report.get("job_id")),
        "authority_snapshot": authority.model_dump(mode="json"),
        "candidate_context": context,
        "mechanism_graph": graph_projection,
        "observed_traces": observed,
        "simulation_traces": _simulation_traces(source_bundle),
        "evidence_links": _evidence_links(maintenance),
        "agent_links": _agent_links(report),
        "source_bundle_status": source_status,
        "source_bundle_sha256": source_bundle.bundle_sha256 if source_bundle else None,
        "claim_boundary": "Read-only UI projection. Selection, graph, simulation, Evidence links, and Agent links do not modify formal diagnosis, EvidenceFact, or release authority.",
    }
    payload["presentation_sha256"] = sha256_json(payload)
    return payload


__all__ = [
    "DiagnosticWorkspaceSourceBundle",
    "WORKSPACE_PRESENTATION_SCHEMA_VERSION",
    "WORKSPACE_SOURCE_SCHEMA_VERSION",
    "WorkspaceAuthoritySnapshot",
    "WorkspaceSimulationSource",
    "authority_snapshot",
    "candidate_context_identity",
    "build_diagnostic_workspace_presentation",
    "build_source_bundle",
]
