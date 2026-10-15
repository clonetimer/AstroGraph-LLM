"""Deterministic engineering-system graph for bounded model-based diagnosis.

This graph is intentionally separate from the runtime EvidenceGraph.  It carries
engineering structure (component instances/types, explicit command bindings,
topology and admitted failure modes) and is content-bound to its source catalog.
Structural edges may constrain candidate generation; they are not formal
EvidenceFact records and do not by themselves prove a fault or causal relation.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.diagnosis.model_based import ModelConflict


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class SystemNodeKind(str, Enum):
    SYSTEM = "SYSTEM"
    COMPONENT_TYPE = "COMPONENT_TYPE"
    COMPONENT = "COMPONENT"
    SENSOR = "SENSOR"
    COMMAND = "COMMAND"
    FAULT_MODE = "FAULT_MODE"


class SystemEdgeKind(str, Enum):
    CONTAINS = "CONTAINS"
    INSTANCE_OF = "INSTANCE_OF"
    CONNECTED_TO = "CONNECTED_TO"
    COMMANDS = "COMMANDS"
    HAS_FAULT_MODE = "HAS_FAULT_MODE"


class SystemModelNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(min_length=1)
    kind: SystemNodeKind
    properties: dict[str, Any] = Field(default_factory=dict)


class SystemModelEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    relation: SystemEdgeKind
    target_id: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)


class SystemModelGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-system-model-graph.v1"
    system_id: str = Field(min_length=1)
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    nodes: tuple[SystemModelNode, ...]
    edges: tuple[SystemModelEdge, ...]
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = (
        "Structural engineering graph derived from an admitted catalog. Connectivity and explicit command bindings "
        "constrain candidate generation but do not by themselves prove physical causality, probability, or a fault."
    )

    @model_validator(mode="after")
    def graph_integrity(self) -> "SystemModelGraph":
        ids = [item.node_id for item in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("system-model node IDs must be unique")
        known = set(ids)
        seen_edges: set[tuple[str, str, str, str]] = set()
        for edge in self.edges:
            if edge.source_id not in known or edge.target_id not in known:
                raise ValueError("system-model edge references unknown node")
            key = (edge.source_id, edge.relation.value, edge.target_id, _digest(edge.properties))
            if key in seen_edges:
                raise ValueError("system-model edges must be unique")
            seen_edges.add(key)
        payload = {
            "system_id": self.system_id,
            "source_artifact_sha256": self.source_artifact_sha256,
            "nodes": [item.model_dump(mode="json") for item in self.nodes],
            "edges": [item.model_dump(mode="json") for item in self.edges],
        }
        if _digest(payload) != self.graph_sha256:
            raise ValueError("system-model graph_sha256 mismatch")
        return self

    def command_target(self, command_id: str) -> str | None:
        source = f"command:{command_id}"
        targets = [
            edge.target_id
            for edge in self.edges
            if edge.source_id == source and edge.relation == SystemEdgeKind.COMMANDS
        ]
        if len(targets) > 1:
            raise ValueError(f"command {command_id} is ambiguously bound")
        return targets[0] if targets else None

    def component_neighbors(self) -> dict[str, set[str]]:
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in self.edges:
            if edge.relation != SystemEdgeKind.CONNECTED_TO:
                continue
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)
        return adjacency

    def fault_ids_for_component(self, component_node_id: str) -> tuple[str, ...]:
        return tuple(sorted(
            edge.target_id.removeprefix("fault:")
            for edge in self.edges
            if edge.source_id == component_node_id and edge.relation == SystemEdgeKind.HAS_FAULT_MODE
        ))


class ObservationMismatch(BaseModel):
    """One host-admitted inconsistency between expected and observed behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mismatch_id: str = Field(min_length=1)
    sensor_component_id: str = Field(min_length=1)
    command_id: str | None = None
    source_ids: tuple[str, ...] = ()
    explanation: str = ""


def build_system_model_graph(catalog: dict[str, Any]) -> SystemModelGraph:
    """Build a content-bound graph from ``nasa_adapt_adapter.system_catalog`` output."""
    nodes: dict[str, SystemModelNode] = {}
    edges: dict[tuple[str, str, str, str], SystemModelEdge] = {}

    def add_node(node: SystemModelNode) -> None:
        existing = nodes.get(node.node_id)
        if existing is not None and existing != node:
            raise ValueError(f"conflicting node definition: {node.node_id}")
        nodes[node.node_id] = node

    def add_edge(
        source: str,
        relation: SystemEdgeKind,
        target: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        props = properties or {}
        edge = SystemModelEdge(source_id=source, relation=relation, target_id=target, properties=props)
        key = (source, relation.value, target, _digest(props))
        edges[key] = edge

    system_id = str(catalog.get("system_universe") or "UNKNOWN_SYSTEM")
    source_sha = str(catalog.get("xml_sha256") or "")
    if len(source_sha) != 64 or any(ch not in "0123456789abcdef" for ch in source_sha):
        raise ValueError("catalog must carry a valid xml_sha256")
    if catalog.get("ambiguous_command_ids"):
        raise ValueError("catalog contains ambiguous command-instance bindings")

    system_node = f"system:{system_id}"
    add_node(SystemModelNode(node_id=system_node, kind=SystemNodeKind.SYSTEM, properties={"name": system_id}))

    type_map = catalog.get("component_types") or {}
    for type_id, item in sorted(type_map.items()):
        type_node = f"component_type:{type_id}"
        add_node(SystemModelNode(
            node_id=type_node,
            kind=SystemNodeKind.COMPONENT_TYPE,
            properties={
                "type_id": type_id,
                "kind": item.get("kind"),
                "unit": item.get("unit"),
                "unit_status": item.get("unit_status"),
                "fault_modes": list(item.get("fault_modes") or []),
                "nominal_modes": list(item.get("nominal_modes") or []),
            },
        ))
        add_edge(system_node, SystemEdgeKind.CONTAINS, type_node)

    component_nodes: dict[str, str] = {}
    for component in sorted(catalog.get("components") or [], key=lambda value: str(value.get("component_id"))):
        component_id = str(component["component_id"])
        component_node = f"component:{component_id}"
        component_nodes[component_id] = component_node
        is_sensor = str(component.get("kind") or "").lower() == "sensor"
        add_node(SystemModelNode(
            node_id=component_node,
            kind=SystemNodeKind.SENSOR if is_sensor else SystemNodeKind.COMPONENT,
            properties={
                "component_id": component_id,
                "component_type": component.get("component_type"),
                "unit": component.get("unit"),
                "unit_status": component.get("unit_status"),
                "data_type": component.get("data_type"),
                "nominal_modes": list(component.get("nominal_modes") or []),
            },
        ))
        add_edge(system_node, SystemEdgeKind.CONTAINS, component_node)
        type_node = f"component_type:{component.get('component_type')}"
        if type_node in nodes:
            add_edge(component_node, SystemEdgeKind.INSTANCE_OF, type_node)
        for mode in sorted(set(component.get("fault_modes") or [])):
            fault_id = f"{component_id}:{mode}"
            fault_node = f"fault:{fault_id}"
            add_node(SystemModelNode(
                node_id=fault_node,
                kind=SystemNodeKind.FAULT_MODE,
                properties={"fault_id": fault_id, "component_id": component_id, "mode": mode},
            ))
            add_edge(component_node, SystemEdgeKind.HAS_FAULT_MODE, fault_node)

    for connection in catalog.get("connections") or []:
        c1 = str(connection.get("c1") or "")
        c2 = str(connection.get("c2") or "")
        if c1 in component_nodes and c2 in component_nodes:
            a, b = sorted((component_nodes[c1], component_nodes[c2]))
            add_edge(a, SystemEdgeKind.CONNECTED_TO, b)

    command_targets: dict[str, set[str]] = defaultdict(set)
    for binding in catalog.get("command_bindings") or []:
        command_id = str(binding.get("command_id") or "")
        component_id = str(binding.get("component_id") or "")
        if not command_id or component_id not in component_nodes:
            continue
        command_targets[command_id].add(component_id)
        command_node = f"command:{command_id}"
        add_node(SystemModelNode(
            node_id=command_node,
            kind=SystemNodeKind.COMMAND,
            properties={
                "command_id": command_id,
                "data_type": binding.get("data_type"),
                "binding_source": binding.get("binding_source"),
            },
        ))
        add_edge(command_node, SystemEdgeKind.COMMANDS, component_nodes[component_id])
    ambiguous = sorted(command_id for command_id, targets in command_targets.items() if len(targets) > 1)
    if ambiguous:
        raise ValueError(f"ambiguous command-instance bindings: {ambiguous}")

    node_models = tuple(sorted(nodes.values(), key=lambda item: item.node_id))
    edge_models = tuple(
        edges[key]
        for key in sorted(edges, key=lambda item: (item[0], item[1], item[2], item[3]))
    )
    payload = {
        "system_id": system_id,
        "source_artifact_sha256": source_sha,
        "nodes": [item.model_dump(mode="json") for item in node_models],
        "edges": [item.model_dump(mode="json") for item in edge_models],
    }
    return SystemModelGraph(
        system_id=system_id,
        source_artifact_sha256=source_sha,
        nodes=node_models,
        edges=edge_models,
        graph_sha256=_digest(payload),
    )


def structural_conflicts_from_mismatches(
    graph: SystemModelGraph,
    mismatches: list[ObservationMismatch],
    *,
    max_topology_hops: int = 1,
) -> tuple[ModelConflict, ...]:
    """Generate bounded *structural* conflicts around admitted inconsistencies.

    This is a conservative candidate-generation bridge, not yet a full hybrid
    physics consistency checker.  Each mismatch maps to the failure modes of the
    sensor component, nearby connected components, and an explicitly commanded
    component when present.  Later simulation/model checkers may replace these
    structural conflicts with stronger physical conflicts without changing the
    hitting-set engine.
    """
    if max_topology_hops < 0:
        raise ValueError("max_topology_hops must be nonnegative")
    known_nodes = {item.node_id for item in graph.nodes}
    adjacency = graph.component_neighbors()
    result: list[ModelConflict] = []
    for mismatch in mismatches:
        start = f"component:{mismatch.sensor_component_id}"
        if start not in known_nodes:
            raise ValueError(f"unknown sensor component: {mismatch.sensor_component_id}")
        candidate_components: set[str] = {start}
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_topology_hops:
                continue
            for neighbor in sorted(adjacency.get(node, set())):
                if neighbor not in candidate_components:
                    candidate_components.add(neighbor)
                    queue.append((neighbor, depth + 1))
        if mismatch.command_id:
            target = graph.command_target(mismatch.command_id)
            if target is None:
                raise ValueError(f"unbound command in mismatch: {mismatch.command_id}")
            candidate_components.add(target)
        fault_ids = sorted({
            fault
            for component in candidate_components
            for fault in graph.fault_ids_for_component(component)
        })
        if not fault_ids:
            raise ValueError(f"no admitted fault modes can explain mismatch {mismatch.mismatch_id}")
        result.append(ModelConflict(
            conflict_id=mismatch.mismatch_id,
            fault_ids=tuple(fault_ids),
            source_ids=mismatch.source_ids,
            explanation=mismatch.explanation or (
                "Structural neighborhood conflict generated from an admitted inconsistent observation."
            ),
        ))
    return tuple(result)


__all__ = [
    "ObservationMismatch",
    "SystemEdgeKind",
    "SystemModelEdge",
    "SystemModelGraph",
    "SystemModelNode",
    "SystemNodeKind",
    "build_system_model_graph",
    "structural_conflicts_from_mismatches",
]
