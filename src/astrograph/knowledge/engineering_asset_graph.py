"""Deterministic graph construction from engineering registries.

R23 uses this builder for schema/catalog assets such as the separately supplied
Sat simulation package.  Entity linking is deliberately conservative: only
exact unique identifiers/display names are linked.  Unresolved labels remain
explicit audit items rather than being guessed by an LLM or fuzzy matcher.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class EngineeringNodeKind(str, Enum):
    COMPONENT = "COMPONENT"
    SIGNAL = "SIGNAL"
    FAULT = "FAULT"


class EngineeringEdgeKind(str, Enum):
    IDENTIFIER_HIERARCHY_PARENT = "IDENTIFIER_HIERARCHY_PARENT"
    HAS_DECLARED_FAULT = "HAS_DECLARED_FAULT"
    HAS_TELEMETRY_SIGNAL = "HAS_TELEMETRY_SIGNAL"


class EngineeringAssetNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    node_id: str = Field(min_length=1)
    kind: EngineeringNodeKind
    properties: dict[str, Any] = Field(default_factory=dict)


class EngineeringAssetEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str = Field(min_length=1)
    relation: EngineeringEdgeKind
    target_id: str = Field(min_length=1)
    binding_semantics: str = Field(min_length=1)


class EngineeringAssetGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-engineering-asset-graph.v1"
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    nodes: tuple[EngineeringAssetNode, ...]
    edges: tuple[EngineeringAssetEdge, ...]
    unresolved_fault_component_labels: tuple[str, ...] = ()
    unresolved_signal_groups: tuple[str, ...] = ()
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = (
        "Registry-derived engineering graph. Exact unique links are admitted; unresolved labels are retained. "
        "Identifier hierarchy is naming structure, not a claimed physical dependency or causal edge."
    )

    @model_validator(mode="after")
    def verify(self) -> "EngineeringAssetGraph":
        ids = [item.node_id for item in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("engineering graph node IDs must be unique")
        known = set(ids)
        for edge in self.edges:
            if edge.source_id not in known or edge.target_id not in known:
                raise ValueError("engineering graph edge references unknown node")
        payload = {
            "source_artifact_sha256": self.source_artifact_sha256,
            "nodes": [item.model_dump(mode="json") for item in self.nodes],
            "edges": [item.model_dump(mode="json") for item in self.edges],
            "unresolved_fault_component_labels": self.unresolved_fault_component_labels,
            "unresolved_signal_groups": self.unresolved_signal_groups,
        }
        if _digest(payload) != self.graph_sha256:
            raise ValueError("engineering asset graph_sha256 mismatch")
        return self


def build_engineering_asset_graph(
    *,
    source_artifact_sha256: str,
    component_registry: dict[str, Any],
    fault_catalog: list[dict[str, Any]],
    signal_registry: dict[str, Any],
) -> EngineeringAssetGraph:
    """Construct a graph with exact-only entity linking and explicit gaps."""
    nodes: dict[str, EngineeringAssetNode] = {}
    edges: dict[tuple[str, str, str], EngineeringAssetEdge] = {}

    def add_node(node: EngineeringAssetNode) -> None:
        existing = nodes.get(node.node_id)
        if existing is not None and existing != node:
            raise ValueError(f"conflicting engineering graph node: {node.node_id}")
        nodes[node.node_id] = node

    def add_edge(source: str, relation: EngineeringEdgeKind, target: str, semantics: str) -> None:
        key = (source, relation.value, target)
        edge = EngineeringAssetEdge(
            source_id=source, relation=relation, target_id=target, binding_semantics=semantics
        )
        existing = edges.get(key)
        if existing is not None and existing != edge:
            raise ValueError("conflicting engineering graph edge semantics")
        edges[key] = edge

    components = component_registry.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("component registry is empty")
    component_ids: set[str] = set()
    display_to_ids: dict[str, list[str]] = defaultdict(list)
    for row in components:
        component_id = str(row.get("id") or "").strip()
        display = str(row.get("display_name") or "").strip()
        if not component_id:
            raise ValueError("component registry row missing id")
        if component_id in component_ids:
            raise ValueError("duplicate component registry id")
        component_ids.add(component_id)
        if display:
            display_to_ids[display].append(component_id)
        add_node(EngineeringAssetNode(
            node_id=f"component:{component_id}",
            kind=EngineeringNodeKind.COMPONENT,
            properties={
                "component_id": component_id,
                "display_name": display or None,
                "component_type": row.get("type"),
            },
        ))

    # Longest exact dotted-prefix parent only. This is identifier hierarchy, not
    # physical topology, and the edge name makes that limitation explicit.
    for component_id in sorted(component_ids):
        prefixes = [candidate for candidate in component_ids if component_id.startswith(candidate + ".")]
        if prefixes:
            parent = max(prefixes, key=len)
            add_edge(
                f"component:{parent}", EngineeringEdgeKind.IDENTIFIER_HIERARCHY_PARENT,
                f"component:{component_id}", "LONGEST_EXACT_DOTTED_IDENTIFIER_PREFIX",
            )

    unresolved_faults: set[str] = set()
    for row in fault_catalog:
        domain = str(row.get("domain") or "").strip()
        component_label = str(row.get("component") or "").strip()
        fault_label = str(row.get("fault") or "").strip()
        if not component_label or not fault_label:
            raise ValueError("fault catalog row missing component/fault")
        identity = _digest({"domain": domain, "component": component_label, "fault": fault_label})[:20]
        fault_node = f"fault:{identity}"
        add_node(EngineeringAssetNode(
            node_id=fault_node,
            kind=EngineeringNodeKind.FAULT,
            properties={
                "domain": domain,
                "component_label": component_label,
                "fault_label": fault_label,
                "enable_var": row.get("enable_var"),
                "time_var": row.get("time_var"),
                "value_var": row.get("value_var"),
            },
        ))
        matches = display_to_ids.get(component_label, [])
        if len(matches) == 1:
            add_edge(
                f"component:{matches[0]}", EngineeringEdgeKind.HAS_DECLARED_FAULT, fault_node,
                "EXACT_UNIQUE_COMPONENT_DISPLAY_NAME",
            )
        else:
            unresolved_faults.add(component_label)

    unresolved_groups: set[str] = set()
    signals = signal_registry.get("signals")
    if not isinstance(signals, list):
        raise ValueError("signal registry signals must be a list")
    for row in signals:
        signal_id = str(row.get("id") or "").strip()
        if not signal_id:
            raise ValueError("signal registry row missing id")
        signal_node = f"signal:{signal_id}"
        add_node(EngineeringAssetNode(
            node_id=signal_node,
            kind=EngineeringNodeKind.SIGNAL,
            properties={
                "signal_id": signal_id,
                "column_name": row.get("column_name"),
                "unit": row.get("unit"),
                "sample_kind": row.get("sample_kind"),
                "telemetry_group": row.get("telemetry_group"),
                "online_eligible": row.get("online_eligible"),
            },
        ))
        group = str(row.get("telemetry_group") or "").strip()
        if group and group in component_ids:
            add_edge(
                f"component:{group}", EngineeringEdgeKind.HAS_TELEMETRY_SIGNAL, signal_node,
                "EXACT_TELEMETRY_GROUP_TO_COMPONENT_ID",
            )
        elif group:
            unresolved_groups.add(group)

    node_models = tuple(sorted(nodes.values(), key=lambda item: item.node_id))
    edge_models = tuple(edges[key] for key in sorted(edges))
    unresolved_fault_tuple = tuple(sorted(unresolved_faults))
    unresolved_group_tuple = tuple(sorted(unresolved_groups))
    payload = {
        "source_artifact_sha256": source_artifact_sha256,
        "nodes": [item.model_dump(mode="json") for item in node_models],
        "edges": [item.model_dump(mode="json") for item in edge_models],
        "unresolved_fault_component_labels": unresolved_fault_tuple,
        "unresolved_signal_groups": unresolved_group_tuple,
    }
    return EngineeringAssetGraph(
        source_artifact_sha256=source_artifact_sha256,
        nodes=node_models,
        edges=edge_models,
        unresolved_fault_component_labels=unresolved_fault_tuple,
        unresolved_signal_groups=unresolved_group_tuple,
        graph_sha256=_digest(payload),
    )


__all__ = [
    "EngineeringAssetEdge",
    "EngineeringAssetGraph",
    "EngineeringAssetNode",
    "EngineeringEdgeKind",
    "EngineeringNodeKind",
    "build_engineering_asset_graph",
]
