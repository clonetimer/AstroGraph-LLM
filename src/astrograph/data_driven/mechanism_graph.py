"""Load reviewed, dataset-bound graph priors for research challengers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_mechanism_graph(
    path: Path,
    graph_id: str,
    expected_nodes: list[str],
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Resolve named edges without learning topology from evaluation data."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    graph = payload["graphs"][graph_id]
    nodes = list(graph["nodes"])
    if nodes != expected_nodes:
        raise ValueError(
            f"mechanism graph nodes do not match runtime hierarchy: {nodes} != {expected_nodes}"
        )
    index = {name: position for position, name in enumerate(nodes)}
    directed: list[tuple[int, int]] = []
    for edge in graph["edges"]:
        source = index[edge["source"]]
        target = index[edge["target"]]
        directed.append((source, target))
        if edge.get("bidirectional", False):
            directed.append((target, source))
    if len(set(directed)) != len(directed):
        raise ValueError(f"mechanism graph contains duplicate directed edges: {graph_id}")
    receipt = {
        "schema_version": payload["schema_version"],
        "graph_id": graph_id,
        "nodes": nodes,
        "directed_edges": [list(edge) for edge in directed],
        "edge_claims": graph["edges"],
        "claim_boundary": graph["claim_boundary"],
        "topology_fit_scope": "configuration_only; no Train/Validation/Test values used",
    }
    return directed, receipt


def load_hierarchical_mechanism_graph(
    path: Path,
    graph_id: str,
    selected_channels: list[str],
    component_names: list[str],
) -> dict[str, Any]:
    """Load a dataset-bound component/subsystem/spacecraft graph contract.

    Routing is based only on reviewed channel names. A channel may feed more
    than one component because a physical measurement can be evidence for
    coupled assets; no telemetry values or evaluation labels fit the topology.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    graph = payload["graphs"][graph_id]
    configured_components = list(graph["components"])
    if set(configured_components) != set(component_names):
        raise ValueError(
            "hierarchical graph components do not match runtime hierarchy: "
            f"{configured_components} != {component_names}"
        )
    subsystems = list(graph["subsystems"])
    subsystem_index = {name: index for index, name in enumerate(subsystems)}

    def route(patterns: list[str]) -> list[int]:
        return [
            index for index, channel in enumerate(selected_channels)
            if any(re.match(pattern, channel) for pattern in patterns)
        ]

    component_groups = [
        route(list(graph["component_channel_patterns"][name])) for name in component_names
    ]
    subsystem_groups = [
        route(list(graph["subsystem_channel_patterns"][name])) for name in subsystems
    ]
    empty_components = [
        name for name, indices in zip(component_names, component_groups, strict=True) if not indices
    ]
    empty_subsystems = [
        name for name, indices in zip(subsystems, subsystem_groups, strict=True) if not indices
    ]
    if empty_components or empty_subsystems:
        raise ValueError({
            "empty_component_channel_groups": empty_components,
            "empty_subsystem_channel_groups": empty_subsystems,
        })
    component_subsystem = [
        subsystem_index[graph["component_to_subsystem"][name]] for name in component_names
    ]
    directed: list[tuple[int, int]] = []
    for edge in graph["subsystem_edges"]:
        source = subsystem_index[edge["source"]]
        target = subsystem_index[edge["target"]]
        directed.append((source, target))
        if edge.get("bidirectional", False):
            directed.append((target, source))
    if len(set(directed)) != len(directed):
        raise ValueError(f"hierarchical graph has duplicate subsystem edges: {graph_id}")
    return {
        "schema_version": payload["schema_version"],
        "graph_id": graph_id,
        "components": component_names,
        "subsystems": subsystems,
        "spacecraft": graph["spacecraft"],
        "component_groups": component_groups,
        "subsystem_groups": subsystem_groups,
        "component_subsystem": component_subsystem,
        "subsystem_edges": directed,
        "channel_routing": {
            "components": {
                name: [selected_channels[index] for index in indices]
                for name, indices in zip(component_names, component_groups, strict=True)
            },
            "subsystems": {
                name: [selected_channels[index] for index in indices]
                for name, indices in zip(subsystems, subsystem_groups, strict=True)
            },
        },
        "edge_claims": graph["subsystem_edges"],
        "claim_boundary": graph["claim_boundary"],
        "topology_fit_scope": "configuration_only; no Train/Validation/Test values used",
    }


def load_progressive_hierarchy(
    path: Path,
    graph_id: str,
    selected_channels: list[str],
    component_names: list[str],
) -> dict[str, Any]:
    """Resolve a reviewed parameter-to-spacecraft hierarchy by channel name."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    graph = payload["graphs"][graph_id]
    if list(graph["components"]) != component_names:
        raise ValueError("progressive hierarchy components do not match runtime order")

    def route(patterns: list[str]) -> list[int]:
        return [
            index for index, channel in enumerate(selected_channels)
            if any(re.match(pattern, channel) for pattern in patterns)
        ]

    parameters = list(graph["parameters"])
    parts = list(graph["parts"])
    modules = list(graph["functional_modules"])
    subsystems = list(graph["subsystems"])
    parameter_groups = [route(graph["parameter_channel_patterns"][name]) for name in parameters]
    empty = [name for name, indices in zip(parameters, parameter_groups, strict=True) if not indices]
    if empty:
        raise ValueError(f"parameter nodes have no selected telemetry: {empty}")

    def bind(children: list[str], parents: list[str], mapping: dict[str, str]) -> list[int]:
        parent_index = {name: index for index, name in enumerate(parents)}
        if set(mapping) != set(children):
            raise ValueError("progressive hierarchy mapping does not cover every child")
        return [parent_index[mapping[name]] for name in children]

    result = {
        "schema_version": payload["schema_version"],
        "graph_id": graph_id,
        "parameters": parameters,
        "parts": parts,
        "components": component_names,
        "functional_modules": modules,
        "subsystems": subsystems,
        "spacecraft": graph["spacecraft"],
        "parameter_groups": parameter_groups,
        "parameter_part": bind(parameters, parts, graph["parameter_to_part"]),
        "part_component": bind(parts, component_names, graph["part_to_component"]),
        "component_module": bind(component_names, modules, graph["component_to_functional_module"]),
        "module_subsystem": bind(modules, subsystems, graph["functional_module_to_subsystem"]),
        "channel_routing": {
            name: [selected_channels[index] for index in indices]
            for name, indices in zip(parameters, parameter_groups, strict=True)
        },
        "claim_boundary": graph["claim_boundary"],
        "topology_fit_scope": "configuration_only; no telemetry values or labels used",
    }
    return result
