from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.data_driven import compare_adcs_deep_encoders as adcs_deep
from scripts.data_driven import compare_phm_causal_windows as phm_deep
from scripts.data_driven.run_hierarchical_group_cv import rotating_split
from scripts.integration import run_adcs_v522_regression as adcs
from astrograph.data_driven.mechanism_graph import (
    load_hierarchical_mechanism_graph,
    load_mechanism_graph,
)


ROOT = Path(__file__).resolve().parents[1]


def test_rotating_split_is_three_train_one_validation_one_test() -> None:
    for test_fold in range(5):
        assignments = [rotating_split(index, test_fold) for index in range(5)]
        assert assignments.count("train") == 3
        assert assignments.count("validation") == 1
        assert assignments.count("test") == 1


def test_adcs_subsystem_tcn_preserves_three_stage_shapes() -> None:
    class_names = sorted(adcs.FAULT_TO_COMPONENT)
    components = list(adcs.COMPONENT_NAMES)
    groups, class_component = adcs_deep.adcs_subsystem_groups(
        adcs.FEATURE_COLUMNS, components, class_names
    )
    model = adcs_deep.make_model(
        "subsystem_tcn", len(adcs.FEATURE_COLUMNS), len(components), len(class_names),
        subsystem_groups=groups, class_component=class_component,
    )
    outputs = model(torch.zeros(3, adcs.WINDOW, len(adcs.FEATURE_COLUMNS)))
    assert [tuple(value.shape) for value in outputs] == [(3, 2), (3, 5), (3, 14)]


def test_phm_subsystem_tcn_uses_all_six_physical_domains() -> None:
    hierarchy = json.loads(phm_deep.HIERARCHY.read_text(encoding="utf-8"))
    selected = [
        "adcs.pointing_error_deg", "eps.battery_soc", "payload.active",
        "comm.downlink_bps", "data.storage_bits", "thermal.payload_temp_k",
        "propulsion.fuel_mass_kg",
    ]
    class_names = sorted(hierarchy["fault_to_component"])
    components = sorted(set(hierarchy["fault_to_component"].values()))
    groups, component_group, class_component, names = phm_deep.phm_subsystem_groups(
        selected, components, class_names, hierarchy["fault_to_component"]
    )
    assert names == ["adcs", "eps", "payload", "comm_data", "thermal", "propulsion"]
    model = phm_deep.make_model(
        "subsystem_tcn", len(selected), len(components), len(class_names),
        subsystem_groups=groups, component_group=component_group,
        class_component=class_component,
    )
    outputs = model(torch.zeros(2, phm_deep.WINDOW, len(selected)))
    assert [tuple(value.shape) for value in outputs] == [(2, 2), (2, 13), (2, 17)]


def test_soft_hierarchy_preserves_uncertain_classes_and_normalizes() -> None:
    probabilities = np.asarray([[0.40, 0.35, 0.25]])
    components = np.asarray([[0.55, 0.45]])
    conditioned, predicted = adcs._conditioned_class_probabilities(
        probabilities, components, {0: np.asarray([0, 1]), 1: np.asarray([2])}, mode="soft"
    )
    assert predicted.tolist() == [0]
    assert conditioned[0, 2] > 0.0
    assert np.allclose(conditioned.sum(1), 1.0)


def test_reviewed_mechanism_graphs_match_runtime_node_orders() -> None:
    config = ROOT / "configs/data_driven/mechanism_graphs.json"
    adcs_edges, adcs_receipt = load_mechanism_graph(
        config, "adcs_v5_2_2", list(adcs.COMPONENT_NAMES)
    )
    assert len(adcs_edges) == 8
    assert adcs_receipt["topology_fit_scope"].startswith("configuration_only")

    phm_nodes = ["adcs", "eps", "payload", "comm_data", "thermal", "propulsion"]
    phm_edges, phm_receipt = load_mechanism_graph(config, "phm_v6_0", phm_nodes)
    assert len(phm_edges) == 14
    assert phm_receipt["nodes"] == phm_nodes


def test_adcs_gat_detection_path_matches_seeded_residual_tcn() -> None:
    class_names = sorted(adcs.FAULT_TO_COMPONENT)
    components = list(adcs.COMPONENT_NAMES)
    groups, class_component = adcs_deep.adcs_subsystem_groups(
        adcs.FEATURE_COLUMNS, components, class_names
    )
    edges, _ = load_mechanism_graph(
        ROOT / "configs/data_driven/mechanism_graphs.json",
        "adcs_v5_2_2", components,
    )
    torch.manual_seed(71)
    baseline = adcs_deep.make_model(
        "residual_tcn", len(adcs.FEATURE_COLUMNS), len(components), len(class_names)
    )
    torch.manual_seed(71)
    candidate = adcs_deep.make_model(
        "mechanism_gated_gat_tcn", len(adcs.FEATURE_COLUMNS),
        len(components), len(class_names), subsystem_groups=groups,
        component_group=list(range(len(components))), class_component=class_component,
        graph_edges=edges,
    )
    values = torch.randn(3, adcs.WINDOW, len(adcs.FEATURE_COLUMNS))
    baseline.train()
    candidate.train()
    torch.manual_seed(99)
    baseline_detection = baseline(values)[0]
    torch.manual_seed(99)
    candidate_detection = candidate(values)[0]
    assert torch.equal(baseline_detection, candidate_detection)


def test_phm_gat_detection_path_matches_seeded_residual_tcn() -> None:
    hierarchy = json.loads(phm_deep.HIERARCHY.read_text(encoding="utf-8"))
    selected = [
        "adcs.pointing_error_deg", "eps.battery_soc", "payload.active",
        "comm.downlink_bps", "data.storage_bits", "thermal.payload_temp_k",
        "propulsion.fuel_mass_kg",
    ]
    class_names = sorted(hierarchy["fault_to_component"])
    components = sorted(set(hierarchy["fault_to_component"].values()))
    groups, component_group, class_component, nodes = phm_deep.phm_subsystem_groups(
        selected, components, class_names, hierarchy["fault_to_component"]
    )
    edges, _ = load_mechanism_graph(
        ROOT / "configs/data_driven/mechanism_graphs.json", "phm_v6_0", nodes
    )
    torch.manual_seed(73)
    baseline = phm_deep.make_model(
        "residual_tcn", len(selected), len(components), len(class_names)
    )
    torch.manual_seed(73)
    candidate = phm_deep.make_model(
        "mechanism_gated_gat_tcn", len(selected), len(components), len(class_names),
        subsystem_groups=groups, component_group=component_group,
        class_component=class_component, graph_edges=edges,
    )
    values = torch.randn(2, phm_deep.WINDOW, len(selected))
    baseline.train()
    candidate.train()
    torch.manual_seed(101)
    baseline_detection = baseline(values)[0]
    torch.manual_seed(101)
    candidate_detection = candidate(values)[0]
    assert torch.equal(baseline_detection, candidate_detection)


def test_phm_hierarchical_graph_routes_components_subsystems_and_spacecraft() -> None:
    hierarchy = json.loads(phm_deep.HIERARCHY.read_text(encoding="utf-8"))
    components = sorted(set(hierarchy["fault_to_component"].values()))
    selected = [
        "adcs.pointing_error_deg",
        "eps.battery_soc", "eps.bus_load_power_w", "eps.pdu.payload_enabled",
        "eps.solar_array_power_w", "payload.active", "comm.downlink_rate_bps",
        "data.storage_bits", "thermal.heater_actual_on", "thermal.payload_temp_k",
        "thermal.radiator_actual_on", "propulsion.fuel_mass_kg",
        "propulsion.thrust_execution_residual_n",
    ]
    contract = load_hierarchical_mechanism_graph(
        ROOT / "configs/data_driven/hierarchical_mechanism_graphs.json",
        "phm_v6_0", selected, components,
    )
    assert len(contract["component_groups"]) == 13
    assert len(contract["subsystem_groups"]) == 6
    assert contract["spacecraft"] == "whole_spacecraft"
    assert contract["subsystems"][contract["component_subsystem"][
        components.index("battery")
    ]] == "eps"


def test_phm_hierarchical_gat_preserves_three_task_contract() -> None:
    hierarchy = json.loads(phm_deep.HIERARCHY.read_text(encoding="utf-8"))
    class_names = sorted(hierarchy["fault_to_component"])
    components = sorted(set(hierarchy["fault_to_component"].values()))
    selected = [
        "adcs.pointing_error_deg",
        "eps.battery_soc", "eps.bus_load_power_w", "eps.pdu.payload_enabled",
        "eps.solar_array_power_w", "payload.active", "comm.downlink_rate_bps",
        "data.storage_bits", "thermal.heater_actual_on", "thermal.payload_temp_k",
        "thermal.radiator.actual_on", "propulsion.fuel_mass_kg",
        "propulsion.thrust_execution_residual_n",
    ]
    # Use the exact generator field spelling covered by the routing contract.
    selected[10] = "thermal.radiator_actual_on"
    contract = load_hierarchical_mechanism_graph(
        ROOT / "configs/data_driven/hierarchical_mechanism_graphs.json",
        "phm_v6_0", selected, components,
    )
    component_index = {name: index for index, name in enumerate(components)}
    class_component = [component_index[hierarchy["fault_to_component"][name]] for name in class_names]
    model = phm_deep.make_model(
        "hierarchical_mechanism_gat_tcn", len(selected), len(components), len(class_names),
        class_component=class_component, hierarchical_contract=contract,
    )
    outputs = model(torch.randn(2, phm_deep.WINDOW, len(selected)))
    assert [tuple(value.shape) for value in outputs] == [(2, 2), (2, 13), (2, 17)]
    assert tuple(model.last_subsystem_logits.shape) == (2, 6)
    assert model.graph_diagnostics()["hierarchy_levels"] == [
        "component", "subsystem", "whole_spacecraft",
    ]
