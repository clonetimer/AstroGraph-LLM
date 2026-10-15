from __future__ import annotations

import json
from pathlib import Path

from astrograph.knowledge import (
    GraphShapeValidator,
    KnowledgeRelationTypeV3,
    load_generic_fault_ontology,
    load_dataset_fault_bindings,
    project_generic_fault_ontology,
)


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = ROOT / "configs/knowledge/generic_fault_ontology.json"
ARCHETYPES = ROOT / "configs/semantics/core_component_archetypes.json"
SEMANTICS = ROOT / "configs/semantics/core_semantic_schema.json"


def test_generic_fault_ontology_projects_to_valid_v3_tbox() -> None:
    graph = project_generic_fault_ontology(ONTOLOGY, ARCHETYPES, SEMANTICS)
    report = GraphShapeValidator().validate(graph)
    assert report.status == "PASS", report.model_dump(mode="json")
    relations = {edge.relation_type for edge in graph.edges}
    assert {
        KnowledgeRelationTypeV3.SUSCEPTIBLE_TO,
        KnowledgeRelationTypeV3.MANIFESTS_AS,
        KnowledgeRelationTypeV3.INDICATED_BY,
    }.issubset(relations)


def test_tbox_excludes_dataset_bindings_and_absolute_limits() -> None:
    payload = json.loads(ONTOLOGY.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    for forbidden in (
        "whole_spacecraft", "v5_2_2", "v6_0", "sample_index", "source_field",
        "trajectory_pair_id", "activation_time_s",
    ):
        assert forbidden not in serialized
    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    assert not {name for name in keys(payload) if "threshold" in name.lower()}
    assert "UNKNOWN_OR_ABSTAIN" == payload["robustness_contract"]["missing_or_out_of_distribution_policy"]


def test_all_mechanism_effect_and_catalog_references_resolve() -> None:
    ontology = load_generic_fault_ontology(ONTOLOGY)
    graph = project_generic_fault_ontology(ONTOLOGY, ARCHETYPES, SEMANTICS)
    node_ids = {node.node_id for node in graph.nodes}
    assert all(item.mechanism_id in node_ids for item in ontology.failure_mechanisms)
    assert all(item.effect_pattern_id in node_ids for item in ontology.effect_patterns)
    assert all(edge.source_id in node_ids and edge.target_id in node_ids for edge in graph.edges)


def test_dataset_bindings_cover_current_fault_classes_without_changing_tbox() -> None:
    ontology = load_generic_fault_ontology(ONTOLOGY)
    cases = (
        (
            ROOT / "configs/datasets/adcs_v5_2_2/fault_ontology_bindings.json",
            ROOT / "configs/datasets/adcs_v5_2_2/diagnosis_hierarchy.json",
        ),
        (
            ROOT / "configs/datasets/phm_v6_0/fault_ontology_bindings.json",
            ROOT / "configs/datasets/phm_v6_0/diagnosis_hierarchy.json",
        ),
    )
    for binding_path, hierarchy_path in cases:
        package = load_dataset_fault_bindings(binding_path, ontology)
        hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))
        assert {item.fault_class_id for item in package.bindings} == set(hierarchy["fault_to_component"])
