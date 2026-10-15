import json
from pathlib import Path

import pytest

from astrograph.knowledge import approve_cross_domain_tbox_vocabulary
from astrograph.knowledge.schema_v3 import KnowledgeGraphV3


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "governance/evidence/ai_native/reusable_kg_construction_20260901/candidate_graph.json"


def test_project_owner_decision_is_tbox_only_and_does_not_unfreeze_other_review():
    graph = KnowledgeGraphV3.model_validate_json(GRAPH.read_text(encoding="utf-8"))
    receipt = approve_cross_domain_tbox_vocabulary(
        graph, decision_source="project-owner-directive:2026-09-02"
    )
    assert len(receipt.approved_failure_mechanism_ids) == 12
    assert len(receipt.approved_effect_pattern_ids) == 10
    assert len(receipt.approved_tbox_association_edge_ids) == 22
    assert receipt.association_semantics == "TBOX_ASSOCIATION_ONLY"
    assert receipt.episode_evidence_approved is False
    assert receipt.actual_causal_propagation_approved is False
    assert receipt.runtime_diagnosis_or_voting_allowed is False
    assert receipt.active_graph_modified is False
    assert receipt.dataset_mapping_review_unfrozen is False


def test_tbox_approval_fails_closed_if_inventory_drifts():
    graph = KnowledgeGraphV3.model_validate_json(GRAPH.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="exactly 12 mechanisms and 10 effects"):
        approve_cross_domain_tbox_vocabulary(
            graph.model_copy(update={"nodes": [node for node in graph.nodes if node.node_id != "mechanism:additive-bias-shift"]}),
            decision_source="test",
        )
