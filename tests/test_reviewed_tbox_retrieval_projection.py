import json
from pathlib import Path

import pytest

from astrograph.knowledge.tbox_retrieval_projection import (
    build_reviewed_tbox_retrieval_projection,
)
from astrograph.retrieval.models import DiagnosticRetrievalQuery, PreReasoningRetrievalRequest
from astrograph.retrieval.service import PreReasoningRetrievalService
from astrograph.retrieval.v3_tbox_adapter import load_reviewed_v3_tbox_retrieval_assets


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "governance/evidence/ai_native/reusable_kg_construction_20260901/candidate_graph.json"
APPROVAL = ROOT / "governance/evidence/ai_native/tbox_vocabulary_approval_20260902/approval.json"


def _build(tmp_path: Path):
    receipt = build_reviewed_tbox_retrieval_projection(
        source_graph_path=SOURCE, approval_path=APPROVAL, output_dir=tmp_path
    )
    concepts, relations, audit = load_reviewed_v3_tbox_retrieval_assets(
        graph_path=tmp_path / "reviewed_tbox_retrieval_graph.json",
        projection_receipt_path=tmp_path / "build_receipt.json",
    )
    return receipt, concepts, relations, audit


def test_owner_approved_tbox_projects_to_retrieval_only_oag_assets(tmp_path: Path) -> None:
    receipt, concepts, relations, audit = _build(tmp_path)
    assert receipt["status"] == "RETRIEVAL_ONLY_READY"
    assert receipt["failure_mechanism_count"] == 12
    assert receipt["effect_pattern_count"] == 10
    assert receipt["association_edge_count"] == 22
    assert receipt["all_edges_human_reviewed"] is True
    assert receipt["diagnostic_evidence_allowed"] is False
    assert receipt["runtime_vote_allowed"] is False
    assert len(concepts) == 22
    assert len(relations) == 22
    assert audit["retrieval_only"] is True

    package = PreReasoningRetrievalService().retrieve(
        PreReasoningRetrievalRequest(
            query=DiagnosticRetrievalQuery(
                query_id="q:tbox",
                episode_id="episode:test",
                text="actuation authority loss",
                seed_concept_ids=["mechanism:actuation-authority-loss"],
                maximum_ontology_hops=1,
            ),
            concepts=concepts,
            relations=relations,
        )
    )
    assert package.ontology_paths
    assert all(path.hops[0].relation_type == "MANIFESTS_AS" for path in package.ontology_paths)
    assert all(path.hops[0].source_fragment_ids for path in package.ontology_paths)
    assert all(
        path.hops[0].association_semantics == "TBOX_ASSOCIATION_ONLY"
        for path in package.ontology_paths
    )
    assert all(path.hops[0].retrieval_only is True for path in package.ontology_paths)
    assert all(path.hops[0].runtime_vote_allowed is False for path in package.ontology_paths)
    assert package.active_graph_modified is False


def test_projection_receipt_tampering_is_rejected(tmp_path: Path) -> None:
    _build(tmp_path)
    path = tmp_path / "build_receipt.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["runtime_vote_allowed"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="receipt hash mismatch"):
        load_reviewed_v3_tbox_retrieval_assets(
            graph_path=tmp_path / "reviewed_tbox_retrieval_graph.json",
            projection_receipt_path=path,
        )
