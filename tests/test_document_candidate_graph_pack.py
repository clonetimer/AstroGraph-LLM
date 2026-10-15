import hashlib
import json
from pathlib import Path

import pytest

from astrograph.agents.knowledge_curation import KnowledgeExtractionProposal
from astrograph.knowledge.document_graph_pack import build_document_pack_candidate_graphs
from astrograph.retrieval.confusion_document_builder import build_fault_confusion_document_pack


class _Backend:
    model_id = "pack-fixture"
    model_version = "1"

    def generate_structured(self, *, agent_id, output_model, payload):
        fragment = payload["source_profiles"][0]["fragments"][0]["fragment_id"]
        return KnowledgeExtractionProposal(
            source_profile_sha256=payload["source_profile_sha256"],
            nodes=[{
                "node_id": "candidate:test:mechanism",
                "node_type": "FailureMechanism",
                "canonical_name": "Fixture mechanism",
                "source_fragment_ids": [fragment],
                "confidence": 0.9,
            }],
            proposer_id="fixture",
            proposer_version="1",
        )


def _pack(tmp_path: Path) -> Path:
    config = tmp_path / "config.json"
    facts = tmp_path / "facts.json"
    config.write_text(json.dumps({
        "schema_version": "astrograph-fault-confusion-groups.v1",
        "configuration_id": "fixture-r001",
        "review_state": "HUMAN_REVIEW_REQUIRED",
        "groups": [{
            "group_id": "g1", "dataset_id": "fixture.v1",
            "fault_classes": ["fault_a", "fault_b"],
            "discriminator_questions": ["Which boundary disagrees?"],
        }],
        "claim_boundary": "Candidate associations only.",
    }), encoding="utf-8")
    facts.write_text(json.dumps([
        {"dataset_id": "fixture.v1", "fault_class_id": fault,
         "component_id": "component", "mechanism_id": "mechanism:x",
         "generator_condition_summaries": [], "evidence_definition_ids": []}
        for fault in ("fault_a", "fault_b")
    ]), encoding="utf-8")
    root = tmp_path / "corpus"
    build_fault_confusion_document_pack(
        config_path=config, fault_facts_path=facts, output_root=root
    )
    return root


def test_pack_persists_review_only_candidate_graphs(tmp_path: Path) -> None:
    corpus = _pack(tmp_path)
    receipt = build_document_pack_candidate_graphs(
        manifest_path=corpus / "manifest.json",
        corpus_root=corpus,
        output_root=tmp_path / "graphs",
        backend=_Backend(),
        excluded_review_refreeze_sha256="6" * 64,
        max_workers=2,
    )
    assert receipt["selected_document_count"] == 2
    assert receipt["candidate_ready_count"] == 2
    assert receipt["blocked_count"] == 0
    assert receipt["publication_allowed"] is False
    assert receipt["active_graph_modified"] is False
    assert all(row["candidate_graph_built"] for row in receipt["results"])
    assert all(row["raw_extraction_proposal_sha256"] for row in receipt["results"])
    assert (tmp_path / "graphs" / "receipt.json").is_file()


def test_pack_detects_tampering_before_creating_output(tmp_path: Path) -> None:
    corpus = _pack(tmp_path)
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    source = corpus / manifest["documents"][0]["path"]
    source.write_text(source.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
    output = tmp_path / "graphs"
    with pytest.raises(ValueError, match="source hash mismatch"):
        build_document_pack_candidate_graphs(
            manifest_path=corpus / "manifest.json",
            corpus_root=corpus,
            output_root=output,
            backend=_Backend(),
            excluded_review_refreeze_sha256="6" * 64,
        )
    assert not output.exists()
