import json
from pathlib import Path

from astrograph.agents.knowledge_curation import KnowledgeExtractionProposal
from astrograph.knowledge.document_graph_pack import build_document_pack_candidate_graphs
from astrograph.knowledge.document_graph_review import build_document_graph_review_handoff
from astrograph.retrieval.confusion_document_builder import build_fault_confusion_document_pack


class _Backend:
    model_id = "review-fixture"
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
            proposer_id="fixture", proposer_version="1",
        )


def test_review_handoff_is_blank_source_bound_and_complete(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    facts = tmp_path / "facts.json"
    config.write_text(json.dumps({
        "schema_version": "astrograph-fault-confusion-groups.v1",
        "configuration_id": "fixture", "review_state": "HUMAN_REVIEW_REQUIRED",
        "groups": [{"group_id": "g", "dataset_id": "d", "fault_classes": ["a", "b"],
                    "discriminator_questions": ["Which boundary?"]}],
        "claim_boundary": "candidate only",
    }), encoding="utf-8")
    facts.write_text(json.dumps([
        {"dataset_id": "d", "fault_class_id": fault, "component_id": "c",
         "mechanism_id": "m", "generator_condition_summaries": [],
         "evidence_definition_ids": []} for fault in ("a", "b")
    ]), encoding="utf-8")
    corpus = tmp_path / "corpus"
    build_fault_confusion_document_pack(config_path=config, fault_facts_path=facts, output_root=corpus)
    graphs = tmp_path / "graphs"
    build_document_pack_candidate_graphs(
        manifest_path=corpus / "manifest.json", corpus_root=corpus,
        output_root=graphs, backend=_Backend(), excluded_review_refreeze_sha256="7" * 64,
    )
    output = tmp_path / "review"
    receipt = build_document_graph_review_handoff(
        primary_root=graphs, supplemental_roots=[], output_dir=output
    )
    assert receipt["source_document_count"] == 2
    assert receipt["review_item_count"] == 2
    assert receipt["node_review_item_count"] == 2
    assert receipt["relation_review_item_count"] == 0
    assert receipt["human_decision_count"] == 0
    assert receipt["unresolved_source_documents"] == []
    import csv
    with (output / "human_review_decisions.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["source_document_id"] for row in rows)
    assert all(row["source_document_sha256"] for row in rows)
    assert all(not row[field] for row in rows for field in (
        "decision", "approved_canonical_id", "reviewer", "review_comment"
    ))
