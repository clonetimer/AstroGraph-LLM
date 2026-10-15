import csv
import json
from pathlib import Path

from astrograph.agents.knowledge_curation import KnowledgeExtractionProposal
from astrograph.knowledge.document_graph_pack import build_document_pack_candidate_graphs
from astrograph.knowledge.document_graph_review import build_document_graph_review_handoff
from astrograph.knowledge.document_graph_review_decisions import (
    freeze_document_graph_review_decisions,
)
from astrograph.retrieval.confusion_document_builder import build_fault_confusion_document_pack


class _Backend:
    model_id = "review-fixture"
    model_version = "1"

    def generate_structured(self, *, agent_id, output_model, payload):
        fragment = payload["source_profiles"][0]["fragments"][0]["fragment_id"]
        return KnowledgeExtractionProposal(
            source_profile_sha256=payload["source_profile_sha256"],
            nodes=[{"node_id": "candidate:test:mechanism", "node_type": "FailureMechanism",
                    "canonical_name": "Fixture mechanism", "source_fragment_ids": [fragment],
                    "confidence": 0.9}], proposer_id="fixture", proposer_version="1",
        )


def _fixture(tmp_path: Path):
    config, facts = tmp_path / "config.json", tmp_path / "facts.json"
    config.write_text(json.dumps({"schema_version": "astrograph-fault-confusion-groups.v1",
        "configuration_id": "fixture", "review_state": "HUMAN_REVIEW_REQUIRED",
        "groups": [{"group_id": "g", "dataset_id": "d", "fault_classes": ["a", "b"],
                    "discriminator_questions": ["Which boundary?"]}], "claim_boundary": "candidate"}))
    facts.write_text(json.dumps([{"dataset_id": "d", "fault_class_id": fault,
        "component_id": "c", "mechanism_id": "m", "generator_condition_summaries": [],
        "evidence_definition_ids": []} for fault in ("a", "b")]))
    corpus, graphs, review = tmp_path / "corpus", tmp_path / "graphs", tmp_path / "review"
    build_fault_confusion_document_pack(config_path=config, fault_facts_path=facts, output_root=corpus)
    build_document_pack_candidate_graphs(manifest_path=corpus / "manifest.json", corpus_root=corpus,
        output_root=graphs, backend=_Backend(), excluded_review_refreeze_sha256="7" * 64)
    build_document_graph_review_handoff(primary_root=graphs, supplemental_roots=[], output_dir=review)
    return graphs, review / "human_review_decisions.csv"


def _edit(path: Path, callback) -> None:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream); fields = list(reader.fieldnames or []); rows = list(reader)
    callback(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def test_completed_review_freezes_approved_subset_without_publication(tmp_path: Path) -> None:
    graphs, sheet = _fixture(tmp_path)
    def decide(rows):
        for index, row in enumerate(rows):
            row.update({"decision": "APPROVE" if index == 0 else "REJECT",
                        "approved_canonical_id": "concept:fixture" if index == 0 else "",
                        "reviewer": "human-reviewer", "review_comment": "checked against source"})
    _edit(sheet, decide)
    receipt = freeze_document_graph_review_decisions(
        review_sheet_path=sheet, graph_roots=[graphs], output_path=tmp_path / "freeze.json"
    )
    assert receipt["status"] == "READY_FOR_KNOWLEDGE_PATCH_PROPOSAL"
    assert len(receipt["approved_items"]) == 1
    assert receipt["publication_allowed"] is False
    assert receipt["active_graph_modified"] is False


def test_review_freeze_rejects_machine_field_tampering(tmp_path: Path) -> None:
    graphs, sheet = _fixture(tmp_path)
    def decide(rows):
        for row in rows:
            row.update({"decision": "REJECT", "reviewer": "human-reviewer",
                        "review_comment": "not accepted"})
        rows[0]["canonical_name"] = "tampered"
    _edit(sheet, decide)
    receipt = freeze_document_graph_review_decisions(
        review_sheet_path=sheet, graph_roots=[graphs], output_path=tmp_path / "freeze.json"
    )
    assert receipt["status"] == "BLOCKED"
    assert "MACHINE_FIELD_TAMPERED" in {item["code"] for item in receipt["blockers"]}
    assert receipt["approved_items"] == []


def test_review_freeze_cannot_relabel_a_blocked_batch_as_ready(tmp_path: Path) -> None:
    graphs, sheet = _fixture(tmp_path)
    def decide(rows):
        for row in rows:
            row.update({"decision": "REJECT", "reviewer": "human-reviewer",
                        "review_comment": "not accepted"})
        rows[0]["batch_status"] = "BLOCKED"
    _edit(sheet, decide)
    receipt = freeze_document_graph_review_decisions(
        review_sheet_path=sheet, graph_roots=[graphs], output_path=tmp_path / "freeze.json"
    )
    assert receipt["status"] == "BLOCKED"
    assert "MACHINE_FIELD_TAMPERED" in {item["code"] for item in receipt["blockers"]}
