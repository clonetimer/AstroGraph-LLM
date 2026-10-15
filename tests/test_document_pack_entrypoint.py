import hashlib
import json
import sys
from pathlib import Path

import pytest

from astrograph.cli import main
from astrograph.agents.knowledge_curation import KnowledgeExtractionProposal
from astrograph.knowledge.document_pack_entrypoint import (
    build_document_pack_candidate_packages,
)
from astrograph.retrieval.confusion_document_builder import (
    build_fault_confusion_document_pack,
)


class _EmptyBackend:
    model_id = "empty-pack-fixture"
    model_version = "1"

    def generate_structured(self, *, agent_id, output_model, payload):
        return KnowledgeExtractionProposal(
            source_profile_sha256=payload["source_profile_sha256"],
            proposer_id=self.model_id,
            proposer_version=self.model_version,
        )


def _inputs(tmp_path: Path) -> tuple[Path, Path, str]:
    config = tmp_path / "config.json"
    facts = tmp_path / "facts.json"
    config.write_text(json.dumps({
        "schema_version": "astrograph-fault-confusion-groups.v1",
        "configuration_id": "stable-pack-fixture-r001",
        "review_state": "HUMAN_REVIEW_REQUIRED",
        "groups": [{
            "group_id": "g1",
            "dataset_id": "fixture.v1",
            "fault_classes": ["fault_a", "fault_b"],
            "discriminator_questions": ["Which boundary disagrees?"],
        }],
        "claim_boundary": "Candidate associations only.",
    }), encoding="utf-8")
    facts.write_text(json.dumps([
        {
            "dataset_id": "fixture.v1",
            "fault_class_id": fault,
            "component_id": f"component_{fault}",
            "mechanism_id": f"mechanism:{fault}",
            "generator_condition_summaries": [],
            "evidence_definition_ids": [f"ED-{fault.upper()}"],
        }
        for fault in ("fault_a", "fault_b")
    ]), encoding="utf-8")
    corpus = tmp_path / "corpus"
    build_fault_confusion_document_pack(
        config_path=config, fault_facts_path=facts, output_root=corpus
    )
    refreeze = tmp_path / "refreeze.json"
    refreeze.write_text(json.dumps({
        "temporary_frozen": True,
        "publication_allowed": False,
        "refreeze_sha256": "6" * 64,
    }), encoding="utf-8")
    return corpus, refreeze, hashlib.sha256(refreeze.read_bytes()).hexdigest()


def test_stable_pack_composes_ingestion_and_hybrid_candidate_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus, refreeze, refreeze_sha = _inputs(tmp_path)
    output = tmp_path / "stable-pack"
    receipt = build_document_pack_candidate_packages(
        manifest_path=corpus / "manifest.json",
        corpus_root=corpus,
        output_root=output,
        review_refreeze_path=refreeze,
        expected_review_refreeze_file_sha256=refreeze_sha,
        backend_factory=_EmptyBackend,
    )
    assert receipt["status"] == "ALL_CANDIDATES_READY_FOR_HUMAN_REVIEW"
    assert receipt["selected_document_count"] == 2
    assert receipt["all_crosswalks_complete"] is True
    assert receipt["status_counts"] == {"CANDIDATE_READY_FOR_HUMAN_REVIEW": 2}
    assert all(item["candidate_node_count"] == 2 for item in receipt["results"])
    assert all(item["candidate_edge_count"] == 0 for item in receipt["results"])
    for fault in ("fault_a", "fault_b"):
        root = output / "fixture.v1" / fault
        assert (root / "ingestion/receipt.json").is_file()
        assert (root / "candidate/receipt.json").is_file()
    assert receipt["publication_allowed"] is False
    assert receipt["active_graph_modified"] is False

    review_root = tmp_path / "pack-review"
    monkeypatch.setattr(sys, "argv", [
        "astrograph", "prepare-document-pack-graph-review",
        "--candidate-pack", str(output), "--output", str(review_root),
    ])
    assert main() == 0
    review = json.loads(capsys.readouterr().out)
    assert review["source_document_count"] == 2
    assert review["review_item_count"] == 4
    assert review["blocked_source_document_count"] == 0

    freeze = tmp_path / "pack-review-freeze.json"
    monkeypatch.setattr(sys, "argv", [
        "astrograph", "freeze-document-pack-graph-review",
        "--candidate-pack", str(output),
        "--review-sheet", str(review_root / "human_review_decisions.csv"),
        "--output", str(freeze),
    ])
    assert main() == 2
    frozen = json.loads(capsys.readouterr().out)
    assert frozen["status"] == "BLOCKED"
    assert frozen["approved_items"] == []
    assert frozen["publication_allowed"] is False

    with pytest.raises(ValueError, match="refusing to overwrite"):
        build_document_pack_candidate_packages(
            manifest_path=corpus / "manifest.json",
            corpus_root=corpus,
            output_root=output,
            review_refreeze_path=refreeze,
            expected_review_refreeze_file_sha256=refreeze_sha,
            backend_factory=_EmptyBackend,
        )


def test_stable_pack_validates_every_selected_source_before_output(tmp_path: Path) -> None:
    corpus, refreeze, refreeze_sha = _inputs(tmp_path)
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    source = corpus / manifest["documents"][1]["path"]
    source.write_text(source.read_text(encoding="utf-8") + "tamper", encoding="utf-8")
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="source hash mismatch"):
        build_document_pack_candidate_packages(
            manifest_path=corpus / "manifest.json",
            corpus_root=corpus,
            output_root=output,
            review_refreeze_path=refreeze,
            expected_review_refreeze_file_sha256=refreeze_sha,
            backend_factory=_EmptyBackend,
        )
    assert not output.exists()
