import json
from pathlib import Path

import pytest

from astrograph.retrieval.confusion_document_builder import (
    build_fault_confusion_document_pack,
)


def _inputs(root: Path) -> tuple[Path, Path]:
    config = {
        "schema_version": "astrograph-fault-confusion-groups.v1",
        "configuration_id": "fixture-r001",
        "review_state": "HUMAN_REVIEW_REQUIRED",
        "groups": [
            {
                "group_id": "g1",
                "dataset_id": "fixture.v1",
                "fault_classes": ["fault_a", "fault_b"],
                "discriminator_questions": ["Does channel A disagree with command A?"],
            }
        ],
        "claim_boundary": "Candidate associations only.",
    }
    facts = [
        {
            "dataset_id": "fixture.v1",
            "fault_class_id": "fault_a",
            "component_id": "component_a",
            "mechanism_id": "mechanism:a",
            "generator_condition_summaries": [{"description": "A degrades."}],
            "evidence_definition_ids": ["ED-A"],
        },
        {
            "dataset_id": "fixture.v1",
            "fault_class_id": "fault_b",
            "component_id": "component_b",
            "mechanism_id": "mechanism:b",
            "generator_condition_summaries": [{"description": "B degrades."}],
            "evidence_definition_ids": [],
        },
    ]
    config_path = root / "config.json"
    facts_path = root / "facts.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    facts_path.write_text(json.dumps(facts), encoding="utf-8")
    return config_path, facts_path


def test_builds_one_dual_processed_contrast_document_per_fault(tmp_path: Path) -> None:
    config, facts = _inputs(tmp_path)
    manifest = build_fault_confusion_document_pack(
        config_path=config,
        fault_facts_path=facts,
        output_root=tmp_path / "pack",
    )

    assert manifest["document_count"] == 2
    assert manifest["fault_coverage_count"] == 2
    assert manifest["all_documents_dual_processed"] is True
    assert manifest["totals"]["parsed_elements"] == manifest["totals"]["crosswalks"]
    assert all(row["candidate_kg_staged"] for row in manifest["documents"])
    assert all(row["candidate_graph_built"] is False for row in manifest["documents"])
    assert all(row["publication_allowed"] is False for row in manifest["documents"])
    assert all(row["runtime_vote_allowed"] is False for row in manifest["documents"])

    by_fault = {row["fault_class_id"]: row for row in manifest["documents"]}
    text_a = (tmp_path / "pack" / by_fault["fault_a"]["path"]).read_text(encoding="utf-8")
    text_b = (tmp_path / "pack" / by_fault["fault_b"]["path"]).read_text(encoding="utf-8")
    assert "Registered evidence definition: `ED-A`" in text_a
    assert "Absence of supporting evidence is not exclusion" in text_a
    assert "NO_REGISTERED_EXECUTABLE_COUNTEREVIDENCE" in text_b
    assert "abstain or escalate" in text_b


def test_rebuild_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    config, facts = _inputs(tmp_path)
    first = build_fault_confusion_document_pack(
        config_path=config, fault_facts_path=facts, output_root=tmp_path / "one"
    )
    second = build_fault_confusion_document_pack(
        config_path=config, fault_facts_path=facts, output_root=tmp_path / "two"
    )
    # The full processing receipt is deliberately bound to the physical source
    # path used for later tamper checks.  The portable content release identity
    # excludes that machine-local path while retaining every generated file hash.
    assert first["content_release_sha256"] == second["content_release_sha256"]
    assert [row["source_sha256"] for row in first["documents"]] == [
        row["source_sha256"] for row in second["documents"]
    ]
    with pytest.raises(ValueError, match="refusing to overwrite"):
        build_fault_confusion_document_pack(
            config_path=config,
            fault_facts_path=facts,
            output_root=tmp_path / "one",
        )


def test_rejects_incomplete_fault_universe(tmp_path: Path) -> None:
    config_path, facts = _inputs(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["groups"][0]["fault_classes"] = ["fault_a", "fault_a"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate faults"):
        build_fault_confusion_document_pack(
            config_path=config_path,
            fault_facts_path=facts,
            output_root=tmp_path / "pack",
        )
