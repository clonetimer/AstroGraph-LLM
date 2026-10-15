import csv
import hashlib
import json
from pathlib import Path

import pytest

from astrograph.agents.knowledge_curation import KnowledgeExtractionProposal
from astrograph.knowledge.document_graph_review import build_document_graph_review_handoff
from astrograph.knowledge.document_candidate_correction import (
    build_document_candidate_correction_suggestions,
    build_document_candidate_correction_workbook,
    compile_document_candidate_correction_workbook,
    prepare_document_candidate_correction,
)
from astrograph.knowledge.document_pack_entrypoint import build_document_pack_candidate_packages
from astrograph.knowledge.document_review_suggestions import (
    build_document_component_archetype_suggestions,
    build_document_review_suggestions,
    build_document_review_workbook,
    compile_document_review_workbook,
)
from astrograph.retrieval.confusion_document_builder import build_fault_confusion_document_pack


class _EmptyBackend:
    model_id = "suggestion-fixture"
    model_version = "1"

    def generate_structured(self, *, agent_id, output_model, payload):
        return KnowledgeExtractionProposal(
            source_profile_sha256=payload["source_profile_sha256"],
            proposer_id=self.model_id,
            proposer_version=self.model_version,
        )


class _BlockedBackend:
    model_id = "blocked-suggestion-fixture"
    model_version = "1"

    def generate_structured(self, *, agent_id, output_model, payload):
        fragment = payload["source_profiles"][0]["fragments"][0]["fragment_id"]
        suffix = hashlib.sha256(fragment.encode()).hexdigest()[:10]
        source_id = f"candidate:block-a:{suffix}"
        target_id = f"candidate:block-b:{suffix}"
        return KnowledgeExtractionProposal(
            source_profile_sha256=payload["source_profile_sha256"],
            nodes=[
                {"node_id": source_id, "node_type": "FailureMechanism",
                 "canonical_name": "blocked a", "source_fragment_ids": [fragment],
                 "confidence": 0.5, "ambiguous": True},
                {"node_id": target_id, "node_type": "FailureMechanism",
                 "canonical_name": "blocked b", "source_fragment_ids": [fragment],
                 "confidence": 0.5, "ambiguous": True},
            ],
            relations=[{
                "edge_id": f"candidate-edge:block-a-to-b:{suffix}", "source_id": source_id,
                "relation_type": "SUSCEPTIBLE_TO", "target_id": target_id,
                "source_fragment_ids": [fragment], "confidence": 0.5, "ambiguous": True,
                "rationale": "Deliberately invalid V3 relation shape for a blocked fixture.",
            }],
            proposer_id=self.model_id, proposer_version=self.model_version,
        )


def _build_inputs(tmp_path: Path, backend_factory=_EmptyBackend) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "config.json"
    facts = tmp_path / "facts.json"
    config.write_text(json.dumps({
        "schema_version": "astrograph-fault-confusion-groups.v1",
        "configuration_id": "suggestion-fixture-r001",
        "review_state": "HUMAN_REVIEW_REQUIRED",
        "groups": [{
            "group_id": "g1", "dataset_id": "fixture.v1",
            "fault_classes": ["fault_a", "fault_b"],
            "discriminator_questions": ["Which boundary disagrees?"],
        }],
        "claim_boundary": "Candidate associations only.",
    }), encoding="utf-8")
    facts.write_text(json.dumps([{
        "dataset_id": "fixture.v1", "fault_class_id": fault,
        "component_id": f"new_component_{fault}", "mechanism_id": "mechanism:fault-a",
        "generator_condition_summaries": [], "evidence_definition_ids": ["ED-A"],
    } for fault in ("fault_a", "fault_b")]), encoding="utf-8")
    corpus = tmp_path / "corpus"
    build_fault_confusion_document_pack(config_path=config, fault_facts_path=facts, output_root=corpus)
    refreeze = tmp_path / "refreeze.json"
    refreeze.write_text(json.dumps({
        "temporary_frozen": True, "publication_allowed": False,
        "refreeze_sha256": "6" * 64,
    }), encoding="utf-8")
    pack = tmp_path / "pack"
    build_document_pack_candidate_packages(
        manifest_path=corpus / "manifest.json", corpus_root=corpus, output_root=pack,
        review_refreeze_path=refreeze,
        expected_review_refreeze_file_sha256=hashlib.sha256(refreeze.read_bytes()).hexdigest(),
        backend_factory=backend_factory,
    )
    review = tmp_path / "review"
    build_document_graph_review_handoff(primary_root=pack, supplemental_roots=[], output_dir=review)
    base = tmp_path / "base.json"
    base.write_text(json.dumps({
        "schema_version": "astrograph-knowledge-schema.v3", "graph_id": "graph:base",
        "nodes": [{
            "node_id": "mechanism:fault-a", "node_type": "FailureMechanism",
            "schema_version": "astrograph-knowledge-schema.v3", "source_id": "source:test",
            "source_sha256": "a" * 64, "lifecycle_status": "ACTIVE",
            "properties": {"canonical_name": "Fault A", "aliases": []},
        }, {
            "node_id": "archetype:storage.energy_buffer",
            "node_type": "ComponentArchetype",
            "schema_version": "astrograph-knowledge-schema.v3",
            "source_id": "source:test",
            "source_sha256": "b" * 64,
            "lifecycle_status": "ACTIVE",
            "properties": {"canonical_name": "storage.energy_buffer", "aliases": []},
        }],
        "edges": [], "persistent_raw_telemetry": False,
    }), encoding="utf-8")
    return pack, review, base


def test_suggestions_align_exact_id_defer_unmatched_and_preserve_review(tmp_path: Path) -> None:
    pack, review, base = _build_inputs(tmp_path)
    before = (review / "human_review_decisions.csv").read_bytes()
    output = tmp_path / "suggestions"
    receipt = build_document_review_suggestions(
        candidate_pack_root=pack, review_root=review, base_graph_path=base, output_dir=output
    )
    assert receipt["suggestion_counts"] == {
        "ALIGN_EXISTING_EXACT_ID": 2, "DEFER_CANONICAL_MAPPING": 2,
    }
    assert receipt["human_decision_modified"] is False
    assert receipt["publication_allowed"] is False
    assert receipt["active_graph_modified"] is False
    assert (review / "human_review_decisions.csv").read_bytes() == before

    with pytest.raises(ValueError, match="refusing to overwrite"):
        build_document_review_suggestions(
            candidate_pack_root=pack, review_root=review,
            base_graph_path=base, output_dir=output,
        )


def test_suggestions_reject_changed_review_or_candidate_artifact(tmp_path: Path) -> None:
    pack, review, base = _build_inputs(tmp_path)
    sheet = review / "human_review_decisions.csv"
    sheet.write_bytes(sheet.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="review sheet changed"):
        build_document_review_suggestions(
            candidate_pack_root=pack, review_root=review,
            base_graph_path=base, output_dir=tmp_path / "bad-review",
        )
    assert not (tmp_path / "bad-review").exists()

    pack, review, base = _build_inputs(tmp_path / "second")
    artifact = pack / "fixture.v1/fault_a/candidate/raw_extraction_proposal.json"
    artifact.write_text(artifact.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact is missing or changed"):
        build_document_review_suggestions(
            candidate_pack_root=pack, review_root=review,
            base_graph_path=base, output_dir=tmp_path / "bad-pack",
        )
    assert not (tmp_path / "bad-pack").exists()


def test_blocked_candidate_rows_only_request_correction(tmp_path: Path) -> None:
    pack, review, base = _build_inputs(tmp_path, _BlockedBackend)
    output = tmp_path / "suggestions"
    receipt = build_document_review_suggestions(
        candidate_pack_root=pack, review_root=review, base_graph_path=base, output_dir=output
    )
    assert receipt["suggestion_counts"] == {
        "REQUIRES_CANDIDATE_CORRECTION": receipt["suggestion_count"]
    }


def test_blocked_candidate_correction_advice_only_drops_deterministic_bad_shape(
    tmp_path: Path,
) -> None:
    pack, _, _ = _build_inputs(tmp_path, _BlockedBackend)
    candidate = pack / "fixture.v1" / "fault_a" / "candidate"
    handoff = tmp_path / "correction-handoff"
    prepare_document_candidate_correction(
        candidate_package_root=candidate,
        output_dir=handoff,
    )
    human_sheet = handoff / "human_correction_decisions.csv"
    before = human_sheet.read_bytes()
    output = tmp_path / "correction-suggestions"
    receipt = build_document_candidate_correction_suggestions(
        candidate_package_root=candidate,
        correction_handoff_root=handoff,
        output_dir=output,
    )
    assert receipt["suggestion_count"] > 0
    assert receipt["suggestion_counts"][
        "DROP_DETERMINISTICALLY_INVALID_RELATION_SHAPE"
    ] > 0
    assert receipt["suggestion_counts"][
        "HUMAN_REPLACE_OR_DROP_REVIEW_REQUIRED"
    ] > 0
    assert receipt["human_decisions_written"] is False
    assert receipt["corrected_candidate_built"] is False
    assert human_sheet.read_bytes() == before
    with (output / "machine_correction_suggestions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    bad_relations = [row for row in rows if row["item_kind"] == "RELATION"]
    assert bad_relations
    assert all(row["suggested_action"] == "DROP" for row in bad_relations)
    assert all(
        "V3_RELATION_SHAPE_NOT_ALLOWED" in row["reason_codes"]
        for row in bad_relations
    )

    human_sheet.write_bytes(before + b"\n")
    with pytest.raises(ValueError, match="not bound"):
        build_document_candidate_correction_suggestions(
            candidate_package_root=candidate,
            correction_handoff_root=handoff,
            output_dir=tmp_path / "tampered-correction-suggestions",
        )


def test_correction_workbook_locks_advice_and_compiles_only_named_actions(
    tmp_path: Path,
) -> None:
    pack, _, _ = _build_inputs(tmp_path, _BlockedBackend)
    candidate = pack / "fixture.v1" / "fault_a" / "candidate"
    handoff = tmp_path / "correction-handoff"
    prepare_document_candidate_correction(
        candidate_package_root=candidate,
        output_dir=handoff,
    )
    suggestions = tmp_path / "correction-suggestions"
    build_document_candidate_correction_suggestions(
        candidate_package_root=candidate,
        correction_handoff_root=handoff,
        output_dir=suggestions,
    )
    workbook = tmp_path / "correction-workbook"
    receipt = build_document_candidate_correction_workbook(
        candidate_package_root=candidate,
        correction_handoff_root=handoff,
        suggestion_root=suggestions,
        output_dir=workbook,
    )
    baseline = workbook / "human_correction_priority_queue.csv"
    working = workbook / "human_correction_working_copy.csv"
    assert baseline.read_bytes() == working.read_bytes()
    assert receipt["priority_counts"][
        "P0_DETERMINISTIC_RELATION_SHAPE_FAILURE"
    ] > 0
    blank = compile_document_candidate_correction_workbook(
        candidate_package_root=candidate,
        workbook_root=workbook,
        workbook_path=working,
        output_dir=tmp_path / "compiled-blank",
    )
    assert blank["status"] == "BLOCKED"
    assert blank["corrected_candidate_built"] is False

    with working.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), list(reader.fieldnames or [])
    for row in rows:
        row["action"] = "DROP" if row["suggested_action"] == "DROP" else "KEEP"
        row["reviewer"] = "domain-reviewer"
        row["review_comment"] = "Checked against source candidate and critic findings."
    filled = tmp_path / "filled-correction.csv"
    with filled.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    compiled = compile_document_candidate_correction_workbook(
        candidate_package_root=candidate,
        workbook_root=workbook,
        workbook_path=filled,
        output_dir=tmp_path / "compiled-ready",
    )
    assert compiled["status"] == "READY_FOR_CORRECTION_REVALIDATION"
    with (tmp_path / "compiled-ready" / "human_correction_decisions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        compiled_reader = csv.DictReader(stream)
        assert list(compiled_reader.fieldnames or [])[-4:] == [
            "action", "replacement_candidate_json", "reviewer", "review_comment"
        ]

    rows[0]["machine_suggestion"] = "FORGED"
    tampered = tmp_path / "tampered-workbook.csv"
    with tampered.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="machine field changed"):
        compile_document_candidate_correction_workbook(
            candidate_package_root=candidate,
            workbook_root=workbook,
            workbook_path=tampered,
            output_dir=tmp_path / "compiled-tampered",
        )


def test_priority_workbook_compiles_named_decisions_without_prefilling_them(
    tmp_path: Path,
) -> None:
    pack, review, base = _build_inputs(tmp_path)
    suggestions = tmp_path / "suggestions"
    build_document_review_suggestions(
        candidate_pack_root=pack,
        review_root=review,
        base_graph_path=base,
        output_dir=suggestions,
    )
    workbook = tmp_path / "workbook"
    receipt = build_document_review_workbook(
        candidate_pack_root=pack,
        review_root=review,
        suggestion_root=suggestions,
        output_dir=workbook,
    )
    assert receipt["priority_counts"] == {
        "P2_CANONICAL_MAPPING_REQUIRED": 2,
        "P3_EXACT_ALIGNMENT_CHECK": 2,
    }
    baseline = workbook / "human_review_priority_queue.csv"
    working = workbook / "human_review_working_copy.csv"
    assert baseline.read_bytes() == working.read_bytes()
    with working.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), list(reader.fieldnames or [])
    assert all(not row["decision"] and not row["reviewer"] for row in rows)

    blank = compile_document_review_workbook(
        candidate_pack_root=pack,
        workbook_root=workbook,
        workbook_path=working,
        output_dir=tmp_path / "compiled-blank",
    )
    assert blank["status"] == "BLOCKED"
    assert blank["knowledge_patch_proposal_allowed"] is False

    for row in rows:
        exact = row["machine_suggestion"].startswith("ALIGN_EXISTING_EXACT")
        row.update(
            {
                "decision": "APPROVE" if exact else "DEFER",
                "approved_canonical_id": row["suggested_canonical_id"] if exact else "",
                "reviewer": "domain-reviewer",
                "review_comment": "Checked against candidate source and base graph.",
            }
        )
    filled = tmp_path / "filled.csv"
    with filled.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    compiled = compile_document_review_workbook(
        candidate_pack_root=pack,
        workbook_root=workbook,
        workbook_path=filled,
        output_dir=tmp_path / "compiled-ready",
    )
    assert compiled["status"] == "READY_FOR_KNOWLEDGE_PATCH_PROPOSAL"
    assert compiled["decision_counts"] == {"APPROVE": 2, "DEFER": 2, "REJECT": 0}
    assert compiled["knowledge_patch_proposal_allowed"] is True
    assert compiled["publication_allowed"] is False


def test_priority_workbook_rejects_advisory_or_candidate_field_tampering(
    tmp_path: Path,
) -> None:
    pack, review, base = _build_inputs(tmp_path)
    suggestions = tmp_path / "suggestions"
    build_document_review_suggestions(
        candidate_pack_root=pack,
        review_root=review,
        base_graph_path=base,
        output_dir=suggestions,
    )
    workbook = tmp_path / "workbook"
    build_document_review_workbook(
        candidate_pack_root=pack,
        review_root=review,
        suggestion_root=suggestions,
        output_dir=workbook,
    )
    with (workbook / "human_review_working_copy.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), list(reader.fieldnames or [])
    rows[0]["machine_suggestion"] = "ALIGN_EXISTING_EXACT_ID"
    tampered = tmp_path / "tampered.csv"
    with tampered.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="machine field changed"):
        compile_document_review_workbook(
            candidate_pack_root=pack,
            workbook_root=workbook,
            workbook_path=tampered,
            output_dir=tmp_path / "compiled-tampered",
        )


def test_exact_channel_map_component_archetypes_enrich_workbook_without_approval(
    tmp_path: Path,
) -> None:
    pack, review, base = _build_inputs(tmp_path)
    suggestions = tmp_path / "suggestions"
    build_document_review_suggestions(
        candidate_pack_root=pack,
        review_root=review,
        base_graph_path=base,
        output_dir=suggestions,
    )
    catalog = tmp_path / "channel-maps.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": "astrograph-channel-map-catalog.v1",
                "maps": [
                    {
                        "mapping_id": f"fixture.component.new-component-{fault}.v1",
                        "entries": [
                            {"component_archetype_id": "storage.energy_buffer"}
                        ],
                    }
                    for fault in ("fault-a", "fault-b")
                ],
            }
        ),
        encoding="utf-8",
    )
    component_suggestions = tmp_path / "component-suggestions"
    component_receipt = build_document_component_archetype_suggestions(
        review_root=review,
        suggestion_root=suggestions,
        channel_map_catalog_path=catalog,
        base_graph_path=base,
        output_dir=component_suggestions,
    )
    assert component_receipt["suggestion_counts"] == {
        "ALIGN_ARCHETYPE_FROM_EXACT_CHANNEL_MAP_COMPONENT_ID": 2
    }
    assert component_receipt["human_decisions_written"] is False
    assert component_receipt["knowledge_patch_proposal_allowed"] is False

    workbook = tmp_path / "enriched-workbook"
    workbook_receipt = build_document_review_workbook(
        candidate_pack_root=pack,
        review_root=review,
        suggestion_root=suggestions,
        component_suggestion_root=component_suggestions,
        output_dir=workbook,
    )
    assert workbook_receipt["priority_counts"] == {
        "P2_EXACT_CHANNEL_MAP_ARCHETYPE_CHECK": 2,
        "P3_EXACT_ALIGNMENT_CHECK": 2,
    }
    with (workbook / "human_review_working_copy.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    component_rows = [row for row in rows if row["node_type"] == "ComponentArchetype"]
    assert all(
        row["component_suggested_canonical_id"]
        == "archetype:storage.energy_buffer"
        for row in component_rows
    )
    assert all(not row["decision"] for row in rows)
