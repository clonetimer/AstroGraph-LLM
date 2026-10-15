import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from astrograph import filesystem
from astrograph.retrieval.chunking_benchmark import run_chunking_machine_benchmark
from astrograph.retrieval.chunking_gold_review import (
    FIELDS,
    _machine_rows,
    build_chunking_gold_review_workbook,
    freeze_chunking_gold_review,
)
from astrograph.retrieval.document_pack_builder import build_fault_document_pack


@pytest.mark.skipif(os.name != "nt", reason="Windows transient handle contract")
def test_directory_publish_retries_transient_windows_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "ready.txt").write_text("ready", encoding="utf-8")
    output = tmp_path / "published"
    real_replace = os.replace
    calls = 0

    def transient_replace(source, target):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("transient scanner lock")
        return real_replace(source, target)

    monkeypatch.setattr(filesystem.os, "replace", transient_replace)
    monkeypatch.setattr(filesystem.time, "sleep", lambda _: None)
    filesystem.publish_staged_directory(staging, output)
    assert calls == 3
    assert (output / "ready.txt").read_text(encoding="utf-8") == "ready"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    coverage = tmp_path / "coverage.csv"
    coverage.write_text(
        "dataset_id,dataset_fault_class_id,physical_condition_ids,generator_condition_summaries,runtime_fault_id,proposed_runtime_fault_id,component_id,mechanism_id,expected_effect_pattern_ids,evidence_definition_ids,coverage_status,review_status\n"
        'fixture.v1,wheel_friction,"[""friction""]","[{""description"":""Wheel drag reduces achieved torque."",""parameter_names"":[""coefficient""]}]",WHEEL_FRICTION,WHEEL_FRICTION,wheel,mechanism:friction,"[""effect:torque-loss""]","[""ED-1""]",EXECUTABLE,HUMAN_REVIEW_REQUIRED\n',
        encoding="utf-8",
    )
    generator = tmp_path / "generator.py"
    generator.write_text("CONDITIONS=[]\n", encoding="utf-8")
    pack = tmp_path / "pack"
    receipt = build_fault_document_pack(
        coverage_matrix_path=coverage,
        output_root=pack,
        generator_paths={"fixture": generator},
    )
    handbook = pack / "fault_design_fmea_test_handbook.md"
    facts = pack / "canonical_fault_document_facts.json"
    prereg = tmp_path / "prereg.json"
    prereg.write_text(
        json.dumps(
            {
                "source_artifacts": {
                    "handbook_markdown_sha256": receipt["artifacts"][handbook.name],
                    "canonical_facts_sha256": receipt["artifacts"][facts.name],
                },
                "case_construction": {"expected_case_count": 1},
                "systems": {
                    "section_aware": {"maximum_section_units": 8},
                    "fixed_character": {
                        "chunk_characters": 1200,
                        "overlap_characters": 0,
                    },
                },
                "retrieval": {
                    "top_k": 5,
                    "bm25_weight": 0.7,
                    "term_coverage_weight": 0.3,
                    "dense_weight": 0.0,
                    "minimum_local_score": 0.0001,
                    "require_local_and_ontology": False,
                },
                "claim_boundary": "machine fixture only",
            }
        ),
        encoding="utf-8",
    )
    raw, _ = run_chunking_machine_benchmark(
        preregistration_path=prereg,
        handbook_markdown_path=handbook,
        canonical_facts_path=facts,
    )
    results = tmp_path / "results.json"
    results.write_text(json.dumps(raw), encoding="utf-8")
    protocol = tmp_path / "human_protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "schema_version": "astrograph-document-chunking-human-gold-protocol.v1",
                "status": "FROZEN_BEFORE_HUMAN_LABELING",
                "machine_preregistration_sha256": _sha(prereg),
                "machine_results_sha256": _sha(results),
                "expected_query_count": 1,
                "judgement": {
                    "relevance_grades": {
                        "0": "IRRELEVANT",
                        "1": "PARTIAL_SUPPORT",
                        "2": "DIRECT_SUPPORT",
                    }
                },
                "promotion": {
                    "minimum_section_recall_at_k": 0.9,
                    "minimum_section_ndcg_at_k": 0.85,
                    "minimum_section_mrr": 0.9,
                    "minimum_citation_locator_validity": 1.0,
                    "minimum_ndcg_delta": 0.05,
                    "bootstrap_confidence": 0.95,
                    "bootstrap_iterations": 1000,
                    "bootstrap_seed": 7,
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "preregistration_path": prereg,
        "machine_results_path": results,
        "handbook_path": handbook,
        "canonical_facts_path": facts,
        "protocol_path": protocol,
    }


def _build(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    inputs = _fixture(tmp_path)
    workbook = tmp_path / "workbook"
    receipt = build_chunking_gold_review_workbook(**inputs, output_dir=workbook)
    assert receipt["query_count"] == 1
    assert receipt["machine_proxy_is_human_gold"] is False
    return inputs, workbook


def test_blank_chunking_gold_freeze_is_blocked(tmp_path: Path) -> None:
    inputs, workbook = _build(tmp_path)
    output = tmp_path / "blank-freeze.json"
    result = freeze_chunking_gold_review(
        **inputs, workbook_root=workbook, output_path=output
    )
    assert result["status"] == "BLOCKED"
    assert result["human_gold_complete"] is False
    assert result["metrics"] is None
    assert result["machine_proxy_used_as_gold"] is False


def test_named_human_labels_recompute_and_promote_section_chunking(tmp_path: Path) -> None:
    inputs, workbook = _build(tmp_path)
    working = workbook / "retrieval_gold_review_working_copy.csv"
    with working.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _, _, protected_mapping = _machine_rows(**inputs)
    for row in rows:
        system = protected_mapping[row["review_item_id"]]["system_id"]
        row["relevance_grade"] = "2" if system == "section_aware" else "0"
        row["citation_locator_valid"] = "PASS"
        row["reviewer"] = "human-reviewer"
        row["review_comment"] = "Compared the query, returned text and source locator."
    with working.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    result = freeze_chunking_gold_review(
        **inputs,
        workbook_root=workbook,
        output_path=tmp_path / "complete-freeze.json",
    )

    assert result["status"] == "COMPLETE"
    assert result["human_gold_complete"] is True
    assert result["promotion_decision"] == "PROMOTE_SECTION_AWARE_HUMAN_GOLD"
    assert result["metrics"]["section_aware"]["mean_recall_at_k"] == 1.0
    assert result["metrics"]["fixed_character"]["mean_recall_at_k"] == 0.0
    assert result["comparison"]["paired_bootstrap_lower_bound"] > 0.0
    assert result["publication_allowed"] is False


def test_chunking_gold_reconstructs_machine_fields(tmp_path: Path) -> None:
    inputs, workbook = _build(tmp_path)
    working = workbook / "retrieval_gold_review_working_copy.csv"
    with working.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["chunk_text"] += " tampered"
    with working.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="machine field changed"):
        freeze_chunking_gold_review(
            **inputs,
            workbook_root=workbook,
            output_path=tmp_path / "tampered-freeze.json",
        )
