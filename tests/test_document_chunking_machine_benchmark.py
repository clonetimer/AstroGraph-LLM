import json
from pathlib import Path

from astrograph.retrieval.chunking_benchmark import run_chunking_machine_benchmark
from astrograph.retrieval.document_pack_builder import build_fault_document_pack


def test_chunking_benchmark_is_hash_frozen_and_keeps_negative_result(tmp_path):
    coverage = tmp_path / "coverage.csv"
    coverage.write_text(
        "dataset_id,dataset_fault_class_id,physical_condition_ids,generator_condition_summaries,runtime_fault_id,proposed_runtime_fault_id,component_id,mechanism_id,expected_effect_pattern_ids,evidence_definition_ids,coverage_status,review_status\n"
        'fixture.v1,wheel_friction,"[""friction""]","[{""description"":""Wheel drag reduces achieved torque."",""parameter_names"":[""coefficient""]}]",WHEEL_FRICTION,WHEEL_FRICTION,wheel,mechanism:friction,"[""effect:torque-loss""]","[""ED-1""]",EXECUTABLE,HUMAN_REVIEW_REQUIRED\n',
        encoding="utf-8",
    )
    generator = tmp_path / "generator.py"
    generator.write_text("CONDITIONS=[]\n", encoding="utf-8")
    pack = tmp_path / "pack"
    pack_receipt = build_fault_document_pack(
        coverage_matrix_path=coverage,
        output_root=pack,
        generator_paths={"fixture": generator},
    )
    prereg = {
        "source_artifacts": {
            "handbook_markdown_sha256": pack_receipt["artifacts"]["fault_design_fmea_test_handbook.md"],
            "canonical_facts_sha256": pack_receipt["artifacts"]["canonical_fault_document_facts.json"],
        },
        "case_construction": {"expected_case_count": 1},
        "systems": {
            "section_aware": {"maximum_section_units": 8},
            "fixed_character": {"chunk_characters": 1200, "overlap_characters": 0},
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
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")

    raw, receipt = run_chunking_machine_benchmark(
        preregistration_path=prereg_path,
        handbook_markdown_path=pack / "fault_design_fmea_test_handbook.md",
        canonical_facts_path=pack / "canonical_fault_document_facts.json",
    )

    assert receipt["case_count"] == 1
    assert receipt["promotion_decision"] == "NONE_MACHINE_BENCHMARK_NOT_HUMAN_GOLD"
    assert set(raw) == {
        "schema_version", "preregistration_sha256", "section_aware", "fixed_character"
    }
    assert "delta" in receipt["comparison"]["mean_recall_at_k"]
