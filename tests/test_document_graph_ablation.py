import hashlib
import json

import pytest

from astrograph.knowledge.document_graph_ablation import (
    build_document_graph_ablation_report,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ablation_requires_same_comparison_set_and_reports_no_promotion(tmp_path) -> None:
    variants = []
    for index, count in enumerate((0, 1)):
        receipt = {
            "configuration_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
            "selected_document_count": 8,
            "lane_counts": {"episode": 4, "historical": 2, "quarantine": 2},
            "deterministic_scaffold_counts": {"nodes": 16, "relations": 0},
            "deterministic_content_extraction_counts": {"nodes": 28, "relations": 52},
            "candidate_ready_count": 8,
            "blocked_count": 0,
            "llm_raw_candidate_counts": {"nodes": count, "relations": count, "unresolved_claims": 0},
            "receipt_sha256": str(index) * 64,
        }
        receipt_path = tmp_path / f"receipt-{index}.json"
        row = {
            "variant_id": f"v{index}", "intervention": f"i{index}",
            "receipt_path": receipt_path.name,
            "receipt_file_sha256": _write(receipt_path, receipt),
        }
        if count:
            audit = {
                "status": "PASS", "finding_counts": {}, "audit_sha256": "c" * 64,
                "results": [{"checked_explicit_field_count": 4,
                             "grounded_explicit_field_count": 4,
                             "valid_cited_candidate_count": 2}],
            }
            audit_path = tmp_path / "audit.json"
            row.update({"grounding_audit_path": audit_path.name,
                        "grounding_audit_file_sha256": _write(audit_path, audit)})
        variants.append(row)
    config = {
        "schema_version": "astrograph-document-graph-ablation-config.v1",
        "ablation_id": "test", "comparison_controls": ["same docs"], "variants": variants,
    }
    config_path = tmp_path / "config.json"
    _write(config_path, config)
    report = build_document_graph_ablation_report(
        config_path=config_path, project_root=tmp_path, output_path=tmp_path / "report.json"
    )
    assert report["decision"] == "NO_PROMOTION_HUMAN_GOLD_REQUIRED"
    assert report["variants"][1]["grounding"]["checked_explicit_field_count"] == 4


def test_ablation_rejects_changed_comparison_set(tmp_path) -> None:
    rows = []
    for index, selected in enumerate((8, 9)):
        receipt = {
            "configuration_sha256": "a" * 64, "source_manifest_sha256": "b" * 64,
            "selected_document_count": selected, "lane_counts": {},
            "deterministic_scaffold_counts": {}, "deterministic_content_extraction_counts": {},
            "candidate_ready_count": selected, "blocked_count": 0,
            "llm_raw_candidate_counts": {"nodes": 0, "relations": 0, "unresolved_claims": 0},
            "receipt_sha256": str(index) * 64,
        }
        path = tmp_path / f"r{index}.json"
        rows.append({"variant_id": str(index), "intervention": "x", "receipt_path": path.name,
                     "receipt_file_sha256": _write(path, receipt)})
    config_path = tmp_path / "config.json"
    _write(config_path, {"schema_version": "astrograph-document-graph-ablation-config.v1",
                         "ablation_id": "test", "comparison_controls": [], "variants": rows})
    with pytest.raises(ValueError, match="same frozen comparison set"):
        build_document_graph_ablation_report(
            config_path=config_path, project_root=tmp_path, output_path=tmp_path / "out.json"
        )
