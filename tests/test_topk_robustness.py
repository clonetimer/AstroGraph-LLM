from __future__ import annotations

from astrograph.evaluation.topk_robustness import run_topk_robustness


def _hypothesis(fault_id: str, status: str, support: int = 0):
    return {
        "hypothesis_type": "REGISTERED_FAULT",
        "target_fault_id": fault_id,
        "status": status,
        "supporting_evidence_fact_ids": [f"EF-{fault_id}-{i}" for i in range(support)],
        "contradicting_evidence_fact_ids": [] if status != "CONTRADICTED" else [f"EF-C-{fault_id}"],
        "missing_evidence_definition_ids": [],
    }


def test_robustness_report_compares_2_4_6_and_preserves_unknown_refusal():
    documents = [
        ("single", {"hypotheses": [
            _hypothesis("F-A", "SUPPORTED", 1),
            _hypothesis("F-B", "CONTRADICTED"),
            _hypothesis("F-C", "CONTRADICTED"),
            _hypothesis("F-D", "CONTRADICTED"),
            _hypothesis("F-E", "CONTRADICTED"),
        ]}),
        ("composite", {"hypotheses": [
            _hypothesis("F-A", "SUPPORTED", 1),
            _hypothesis("F-B", "SUPPORTED", 1),
            _hypothesis("F-C", "CONTRADICTED"),
            _hypothesis("F-D", "CONTRADICTED"),
            _hypothesis("F-E", "CONTRADICTED"),
        ]}),
        ("nominal", {"hypotheses": [
            _hypothesis("F-A", "CONTRADICTED"),
            _hypothesis("F-B", "CONTRADICTED"),
            _hypothesis("F-C", "CONTRADICTED"),
        ]}),
    ]
    report = run_topk_robustness(documents)
    assert [row["top_k"] for row in report["summary"]] == [2, 4, 6]
    assert all(row["safe_empty_unknown_proxy"] for row in report["summary"])
    assert report["summary"][2]["member_recall"] >= report["summary"][0]["member_recall"]
