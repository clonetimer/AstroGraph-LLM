"""Regression tests for audit counterexamples; no measured model scores here."""
from __future__ import annotations
import math
import pytest
from pydantic import ValidationError
from astrograph.benchmark import BenchmarkRecord, evaluate
from astrograph.evaluation import LLMMethodAblationRow, summarize_llm_method_ablation
from astrograph.fusion import FaultRanking, RankedFault, RankingSource, BiRRFConfig, fuse_rankings


def record(**changes):
    return dict(case_id="c1", dataset_id="d", group_id="run1", true_fault_id="B",
                predicted_fault_id="A", ranked_fault_ids=["A"], cited_evidence_ids=["e1"],
                latency_s=1.0, **changes)


def test_wrong_prediction_does_not_make_perfect_metrics():
    result = evaluate([record()])
    assert result["accuracy"] == 0
    assert result["top_k_recall"] == 0
    assert result["evidence_coverage"] is None
    assert result["hallucination_rate"] is None


def test_metric_denominators_use_truth_and_independent_annotations():
    row = record(expected_evidence_ids=["e1", "e2"], claim_assessments=[
        {"claim_id": "1", "support": "UNSUPPORTED", "assessor_id": "expert-A", "assessment_basis": "counterevidence"},
        {"claim_id": "2", "support": "SUPPORTED", "assessor_id": "expert-A", "assessment_basis": "source quote"},
        {"claim_id": "3", "support": "UNASSESSED", "assessor_id": "expert-A", "assessment_basis": "not reviewed"}])
    row["ranked_fault_ids"] = ["A", "B"]
    result = evaluate([row], k=2)
    assert result["accuracy"] == 0
    assert result["top_k_recall"] == 1
    assert result["evidence_coverage"] == .5
    assert result["hallucination_rate"] == .5
    assert result["claim_review_coverage"] == pytest.approx(2/3)
    assert evaluate([row], k=1)["top_k_recall"] == 0


@pytest.mark.parametrize("change", [
    {"latency_s": float("nan")}, {"latency_s": float("inf")}, {"latency_s": -1},
    {"ranked_fault_ids": ["A", "A"]}, {"cited_evidence_ids": ["e1", "e1"]},
    {"expected_evidence_ids": ["e1", "e1"]}, {"predicted_fault_id": " "},
    {"correct": "false"}])
def test_malformed_measurements_rejected(change):
    value = record(); value.update(change)
    with pytest.raises(ValidationError):
        evaluate([value])


def test_duplicate_cases_and_mixed_latency_scopes_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        evaluate([record(), record()])
    other = record(); other.update(case_id="c2", latency_scope="CORE_ONLY")
    with pytest.raises(ValueError, match="mixed"):
        evaluate([record(), other])


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_top_k_rejected(k):
    with pytest.raises(ValueError):
        evaluate([record()], k=k)


def test_empty_benchmark_does_not_report_zero_error():
    with pytest.raises(ValueError):
        evaluate([])


def test_abstention_and_latency_quantiles():
    a = record(); a.update(predicted_fault_id=None, latency_s=0.1)
    b = record(); b.update(case_id="c2", predicted_fault_id="B", latency_s=.3)
    result = evaluate([a,b])
    assert result["accuracy"] == .5
    assert result["diagnosis_latency_s"]["p50"] == pytest.approx(.2)
    assert result["diagnosis_latency_s"]["p95"] == pytest.approx(.29)


def ablation_row(arm="A", **changes):
    data = dict(case_id="c", dataset_id="d", group_id="g", arm_id=arm,
                true_fault_id="F", predicted_fault_id="F", abstained=False, latency_s=1.)
    data.update(changes)
    return LLMMethodAblationRow(**data)


def test_duplicate_easy_case_cannot_inflate_ablation():
    with pytest.raises(ValueError, match="duplicate"):
        summarize_llm_method_ablation([ablation_row(), ablation_row(), ablation_row("B")], expected_arm_ids=["A", "B"])


@pytest.mark.parametrize("change", [{"true_fault_id":"OTHER"},{"dataset_id":"other"},{"group_id":"different"}])
def test_ablation_case_identity_frozen_across_arms(change):
    with pytest.raises(ValueError, match="identity"):
        summarize_llm_method_ablation([ablation_row(), ablation_row("B", **change)], expected_arm_ids=["A","B"])


def test_expected_arms_cannot_repeat():
    with pytest.raises(ValueError, match="unique"):
        summarize_llm_method_ablation([ablation_row()], expected_arm_ids=["A","A"])


def ranking(source, faults):
    return FaultRanking(ranking_id=source.value, episode_id="EP", source=source,
                        producer_id="test", producer_version="1", candidates=[
        RankedFault(fault_id=f, rank=i+1, score=1/(i+1), source=source) for i,f in enumerate(faults)])


def test_native_birrf_is_not_list_concatenation():
    result = fuse_rankings(ranking(RankingSource.DATA_DRIVEN,["A","B"]),
                           ranking(RankingSource.KG_FORWARD,["B","C"]),
                           ranking(RankingSource.KG_BACKWARD,[]),
                           config=BiRRFConfig(collapse_identical_kg_rankings=False,
                                              collapse_highly_correlated_kg_rankings=False))
    assert result.candidates[0].fault_id == "B"
    assert result.candidates[0].fused_score == pytest.approx(1/62+1/61)


@pytest.mark.parametrize("source", list(RankingSource))
def test_nonfinite_ranking_scores_rejected(source):
    with pytest.raises(ValidationError):
        RankedFault(fault_id="F", rank=1, score=float("nan"), source=source)


def test_source_mismatch_cannot_impersonate_another_branch():
    data = ranking(RankingSource.DATA_DRIVEN,["F"]).model_dump()
    data["candidates"][0]["source"] = RankingSource.KG_FORWARD
    with pytest.raises(ValidationError):
        FaultRanking.model_validate(data)


def test_validation_extra_declares_direct_schema_dependencies():
    import tomllib
    from pathlib import Path
    data=tomllib.loads((Path(__file__).resolve().parents[1]/"pyproject.toml").read_text(encoding="utf-8"))
    requirements=data["project"]["optional-dependencies"]["test"]
    assert any(item.startswith("jsonschema") for item in requirements)
    assert any(item.startswith("PyYAML") for item in requirements)
