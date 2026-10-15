import json
from pathlib import Path

import pytest

from astrograph.evaluation.fusion_dependence import FusionEvaluationCase, audit_birrf_leave_branch_out
from astrograph.fusion import BiRRFConfig, FaultRanking, RankedFault, RankingSource, ScoreSemantics


def _ranking(source, episode, order):
    return FaultRanking(
        ranking_id=f"{episode}-{source.value}", episode_id=episode, source=source,
        score_semantics=ScoreSemantics.RANKING_SCORE,
        producer_id=source.value, producer_version="1",
        candidates=[RankedFault(fault_id=fault, rank=i + 1, score=float(len(order) - i), source=source) for i, fault in enumerate(order)],
    )


def _case(case_id, group_id, truth, data_order, forward_order, backward_order):
    return FusionEvaluationCase(
        case_id=case_id, group_id=group_id, true_fault_id=truth,
        data_ranking=_ranking(RankingSource.DATA_DRIVEN, case_id, data_order),
        forward_ranking=_ranking(RankingSource.KG_FORWARD, case_id, forward_order),
        backward_ranking=_ranking(RankingSource.KG_BACKWARD, case_id, backward_order),
    )


def test_leave_branch_out_is_group_aware_and_noncausal():
    cases = [
        _case("c1", "g1", "A", ["A", "B"], ["B", "A"], ["B", "A"]),
        _case("c2", "g1", "A", ["A", "B"], ["B", "A"], ["B", "A"]),
        _case("c3", "g2", "B", ["B", "A"], ["B", "A"], ["A", "B"]),
        _case("c4", "g3", "A", ["B", "A"], ["A", "B"], ["A", "B"]),
    ]
    report = audit_birrf_leave_branch_out(cases, BiRRFConfig(), bootstrap_resamples=500, bootstrap_seed=3)
    assert report["case_count"] == 4
    assert report["independent_group_count"] == 3
    assert set(report["policies"]) == {"FULL", "DATA_ONLY", "FORWARD_ONLY", "BACKWARD_ONLY", "NO_DATA", "NO_FORWARD", "NO_BACKWARD"}
    assert report["source_dependence_status"] == "STRUCTURAL_SENSITIVITY_NOT_CAUSAL_ATTRIBUTION"
    assert report["calibration_status"] == "NOT_PROBABILITY_CALIBRATION"
    assert report["full_vs_variants"]["NO_DATA"]["full_minus_variant_equal_group_accuracy_delta"]["group_count"] == 3
    assert len(report["report_sha256"]) == 64


def test_fusion_case_rejects_branch_source_mismatch():
    with pytest.raises(ValueError, match="data_ranking"):
        FusionEvaluationCase(
            case_id="x", group_id="g", true_fault_id="A",
            data_ranking=_ranking(RankingSource.KG_FORWARD, "x", ["A"]),
            forward_ranking=_ranking(RankingSource.KG_FORWARD, "x", ["A"]),
            backward_ranking=_ranking(RankingSource.KG_BACKWARD, "x", ["A"]),
        )


def test_r18_birrf_dependence_design_is_not_calibration_claim():
    design = json.loads(Path("configs/evaluation/birrf_dependence_r18.v1.json").read_text(encoding="utf-8"))
    assert design["inferential_unit"] == "independent_group_id"
    assert "NOT" not in design.get("schema_version", "")
    assert "does not identify causal source value" in design["claim_boundary"]
