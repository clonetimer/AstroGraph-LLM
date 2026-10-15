from __future__ import annotations

import pytest

from astrograph.diagnosis import (
    CapabilityAxesEvidence,
    DiagnosisEvidenceBundle,
    DiagnosisPolicyGate,
    DiagnosisReleaseStatus,
    KGEvidenceReference,
    ModelEvidenceReference,
    SafetyReplayReference,
)
from astrograph.fusion import (
    BiRRFConfig,
    FaultRanking,
    RankedFault,
    RankingSource,
    ScoreSemantics,
    fuse_rankings,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _ranking(
    source: RankingSource,
    rows: list[tuple[str, int, float]],
    *,
    calibrated: bool = False,
    lineage_complete: bool = False,
    groups_by_fault: dict[str, list[str]] | None = None,
) -> FaultRanking:
    if source == RankingSource.DATA_DRIVEN:
        semantics = (
            ScoreSemantics.CALIBRATED_CLASS_PROBABILITY
            if calibrated
            else ScoreSemantics.UNCALIBRATED_CLASS_PROBABILITY
        )
    elif source == RankingSource.KG_FORWARD:
        semantics = ScoreSemantics.EVIDENCE_STRENGTH
    else:
        semantics = ScoreSemantics.EVIDENCE_COVERAGE
    return FaultRanking(
        ranking_id=f"R-{source.value}",
        episode_id="EP-1",
        source=source,
        score_semantics=semantics,
        calibration_id="CAL-TEST" if calibrated else None,
        producer_id="test",
        producer_version="1",
        candidates=[
            RankedFault(
                fault_id=fault_id,
                rank=rank,
                score=score,
                source=source,
                correlation_group_ids=(groups_by_fault or {}).get(fault_id, []),
                lineage_complete=lineage_complete,
                metadata={"coverage": score} if source == RankingSource.KG_BACKWARD else {},
            )
            for fault_id, rank, score in rows
        ],
    )


def _policy_bundle() -> DiagnosisEvidenceBundle:
    return DiagnosisEvidenceBundle(
        bundle_id="BUNDLE-1",
        sample_id="SAMPLE-1",
        episode_id="EP-1",
        split="ONLINE",
        telemetry_sha256=HASH_A,
        model_evidence=ModelEvidenceReference(
            model_id="MODEL-1",
            model_version="1",
            model_sha256=HASH_A,
            prediction_id="PRED-1",
            prediction_sha256=HASH_B,
            probability_ranking_id="RANK-1",
            top_fault_id="F-A",
            calibrated=True,
        ),
        kg_evidence=KGEvidenceReference(
            release_id="KG-1",
            knowledge_version="1",
            nodes_sha256=HASH_A,
            edges_sha256=HASH_B,
            query_ids=["Q-1"],
            query_sha256_values=[HASH_C],
            read_user="reader",
            row_count=1,
        ),
        safety_replay=SafetyReplayReference(
            replay_id="REPLAY-1",
            replay_sha256=HASH_C,
            status="PASS",
            wrong_automatic_accept_rate=0.0,
        ),
        capability_axes=CapabilityAxesEvidence(
            supported_axes=["seed"],
            unsupported_axes=[],
            requested_axes=["seed"],
            capability_manifest_id="CAP-1",
            capability_manifest_sha256=HASH_A,
        ),
    )


def test_rank_disagreement_and_redundancy_are_warnings_not_blocking_conflicts() -> None:
    data = _ranking(RankingSource.DATA_DRIVEN, [("F-A", 1, 0.4), ("F-B", 2, 0.3)])
    forward = _ranking(RankingSource.KG_FORWARD, [("F-B", 1, 1.0), ("F-A", 2, 0.8)])
    backward = _ranking(RankingSource.KG_BACKWARD, [("F-B", 1, 1.0), ("F-A", 2, 0.8)])

    result = fuse_rankings(data, forward, backward, config=BiRRFConfig())

    assert result.blocking_conflicts == []
    assert "DATA_TOP_DIFFERS_FROM_FORWARD_GRAPH_TOP" in result.quality_warnings
    assert "KG_BRANCH_HIGH_CORRELATION=1.000" in result.quality_warnings
    assert "KG_REDUNDANT_BRANCH_COLLAPSED=KG_BACKWARD" in result.quality_warnings
    assert result.metadata["configured_branch_weights"]["KG_FORWARD"] == 1.0
    assert result.metadata["effective_branch_weights"]["KG_FORWARD"] == 1.0
    assert result.metadata["effective_branch_weights"]["KG_BACKWARD"] == 0.0
    assert result.metadata["kg_collapse_reason"] == "IDENTICAL_RANKING"
    assert result.top_fault_id is None
    assert result.conflict_flags  # legacy aggregate remains available

    decision = DiagnosisPolicyGate().decide(
        _policy_bundle(),
        blocking_conflicts=result.blocking_conflicts,
        quality_warnings=result.quality_warnings,
    )
    assert decision.status == DiagnosisReleaseStatus.DETERMINATE_DIAGNOSIS
    assert decision.blocking_reasons == []
    assert "KG_BRANCH_HIGH_CORRELATION=1.000" in decision.warnings


def test_rank_correlation_is_not_computable_with_one_shared_candidate() -> None:
    data = _ranking(RankingSource.DATA_DRIVEN, [("F-A", 1, 0.4)])
    forward = _ranking(RankingSource.KG_FORWARD, [("F-A", 1, 1.0)])
    backward = _ranking(RankingSource.KG_BACKWARD, [("F-A", 1, 1.0)])

    result = fuse_rankings(data, forward, backward)

    assert result.metadata["kg_branch_correlation"] is None
    assert not any(flag.startswith("KG_BRANCH_HIGH_CORRELATION") for flag in result.quality_warnings)


def test_open_set_action_returns_unknown_without_mislabeling_evidence_conflict() -> None:
    decision = DiagnosisPolicyGate().decide(
        _policy_bundle(),
        blocking_conflicts=[],
        quality_warnings=["KG_BRANCH_HIGH_CORRELATION=1.000"],
        open_set_actions=["DATA_BRANCH_SUPPRESSED_BY_OPEN_SET_POLICY"],
    )

    assert decision.status == DiagnosisReleaseStatus.UNKNOWN
    assert decision.blocking_reasons == ["OPEN_SET_ACTION_REQUIRED"]
    assert "EVIDENCE_CONFLICT_DETECTED" not in decision.blocking_reasons


def test_high_confidence_data_conflict_is_non_blocking_when_fusion_resolves_it() -> None:
    data = _ranking(
        RankingSource.DATA_DRIVEN,
        [("F-A", 1, 0.90), ("F-B", 2, 0.20)],
        calibrated=True,
    )
    forward = _ranking(RankingSource.KG_FORWARD, [("F-B", 1, 1.0), ("F-A", 2, 0.20)])
    backward = _ranking(RankingSource.KG_BACKWARD, [("F-B", 1, 1.0), ("F-A", 2, 0.10)])

    result = fuse_rankings(data, forward, backward)

    assert result.top_fault_id is None
    assert result.blocking_conflicts == []
    assert "HIGH_DATA_SCORE_LOW_BACKWARD_EVIDENCE_COVERAGE" in result.quality_warnings


def test_high_confidence_data_conflict_blocks_when_fusion_keeps_data_top() -> None:
    data = _ranking(
        RankingSource.DATA_DRIVEN,
        [("F-A", 1, 0.90), ("F-B", 2, 0.20)],
        calibrated=True,
    )
    forward = _ranking(RankingSource.KG_FORWARD, [("F-A", 1, 1.0), ("F-B", 2, 0.20)])
    backward = _ranking(RankingSource.KG_BACKWARD, [("F-A", 1, 0.10), ("F-B", 2, 0.05)])

    result = fuse_rankings(data, forward, backward)

    assert result.top_fault_id == "F-A"
    assert result.blocking_conflicts == ["HIGH_DATA_SCORE_LOW_BACKWARD_EVIDENCE_COVERAGE"]

    decision = DiagnosisPolicyGate().decide(
        _policy_bundle(),
        blocking_conflicts=result.blocking_conflicts,
        quality_warnings=result.quality_warnings,
    )
    assert decision.status == DiagnosisReleaseStatus.CONDITIONAL_DIAGNOSIS
    assert decision.blocking_reasons == ["EVIDENCE_CONFLICT_DETECTED"]


def test_identical_kg_collapse_can_be_disabled_for_legacy_replay() -> None:
    data = _ranking(RankingSource.DATA_DRIVEN, [("F-A", 1, 0.51), ("F-B", 2, 0.49)])
    forward = _ranking(RankingSource.KG_FORWARD, [("F-B", 1, 1.0), ("F-A", 2, 0.8)])
    backward = _ranking(RankingSource.KG_BACKWARD, [("F-B", 1, 1.0), ("F-A", 2, 0.8)])

    result = fuse_rankings(
        data,
        forward,
        backward,
        config=BiRRFConfig(collapse_identical_kg_rankings=False),
    )

    assert result.metadata["collapsed_kg_source"] is None
    assert set(result.candidates[0].reciprocal_contributions) == {
        "DATA_DRIVEN",
        "KG_FORWARD",
        "KG_BACKWARD",
    }
    assert not any(item.startswith("KG_REDUNDANT_BRANCH_COLLAPSED") for item in result.quality_warnings)


def test_highly_correlated_overlapping_kg_rankings_are_collapsed() -> None:
    data = _ranking(
        RankingSource.DATA_DRIVEN,
        [("F-A", 1, 0.4), ("F-B", 2, 0.3), ("F-C", 3, 0.2), ("F-D", 4, 0.1)],
    )
    forward = _ranking(
        RankingSource.KG_FORWARD,
        [("F-A", 1, 1.0), ("F-B", 2, 0.8), ("F-C", 3, 0.6), ("F-D", 4, 0.4)],
    )
    backward = _ranking(
        RankingSource.KG_BACKWARD,
        [("F-A", 1, 1.0), ("F-B", 2, 0.8), ("F-D", 3, 0.6), ("F-C", 4, 0.4)],
    )

    result = fuse_rankings(
        data,
        forward,
        backward,
        config=BiRRFConfig(kg_redundancy_correlation_threshold=0.75),
    )

    assert result.metadata["kg_rank_orders_identical"] is False
    assert result.metadata["kg_branch_correlation"] == pytest.approx(0.8)
    assert result.metadata["kg_candidate_overlap"] == 1.0
    assert result.metadata["collapsed_kg_source"] == "KG_BACKWARD"
    assert result.metadata["kg_collapse_reason"] == "HIGH_CORRELATION_AND_OVERLAP"


def test_backward_conflict_threshold_uses_raw_coverage_not_composite_score() -> None:
    data = _ranking(
        RankingSource.DATA_DRIVEN,
        [("F-A", 1, 0.90), ("F-B", 2, 0.20)],
        calibrated=True,
    )
    forward = _ranking(RankingSource.KG_FORWARD, [("F-A", 1, 1.0), ("F-B", 2, 0.20)])
    backward = _ranking(RankingSource.KG_BACKWARD, [("F-A", 1, 0.10), ("F-B", 2, 0.05)])
    backward.candidates[0].metadata["coverage"] = 0.90

    result = fuse_rankings(data, forward, backward)

    assert "HIGH_DATA_SCORE_LOW_BACKWARD_EVIDENCE_COVERAGE" not in result.conflict_flags


def test_missing_backward_coverage_blocks_an_unresolved_high_data_top() -> None:
    data = _ranking(
        RankingSource.DATA_DRIVEN,
        [("F-A", 1, 0.90), ("F-B", 2, 0.20)],
        calibrated=True,
    )
    forward = _ranking(RankingSource.KG_FORWARD, [("F-A", 1, 1.0), ("F-B", 2, 0.20)])
    backward = _ranking(RankingSource.KG_BACKWARD, [("F-A", 1, 0.90), ("F-B", 2, 0.20)])
    for item in backward.candidates:
        item.metadata.clear()

    result = fuse_rankings(data, forward, backward)

    assert result.top_fault_id == "F-A"
    assert result.blocking_conflicts == ["BACKWARD_EVIDENCE_COVERAGE_UNAVAILABLE"]


def test_uncalibrated_data_score_never_enters_probability_conflict_gate() -> None:
    data = _ranking(RankingSource.DATA_DRIVEN, [("F-A", 1, 0.99), ("F-B", 2, 0.01)])
    forward = _ranking(RankingSource.KG_FORWARD, [("F-A", 1, 1.0), ("F-B", 2, 0.2)])
    backward = _ranking(RankingSource.KG_BACKWARD, [("F-A", 1, 0.01), ("F-B", 2, 0.0)])

    result = fuse_rankings(data, forward, backward)

    assert result.blocking_conflicts == []
    assert result.metadata["data_conflict_gate_applied"] is False
    assert (
        "DATA_CONFLICT_GATE_SKIPPED_UNCALIBRATED_SCORE=UNCALIBRATED_CLASS_PROBABILITY"
        in result.quality_warnings
    )


def test_shared_cross_branch_lineage_is_capped_to_one_effective_vote() -> None:
    shared = {"F-A": ["SENSOR:battery_voltage"]}
    data = _ranking(
        RankingSource.DATA_DRIVEN,
        [("F-A", 1, 0.8)],
        lineage_complete=True,
        groups_by_fault=shared,
    )
    forward = _ranking(
        RankingSource.KG_FORWARD,
        [("F-A", 1, 1.0)],
        lineage_complete=True,
        groups_by_fault=shared,
    )
    backward = _ranking(
        RankingSource.KG_BACKWARD,
        [("F-A", 1, 1.0)],
        lineage_complete=True,
        groups_by_fault={"F-A": ["SENSOR:independent_current"]},
    )

    result = fuse_rankings(
        data,
        forward,
        backward,
        config=BiRRFConfig(
            backward_weight=0.0,
            collapse_identical_kg_rankings=False,
            collapse_highly_correlated_kg_rankings=False,
        ),
    )
    candidate = result.candidates[0]

    assert candidate.dependency_adjusted is True
    assert candidate.shared_correlation_groups == {
        "SENSOR:battery_voltage": ["DATA_DRIVEN", "KG_FORWARD"]
    }
    assert sum(candidate.reciprocal_contributions.values()) == pytest.approx(
        max(candidate.raw_reciprocal_contributions.values())
    )
    assert result.metadata["evidence_lineage_dependency_adjusted_fault_ids"] == ["F-A"]


def test_formal_lineage_policy_blocks_when_any_active_branch_is_undeclared() -> None:
    data = _ranking(RankingSource.DATA_DRIVEN, [("F-A", 1, 0.8)])
    forward = _ranking(RankingSource.KG_FORWARD, [("F-A", 1, 1.0)])
    backward = _ranking(RankingSource.KG_BACKWARD, [("F-A", 1, 1.0)])

    result = fuse_rankings(
        data,
        forward,
        backward,
        config=BiRRFConfig(require_complete_evidence_lineage=True),
    )

    assert "EVIDENCE_LINEAGE_INCOMPLETE" in result.blocking_conflicts
