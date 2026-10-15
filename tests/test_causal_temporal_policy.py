from __future__ import annotations

import pytest
from pydantic import ValidationError

from astrograph.diagnosis.causal_temporal import (
    CausalTemporalPolicy,
    CausalTemporalSelector,
    PairedResidualScoringContract,
    TemporalScorePoint,
    TemporalSelectionStatus,
    TemporalTriggerSource,
    paired_residual_trigger_score,
)
from astrograph.experiments.temporal_policy import (
    TemporalDevelopmentTrajectory,
    evaluate_temporal_policy,
    freeze_temporal_policy,
    rank_temporal_policies,
)


def _point(
    sequence_index: int,
    *,
    endpoint_time_s: float | None = None,
    trigger_score: float = 1.0,
    top_fault_id: str = "F1",
    top_fault_score: float = 0.8,
    runner_up_fault_score: float = 0.2,
    source_missing_fraction: float = 0.0,
) -> TemporalScorePoint:
    endpoint = 19 + sequence_index * 5
    return TemporalScorePoint(
        sequence_index=sequence_index,
        endpoint_index=endpoint,
        endpoint_time_s=float(endpoint if endpoint_time_s is None else endpoint_time_s),
        trigger_score=trigger_score,
        detection_score=0.9,
        top_fault_id=top_fault_id,
        top_fault_score=top_fault_score,
        runner_up_fault_score=runner_up_fault_score,
        prediction_id=f"prediction-{sequence_index}",
        input_data_sha256=f"{sequence_index + 1:064x}",
        source_missing_fraction=source_missing_fraction,
    )


def _policy(**updates) -> CausalTemporalPolicy:
    values = {
        "policy_id": "causal-policy-test",
        "trigger_source": TemporalTriggerSource.EXTERNAL_CAUSAL_SCORE,
        "window_length_samples": 20,
        "stride_samples": 5,
        "trigger_threshold": 0.5,
        "trigger_persistence_points": 1,
        "attribution_persistence_points": 1,
        "maximum_endpoint_gap_samples": 5,
    }
    values.update(updates)
    return CausalTemporalPolicy(**values)


def test_paired_residual_score_is_deterministic_and_has_explicit_missing_semantics():
    contract = PairedResidualScoringContract(
        contract_id="paired-residual-test",
        channel_order=("a", "b"),
        channel_scale=(2.0, 1.0),
        top_k_elements=2,
        one_sided_missing_score=7.0,
    )
    current = [[2.0, None], [4.0, None]]
    nominal = [[0.0, None], [0.0, 3.0]]

    # Normalized residuals are [1, 0, 2, 7]; the top-two mean is 4.5.
    assert paired_residual_trigger_score(current, nominal, contract) == pytest.approx(4.5)
    assert contract.contract_sha256 == contract.contract_sha256
    with pytest.raises(ValueError, match="channel count"):
        paired_residual_trigger_score([[1.0]], [[0.0]], contract)


def test_selector_uses_first_stable_prefix_and_future_suffix_cannot_change_it():
    policy = _policy(trigger_persistence_points=2, attribution_persistence_points=2)
    prefix = (
        _point(0, trigger_score=0.1),
        _point(1, top_fault_id="F1"),
        _point(2, top_fault_id="F1"),
    )
    selected = CausalTemporalSelector.select(policy, prefix)
    extended = CausalTemporalSelector.select(
        policy,
        (*prefix, _point(3, top_fault_id="F2"), _point(4, top_fault_id="F2")),
    )

    assert selected.status == TemporalSelectionStatus.SELECTED
    assert selected.selected_point == prefix[2]
    assert extended.selected_point == selected.selected_point
    assert extended.trace_sha256 == selected.trace_sha256
    assert extended.points_consumed == selected.points_consumed == 3


def test_gap_and_failed_gate_reset_persistence_and_selection_delay_is_causal():
    policy = _policy(
        trigger_persistence_points=2,
        attribution_persistence_points=2,
        selection_delay_points=1,
    )
    selector = CausalTemporalSelector(policy)
    selector.update(_point(0))
    # A discontinuity larger than maximum_endpoint_gap_samples resets the run.
    selector.update(_point(1).model_copy(update={"endpoint_index": 30, "endpoint_time_s": 30.0}))
    assert selector.selected_point is None
    selector.update(_point(2).model_copy(update={"endpoint_index": 35, "endpoint_time_s": 35.0}))
    assert selector.selected_point is None
    selector.update(_point(3).model_copy(update={"endpoint_index": 40, "endpoint_time_s": 40.0}))
    assert selector.selected_point is not None
    assert selector.selected_point.endpoint_index == 40
    assert selector.result().reset_count == 1


def test_no_alarm_and_truth_fields_are_not_part_of_the_selector_contract():
    policy = _policy(warmup_points=2)
    insufficient = CausalTemporalSelector.select(policy, (_point(0, trigger_score=0.0),))
    assert insufficient.status == TemporalSelectionStatus.INSUFFICIENT_OBSERVATIONS
    no_alarm = CausalTemporalSelector.select(
        policy,
        (_point(0, trigger_score=0.0), _point(1, trigger_score=0.0), _point(2, trigger_score=0.0)),
    )
    assert no_alarm.status == TemporalSelectionStatus.NO_ALARM
    with pytest.raises(ValidationError):
        TemporalScorePoint.model_validate(
            {**_point(0).model_dump(mode="json"), "truth_fault_id": "F1"}
        )


def test_temporal_metrics_require_both_correct_time_and_correct_fault_identity():
    policy = _policy()
    trajectories = (
        TemporalDevelopmentTrajectory(
            trajectory_id="pre-onset",
            group_id="g1",
            truth_fault_id="F1",
            active_start_s=10.0,
            active_end_s=20.0,
            points=(_point(0, endpoint_time_s=5.0, top_fault_id="F1"),),
        ),
        TemporalDevelopmentTrajectory(
            trajectory_id="wrong-class",
            group_id="g2",
            truth_fault_id="F1",
            active_start_s=10.0,
            active_end_s=20.0,
            points=(_point(0, endpoint_time_s=12.0, top_fault_id="F2"),),
        ),
        TemporalDevelopmentTrajectory(
            trajectory_id="correct",
            group_id="g3",
            truth_fault_id="F1",
            active_start_s=10.0,
            active_end_s=20.0,
            points=(_point(0, endpoint_time_s=12.0, top_fault_id="F1"),),
        ),
        TemporalDevelopmentTrajectory(
            trajectory_id="nominal-alarm",
            group_id="g4",
            points=(_point(0, endpoint_time_s=12.0, top_fault_id="F1"),),
        ),
    )

    metrics = evaluate_temporal_policy(policy, trajectories)

    assert metrics.selected_fault_trajectory_count == 3
    assert metrics.selected_within_active_count == 2
    assert metrics.pre_onset_selection_count == 1
    assert metrics.correct_identification_count == 1
    assert metrics.selected_classification_correct_count == 2
    assert metrics.end_to_end_identification_accuracy == pytest.approx(1 / 3)
    assert metrics.selected_identification_accuracy == pytest.approx(1 / 3)
    assert metrics.selected_classification_accuracy == pytest.approx(2 / 3)
    assert metrics.nominal_false_alarm_count == 1
    assert metrics.nominal_false_alarm_group_count == 1
    assert metrics.group_macro_end_to_end_accuracy == pytest.approx(1 / 3)


def test_policy_ranking_is_group_first_and_freezing_rejects_unsorted_or_overlapping_splits():
    conservative = _policy(policy_id="conservative", trigger_threshold=1.5)
    permissive = _policy(policy_id="permissive", trigger_threshold=0.5)
    trajectories = (
        TemporalDevelopmentTrajectory(
            trajectory_id="fault",
            group_id="g1",
            truth_fault_id="F1",
            active_start_s=10.0,
            active_end_s=20.0,
            points=(_point(0, endpoint_time_s=12.0, trigger_score=1.0),),
        ),
        TemporalDevelopmentTrajectory(
            trajectory_id="nominal",
            group_id="g2",
            points=(_point(0, endpoint_time_s=12.0, trigger_score=1.0),),
        ),
    )
    ranked = rank_temporal_policies((permissive, conservative), trajectories)
    assert ranked[0].policy.policy_id == "conservative"

    common = dict(
        model_manifest_id="model@1",
        model_artifact_sha256="a" * 64,
        development_dataset_id="dataset",
        development_dataset_sha256="b" * 64,
        development_split_id="validation",
        development_split_sha256="c" * 64,
        candidate_grid_sha256="d" * 64,
        trigger_contract_id="trigger",
        trigger_contract_sha256="e" * 64,
        evaluation_set_status="DEVELOPMENT_ONLY",
    )
    with pytest.raises(ValueError, match="ranked order"):
        freeze_temporal_policy(
            ranked_candidates=tuple(reversed(ranked)),
            excluded_evaluation_splits=("test",),
            **common,
        )
    with pytest.raises(ValueError, match="development split"):
        freeze_temporal_policy(
            ranked_candidates=ranked,
            excluded_evaluation_splits=("validation",),
            **common,
        )
    receipt = freeze_temporal_policy(
        ranked_candidates=ranked,
        excluded_evaluation_splits=("test",),
        **common,
    )
    assert receipt.selected_candidate_rank == 1
    assert receipt.policy.policy_id == "conservative"
    assert len(receipt.receipt_sha256) == 64


def test_duplicate_trajectory_ids_are_rejected():
    policy = _policy()
    item = TemporalDevelopmentTrajectory(
        trajectory_id="duplicate",
        group_id="g1",
        truth_fault_id="F1",
        active_start_s=0.0,
        active_end_s=100.0,
        points=(_point(0),),
    )
    with pytest.raises(ValueError, match="duplicate trajectory_id"):
        evaluate_temporal_policy(policy, (item, item))
