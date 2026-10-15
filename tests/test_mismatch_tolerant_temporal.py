from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from astrograph.experiments.mismatch_tolerant_temporal import (
    AlignmentDecision,
    FrozenMismatchTolerantTrace,
    MismatchLocalizationStatus,
    MismatchTolerantAlignmentContract,
    MismatchTolerantTemporalPolicy,
    PrefixAlignmentResult,
    bounded_prefix_alignment,
    generate_mismatch_tolerant_scores,
    select_mismatch_tolerant_trace,
    shift_reference_with_validity,
)
from astrograph.experiments.model_independent_temporal import (
    ModelIndependentScaleContract,
    TraceVariant,
)

SHA = "a" * 64


def scale_contract() -> ModelIndependentScaleContract:
    return ModelIndependentScaleContract(
        contract_id="test-scale",
        dataset_archive_sha256=SHA,
        scale_fit_manifest_sha256="b" * 64,
        scale_fit_groups=(3, 8),
        channel_order=("adcs.x", "adcs.y"),
        channel_scale=(1.0, 1.0),
    )


def alignment_contract(*, prefix_points: int = 4, max_lag: int = 2) -> MismatchTolerantAlignmentContract:
    scale = scale_contract()
    return MismatchTolerantAlignmentContract(
        contract_id="test-alignment",
        dataset_archive_sha256=SHA,
        scale_contract_sha256=scale.contract_sha256,
        prefix_points=prefix_points,
        maximum_absolute_lag_points=max_lag,
        notification_buffer_points=max_lag,
    )


def alignment_result(*, fit: float = 0.01) -> PrefixAlignmentResult:
    return PrefixAlignmentResult(
        selected_lag_points=0,
        fit_score=fit,
        second_best_margin=0.1,
        candidate_lags=(-1, 0, 1),
        candidate_fit_scores=(0.2, fit, 0.3),
        invalid_prefix_points=0,
        invalid_suffix_points=0,
        fit_prefix_valid_points=3,
    )


def trace(*, group: int = 3, fit: float = 0.01) -> FrozenMismatchTolerantTrace:
    contract = alignment_contract(prefix_points=4, max_lag=1)
    return FrozenMismatchTolerantTrace(
        opaque_trajectory_id="TRJ-" + "A" * 20,
        independent_group_id="GRP-" + "B" * 16,
        development_group=group,
        variant=TraceVariant.EXACT_PAIRED,
        current_telemetry_sha256="c" * 64,
        reference_telemetry_sha256="d" * 64,
        alternate_reference_sha256="e" * 64,
        scale_contract_sha256=scale_contract().contract_sha256,
        alignment_contract_sha256=contract.contract_sha256,
        time_s=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        trigger_score=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        source_missing_fraction=(0.0,) * 6,
        alignment_valid=(True,) * 6,
        prefix_baseline_score=0.0,
        alignment=alignment_result(fit=fit),
    )


def test_bounded_alignment_recovers_known_reference_delay():
    scale = scale_contract()
    contract = alignment_contract(prefix_points=5, max_lag=2)
    reference = np.column_stack((np.arange(10.0), np.arange(10.0) * 2.0))
    current = reference.copy()
    delayed, _ = shift_reference_with_validity(reference, 2)
    aligned, valid, result = bounded_prefix_alignment(current, delayed, scale, contract)
    assert result.selected_lag_points == -2
    assert result.fit_score == 0.0
    assert result.invalid_suffix_points == 2
    assert valid[-2:].tolist() == [False, False]
    assert np.array_equal(aligned[:-2], reference[:-2])


def test_alignment_is_prefix_only_and_tail_invariant():
    scale = scale_contract()
    contract = alignment_contract(prefix_points=5, max_lag=2)
    reference = np.column_stack((np.arange(10.0), np.arange(10.0)))
    delayed, _ = shift_reference_with_validity(reference, 1)
    current_a = reference.copy()
    current_b = reference.copy()
    current_b[4:] = 1e6
    result_a = bounded_prefix_alignment(current_a, delayed, scale, contract)[2]
    result_b = bounded_prefix_alignment(current_b, delayed, scale, contract)[2]
    assert result_a.selected_lag_points == result_b.selected_lag_points == -1
    assert result_a.fit_score == result_b.fit_score
    assert result_a.candidate_fit_scores == result_b.candidate_fit_scores


def test_generated_scores_invalidate_padded_edges():
    scale = scale_contract()
    contract = alignment_contract(prefix_points=5, max_lag=2)
    reference = np.column_stack((np.arange(10.0), np.arange(10.0) * 2.0))
    delayed, _ = shift_reference_with_validity(reference, 2)
    score, missing, valid, _, alignment = generate_mismatch_tolerant_scores(
        reference, delayed, scale, contract
    )
    assert alignment.selected_lag_points == -2
    assert valid[-2:].tolist() == [False, False]
    assert score[-2:].tolist() == [0.0, 0.0]
    assert missing[-2:].tolist() == [1.0, 1.0]


def test_runtime_trace_rejects_truth_fields():
    payload = trace().model_dump(mode="python", exclude_computed_fields=True)
    payload["fault_label"] = "forbidden"
    with pytest.raises(ValidationError):
        FrozenMismatchTolerantTrace.model_validate(payload)


def test_groups_4_9_are_rejected():
    with pytest.raises(ValidationError):
        trace(group=4)


def test_alignment_fit_gate_abstains_before_trigger():
    contract = alignment_contract(prefix_points=4, max_lag=1)
    policy = MismatchTolerantTemporalPolicy(
        policy_id="guarded",
        trigger_threshold=0.5,
        trigger_persistence_points=1,
        warmup_points=4,
        alignment_fit_max=0.08,
        notification_buffer_points=1,
    )
    result = select_mismatch_tolerant_trace(trace(fit=0.09), policy, contract)
    assert result.alignment_decision == AlignmentDecision.ABSTAIN_ALIGNMENT_OUT_OF_ENVELOPE
    assert result.status == MismatchLocalizationStatus.ABSTAIN_ALIGNMENT_OUT_OF_ENVELOPE
    assert result.selected_index is None


def test_selected_notification_includes_alignment_buffer():
    contract = alignment_contract(prefix_points=4, max_lag=1)
    policy = MismatchTolerantTemporalPolicy(
        policy_id="buffered",
        trigger_threshold=0.5,
        trigger_persistence_points=1,
        warmup_points=4,
        alignment_fit_max=0.08,
        notification_buffer_points=1,
    )
    result = select_mismatch_tolerant_trace(trace(), policy, contract)
    assert result.status == MismatchLocalizationStatus.SELECTED
    assert result.selected_time_s == 4.0
    assert result.notification_available_time_s == 5.0


def test_invalid_alignment_edge_cannot_trigger():
    contract = alignment_contract(prefix_points=4, max_lag=1)
    payload = trace().model_dump(mode="python", exclude_computed_fields=True)
    payload["trigger_score"] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    payload["source_missing_fraction"] = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    payload["alignment_valid"] = (True, True, True, True, True, False)
    invalid_edge_trace = FrozenMismatchTolerantTrace.model_validate(payload)
    policy = MismatchTolerantTemporalPolicy(
        policy_id="invalid-edge",
        trigger_threshold=0.5,
        trigger_persistence_points=1,
        warmup_points=4,
        alignment_fit_max=0.08,
        notification_buffer_points=1,
    )
    assert invalid_edge_trace.trigger_score[-1] == 0.0
    result = select_mismatch_tolerant_trace(invalid_edge_trace, policy, contract)
    assert result.status == MismatchLocalizationStatus.NO_ALARM
