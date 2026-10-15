from __future__ import annotations

import pytest
from pydantic import ValidationError

from astrograph.experiments.model_independent_temporal import (
    FrozenModelIndependentTrace,
    FrozenTraceBundle,
    LocalizationStatus,
    ModelIndependentScaleContract,
    ModelIndependentTemporalPolicy,
    ModelIndependentTruthRow,
    ModelIndependentTruthTable,
    ReferenceDecision,
    TraceVariant,
    evaluate_model_independent_policy,
    normalized_residual_scores,
    rank_model_independent_policies,
    reference_prefix_statistic,
    select_model_independent_trace,
)

SHA = "a" * 64


def contract() -> ModelIndependentScaleContract:
    return ModelIndependentScaleContract(
        contract_id="test",
        dataset_archive_sha256=SHA,
        scale_fit_manifest_sha256="b" * 64,
        scale_fit_groups=(0, 1, 2, 5, 6, 7),
        channel_order=("adcs.a", "adcs.b"),
        channel_scale=(1.0, 2.0),
    )


def trace(*, trajectory: str = "TRJ-" + "A" * 20, group: int = 3, prefix: float = 0.0, scores=(0.0, 0.0, 1.0, 1.0)) -> FrozenModelIndependentTrace:
    return FrozenModelIndependentTrace(
        opaque_trajectory_id=trajectory,
        independent_group_id="GRP-" + "B" * 16,
        development_group=group,
        variant=TraceVariant.EXACT_PAIRED,
        current_telemetry_sha256="c" * 64,
        reference_telemetry_sha256="d" * 64,
        alternate_reference_sha256="e" * 64,
        scale_contract_sha256=contract().contract_sha256,
        time_s=(0.0, 1.0, 2.0, 3.0),
        trigger_score=tuple(scores),
        source_missing_fraction=(0.0, 0.0, 0.0, 0.0),
        reference_prefix_points=2,
        reference_prefix_quantile=0.95,
        reference_prefix_statistic=prefix,
    )


def test_normalized_residual_score_and_missingness_are_deterministic():
    score, missing = normalized_residual_scores(
        [[0.0, 0.0], [1.0, None]],
        [[0.0, 0.0], [0.0, 2.0]],
        contract(),
    )
    assert score.tolist() == [0.0, 10.0]
    assert missing.tolist() == [0.0, 0.5]


def test_runtime_trace_rejects_truth_fields():
    payload = trace().model_dump(mode="python", exclude_computed_fields=True)
    payload["fault_label"] = "forbidden"
    with pytest.raises(ValidationError):
        FrozenModelIndependentTrace.model_validate(payload)


def test_groups_4_9_are_rejected_from_development_traces():
    with pytest.raises(ValidationError):
        trace(group=4)


def test_reference_guard_abstains_before_trigger_evaluation():
    policy = ModelIndependentTemporalPolicy(
        policy_id="guarded",
        trigger_threshold=0.5,
        trigger_persistence_points=1,
        warmup_points=0,
        reference_prefix_max=0.05,
    )
    result = select_model_independent_trace(trace(prefix=0.051), policy)
    assert result.reference_decision == ReferenceDecision.ABSTAIN_REFERENCE_OUT_OF_ENVELOPE
    assert result.status == LocalizationStatus.ABSTAIN_REFERENCE_OUT_OF_ENVELOPE
    assert result.selected_index is None


def test_first_stable_selection_is_causal_and_prefix_invariant():
    policy = ModelIndependentTemporalPolicy(
        policy_id="causal",
        trigger_threshold=0.5,
        trigger_persistence_points=2,
        warmup_points=0,
        reference_prefix_max=0.05,
    )
    short = trace(scores=(0.0, 0.0, 1.0, 1.0))
    long = FrozenModelIndependentTrace(
        **{
            **short.model_dump(mode="python", exclude_computed_fields=True),
            "time_s": (0.0, 1.0, 2.0, 3.0, 4.0),
            "trigger_score": (0.0, 0.0, 1.0, 1.0, 100.0),
            "source_missing_fraction": (0.0,) * 5,
        }
    )
    assert select_model_independent_trace(short, policy).selected_index == 3
    assert select_model_independent_trace(long, policy).selected_index == 3


def test_truth_table_is_separate_and_requires_fault_interval():
    with pytest.raises(ValidationError):
        ModelIndependentTruthRow(
            opaque_trajectory_id="TRJ-" + "A" * 20,
            independent_group_id="GRP-" + "B" * 16,
            development_group=3,
            is_nominal=False,
            diagnostic_class="fault",
            source_metadata_sha256="f" * 64,
            source_labels_sha256="0" * 64,
        )


def test_evaluator_uses_independent_groups_and_reports_timing():
    fault_trace = trace()
    nominal_trace = FrozenModelIndependentTrace(
        **{
            **trace(trajectory="TRJ-" + "C" * 20, scores=(0.0, 0.0, 0.0, 0.0)).model_dump(mode="python", exclude_computed_fields=True),
            "independent_group_id": "GRP-" + "D" * 16,
        }
    )
    bundle = FrozenTraceBundle(
        bundle_id="test",
        variant=TraceVariant.EXACT_PAIRED,
        dataset_archive_sha256=SHA,
        runtime_manifest_sha256="1" * 64,
        scale_contract_sha256=contract().contract_sha256,
        development_groups=(3, 8),
        traces=(
            fault_trace,
            nominal_trace,
            trace(trajectory="TRJ-" + "E" * 20, group=8),
        ),
    )
    truth = ModelIndependentTruthTable(
        dataset_archive_sha256=SHA,
        rows=(
            ModelIndependentTruthRow(
                opaque_trajectory_id=fault_trace.opaque_trajectory_id,
                independent_group_id=fault_trace.independent_group_id,
                development_group=3,
                is_nominal=False,
                diagnostic_class="fault_a",
                active_start_s=2.0,
                active_end_s=3.0,
                source_metadata_sha256="2" * 64,
                source_labels_sha256="3" * 64,
            ),
            ModelIndependentTruthRow(
                opaque_trajectory_id=nominal_trace.opaque_trajectory_id,
                independent_group_id=nominal_trace.independent_group_id,
                development_group=3,
                is_nominal=True,
                diagnostic_class="nominal",
                source_metadata_sha256="4" * 64,
                source_labels_sha256="5" * 64,
            ),
            ModelIndependentTruthRow(
                opaque_trajectory_id="TRJ-" + "E" * 20,
                independent_group_id="GRP-" + "B" * 16,
                development_group=8,
                is_nominal=False,
                diagnostic_class="fault_a",
                active_start_s=2.0,
                active_end_s=3.0,
                source_metadata_sha256="6" * 64,
                source_labels_sha256="7" * 64,
            ),
        ),
    )
    policy = ModelIndependentTemporalPolicy(
        policy_id="test-policy",
        trigger_threshold=0.5,
        trigger_persistence_points=1,
        warmup_points=0,
    )
    metrics = evaluate_model_independent_policy(policy, bundle, truth)
    assert metrics.independent_group_count == 2
    assert metrics.detected_within_5s_rate == 1.0
    assert metrics.nominal_false_alarm_rate == 0.0
    assert metrics.pre_onset_selection_count == 0


def test_candidate_ranking_does_not_read_stress_variants():
    fault = trace()
    other = trace(trajectory="TRJ-" + "E" * 20, group=8)
    bundle = FrozenTraceBundle(
        bundle_id="canonical",
        variant=TraceVariant.EXACT_PAIRED,
        dataset_archive_sha256=SHA,
        runtime_manifest_sha256="1" * 64,
        scale_contract_sha256=contract().contract_sha256,
        traces=(fault, other),
    )
    truth = ModelIndependentTruthTable(
        dataset_archive_sha256=SHA,
        rows=(
            ModelIndependentTruthRow(
                opaque_trajectory_id=fault.opaque_trajectory_id,
                independent_group_id=fault.independent_group_id,
                development_group=3,
                is_nominal=False,
                diagnostic_class="fault",
                active_start_s=2.0,
                active_end_s=3.0,
                source_metadata_sha256="2" * 64,
                source_labels_sha256="3" * 64,
            ),
            ModelIndependentTruthRow(
                opaque_trajectory_id=other.opaque_trajectory_id,
                independent_group_id=other.independent_group_id,
                development_group=8,
                is_nominal=False,
                diagnostic_class="fault",
                active_start_s=2.0,
                active_end_s=3.0,
                source_metadata_sha256="4" * 64,
                source_labels_sha256="5" * 64,
            ),
        ),
    )
    policies = (
        ModelIndependentTemporalPolicy(policy_id="slow", trigger_threshold=0.5, trigger_persistence_points=2, warmup_points=0),
        ModelIndependentTemporalPolicy(policy_id="fast", trigger_threshold=0.5, trigger_persistence_points=1, warmup_points=0),
    )
    ranked = rank_model_independent_policies(policies, bundle, truth)
    assert ranked[0].policy.policy_id == "fast"


def test_reference_prefix_statistic_is_label_free_numeric_summary():
    assert reference_prefix_statistic([0.0, 0.01, 0.02, 99.0], prefix_points=3, quantile=0.5) == 0.01
