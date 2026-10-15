from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from astrograph.experiments.model_independent_temporal import (
    ModelIndependentScaleContract,
    ModelIndependentTruthRow,
    ModelIndependentTruthTable,
)
from astrograph.experiments.reference_set_temporal import (
    FrozenReferenceLibrary,
    FrozenReferenceSetTrace,
    FrozenReferenceSetTraceBundle,
    ReferenceLibraryEntry,
    ReferenceSetDecision,
    ReferenceSetLocalizationStatus,
    ReferenceSetTemporalContract,
    ReferenceSetTemporalPolicy,
    ReferenceSetVariant,
    evaluate_reference_set_policy,
    generate_reference_set_scores,
    prepare_reference_set_candidates,
    generate_reference_set_scores_from_prepared,
    select_reference_set_trace,
)

SHA = "a" * 64
SHA2 = "b" * 64


def scale_contract() -> ModelIndependentScaleContract:
    return ModelIndependentScaleContract(
        contract_id="SCALE-TEST",
        dataset_archive_sha256=SHA,
        scale_fit_manifest_sha256=SHA2,
        scale_fit_groups=(0, 1),
        channel_order=("adcs.a", "adcs.b"),
        channel_scale=(1.0, 1.0),
    )


def entry(index: int, group: str) -> ReferenceLibraryEntry:
    return ReferenceLibraryEntry(
        reference_id=f"REF-{index:020X}",
        reference_family_id="FAM-0000000000000001",
        associated_independent_group_id=group,
        telemetry_relative_path=f"telemetry/ref_{index}.csv",
        telemetry_sha256=f"{index + 1:064x}"[-64:],
        row_count=40,
        channel_count=2,
    )


def library() -> FrozenReferenceLibrary:
    entries = (
        entry(1, "GRP-0000000000000001"),
        entry(2, "GRP-0000000000000002"),
        entry(3, "GRP-0000000000000003"),
    )
    return FrozenReferenceLibrary(
        library_id="LIB-TEST",
        dataset_archive_sha256=SHA,
        reference_source_manifest_sha256="c" * 64,
        scale_contract_sha256=scale_contract().contract_sha256,
        channel_order=("adcs.a", "adcs.b"),
        entries=entries,
    )


def temporal_contract(lib: FrozenReferenceLibrary | None = None) -> ReferenceSetTemporalContract:
    lib = lib or library()
    return ReferenceSetTemporalContract(
        contract_id="TEMP-TEST",
        dataset_archive_sha256=SHA,
        runtime_manifest_sha256="d" * 64,
        scale_contract_sha256=scale_contract().contract_sha256,
        reference_library_sha256=lib.library_sha256,
        prefix_points=10,
        maximum_absolute_lag_points=2,
        minimum_candidate_references=2,
        notification_buffer_points=2,
        level_half_width_multiplier=2.0,
        gradient_half_width_multiplier=2.0,
    )


def matrices() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    t = np.arange(40, dtype=float)
    base = np.column_stack((0.1 * t, np.sin(t / 8.0)))
    refs = {
        "REF-00000000000000000001": base + np.array([0.0, 0.01]),
        "REF-00000000000000000002": base + np.array([0.02, -0.01]),
        "REF-00000000000000000003": base + np.array([-0.02, 0.0]),
    }
    return base.copy(), refs


def make_trace(*, fit: float = 0.01, selected_signal: bool = True) -> FrozenReferenceSetTrace:
    lib = library()
    contract = temporal_contract(lib)
    current, refs = matrices()
    if selected_signal:
        current[20:, 0] += 4.0
    level, gradient, missing, valid, lb, gb, diagnostics = generate_reference_set_scores(
        current,
        refs,
        lib,
        scale_contract(),
        contract,
        target_independent_group_id="GRP-0000000000000001",
    )
    diagnostics = tuple(item.model_copy(update={"fit_score": fit}) for item in diagnostics)
    return FrozenReferenceSetTrace(
        opaque_trajectory_id="TRJ-00000000000000000001",
        independent_group_id="GRP-0000000000000001",
        development_group=3,
        variant=ReferenceSetVariant.LEAVE_ONE_GROUP_OUT,
        current_telemetry_sha256="e" * 64,
        scale_contract_sha256=scale_contract().contract_sha256,
        reference_library_sha256=lib.library_sha256,
        temporal_contract_sha256=contract.contract_sha256,
        time_s=tuple(float(i) for i in range(40)),
        level_score=tuple(float(x) for x in level),
        gradient_score=tuple(float(x) for x in gradient),
        source_missing_fraction=tuple(float(x) for x in missing),
        score_valid=tuple(bool(x) for x in valid),
        level_prefix_baseline=lb,
        gradient_prefix_baseline=gb,
        best_reference_fit_score=fit,
        median_reference_fit_score=fit,
        maximum_reference_fit_score=fit,
        candidate_alignments=diagnostics,
    )


def test_library_rejects_duplicate_provenance_groups() -> None:
    first = entry(1, "GRP-0000000000000001")
    second = entry(2, "GRP-0000000000000001")
    with pytest.raises(ValidationError, match="unique provenance group"):
        FrozenReferenceLibrary(
            library_id="BAD",
            dataset_archive_sha256=SHA,
            reference_source_manifest_sha256="c" * 64,
            scale_contract_sha256=scale_contract().contract_sha256,
            channel_order=("adcs.a", "adcs.b"),
            entries=(first, second),
        )


def test_reference_path_traversal_is_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot traverse"):
        entry(1, "GRP-0000000000000001").model_copy(
            update={"telemetry_relative_path": "../secret.csv"}
        ).model_validate(
            entry(1, "GRP-0000000000000001").model_dump()
            | {"telemetry_relative_path": "../secret.csv"}
        )


def test_contract_rejects_posthoc_development_groups() -> None:
    lib = library()
    with pytest.raises(ValidationError, match="groups 3/8"):
        ReferenceSetTemporalContract(
            contract_id="BAD",
            dataset_archive_sha256=SHA,
            runtime_manifest_sha256="d" * 64,
            scale_contract_sha256=scale_contract().contract_sha256,
            reference_library_sha256=lib.library_sha256,
            development_groups=(4, 9),
            prefix_points=10,
            maximum_absolute_lag_points=2,
            minimum_candidate_references=2,
            notification_buffer_points=2,
        )


def test_target_provenance_group_is_excluded() -> None:
    lib = library()
    current, refs = matrices()
    _, _, diagnostics = prepare_reference_set_candidates(
        current,
        refs,
        lib,
        scale_contract(),
        temporal_contract(lib),
        target_independent_group_id="GRP-0000000000000001",
    )
    assert {item.associated_independent_group_id for item in diagnostics} == {
        "GRP-0000000000000002",
        "GRP-0000000000000003",
    }


def test_post_prefix_change_cannot_change_alignment_diagnostics() -> None:
    lib = library()
    current, refs = matrices()
    changed = current.copy()
    changed[15:, 0] += 1000.0
    first = prepare_reference_set_candidates(
        current, refs, lib, scale_contract(), temporal_contract(lib),
        target_independent_group_id="GRP-0000000000000001",
    )[2]
    second = prepare_reference_set_candidates(
        changed, refs, lib, scale_contract(), temporal_contract(lib),
        target_independent_group_id="GRP-0000000000000001",
    )[2]
    assert first == second


def test_prepared_and_direct_score_paths_are_identical() -> None:
    lib = library()
    contract = temporal_contract(lib)
    current, refs = matrices()
    prepared, validity, diagnostics = prepare_reference_set_candidates(
        current, refs, lib, scale_contract(), contract,
        target_independent_group_id="GRP-0000000000000001",
    )
    direct = generate_reference_set_scores(
        current, refs, lib, scale_contract(), contract,
        target_independent_group_id="GRP-0000000000000001",
    )
    cached = generate_reference_set_scores_from_prepared(
        current, prepared, validity, diagnostics, scale_contract(), contract
    )
    for left, right in zip(direct[:6], cached[:6]):
        assert np.array_equal(left, right)
    assert direct[6] == cached[6]


def test_step_change_produces_post_prefix_score_without_prefix_alarm() -> None:
    trace = make_trace()
    assert max(trace.level_score[:10]) == 0.0
    assert max(trace.level_score[20:]) > 0.0


def test_policy_selects_persistent_post_prefix_change() -> None:
    trace = make_trace()
    policy = ReferenceSetTemporalPolicy(
        policy_id="POLICY",
        reference_fit_max=0.08,
        trigger_threshold=0.01,
        trigger_persistence_points=2,
        warmup_points=10,
        gradient_weight=0.0,
        notification_buffer_points=2,
    )
    selection = select_reference_set_trace(trace, policy)
    assert selection.status == ReferenceSetLocalizationStatus.SELECTED
    assert selection.reference_set_decision == ReferenceSetDecision.ADMITTED
    assert selection.selected_index >= 20
    assert selection.notification_available_time_s == selection.selected_time_s + 2


def test_out_of_envelope_fit_abstains() -> None:
    trace = make_trace(fit=0.5)
    policy = ReferenceSetTemporalPolicy(
        policy_id="POLICY",
        reference_fit_max=0.08,
        trigger_threshold=0.01,
        warmup_points=10,
        notification_buffer_points=2,
    )
    selection = select_reference_set_trace(trace, policy)
    assert selection.status == ReferenceSetLocalizationStatus.ABSTAIN_REFERENCE_SET_OUT_OF_ENVELOPE


def test_trace_schema_forbids_truth_fields() -> None:
    trace = make_trace()
    with pytest.raises(ValidationError):
        FrozenReferenceSetTrace.model_validate(trace.model_dump() | {"fault_label": "x"})


def test_trace_rejects_target_reference_in_candidate_set() -> None:
    trace = make_trace()
    bad = trace.candidate_alignments[0].model_copy(
        update={"associated_independent_group_id": trace.independent_group_id}
    )
    with pytest.raises(ValidationError, match="leaked"):
        FrozenReferenceSetTrace.model_validate(
            trace.model_dump(exclude_computed_fields=True)
            | {"candidate_alignments": [bad.model_dump(), trace.candidate_alignments[1].model_dump()]}
        )


def test_offline_evaluator_uses_exact_trajectory_and_group_binding() -> None:
    trace = make_trace()
    bundle = FrozenReferenceSetTraceBundle(
        bundle_id="BUNDLE",
        variant=trace.variant,
        dataset_archive_sha256=SHA,
        runtime_manifest_sha256="d" * 64,
        scale_contract_sha256=trace.scale_contract_sha256,
        reference_library_sha256=trace.reference_library_sha256,
        temporal_contract_sha256=trace.temporal_contract_sha256,
        traces=(trace, trace.model_copy(update={
            "opaque_trajectory_id": "TRJ-00000000000000000002",
            "independent_group_id": "GRP-0000000000000002",
            "development_group": 8,
            "candidate_alignments": tuple(
                item.model_copy(update={"associated_independent_group_id": "GRP-0000000000000003" if i == 0 else "GRP-0000000000000001"})
                for i, item in enumerate(trace.candidate_alignments)
            ),
        })),
    )
    truth = ModelIndependentTruthTable(
        dataset_archive_sha256=SHA,
        rows=(
            ModelIndependentTruthRow(
                opaque_trajectory_id=trace.opaque_trajectory_id,
                independent_group_id=trace.independent_group_id,
                development_group=3,
                is_nominal=False,
                diagnostic_class="fault-a",
                active_start_s=20.0,
                active_end_s=39.0,
                source_metadata_sha256="1" * 64,
                source_labels_sha256="2" * 64,
            ),
            ModelIndependentTruthRow(
                opaque_trajectory_id="TRJ-00000000000000000002",
                independent_group_id="GRP-0000000000000002",
                development_group=8,
                is_nominal=True,
                diagnostic_class="nominal",
                source_metadata_sha256="3" * 64,
                source_labels_sha256="4" * 64,
            ),
        ),
    )
    policy = ReferenceSetTemporalPolicy(
        policy_id="POLICY",
        reference_fit_max=0.08,
        trigger_threshold=0.01,
        trigger_persistence_points=2,
        warmup_points=10,
        gradient_weight=0.0,
        notification_buffer_points=2,
    )
    metrics = evaluate_reference_set_policy(policy, bundle, truth)
    assert metrics.independent_group_count == 2
    assert metrics.fault_trajectory_count == 1

