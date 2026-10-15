from __future__ import annotations

import pytest

from astrograph.assurance.analysis_plan import (
    MultiplicityMethod,
    PrimaryAnalysisPlanV2,
    RouteANonInferiorityMargins,
)
from astrograph.experiments.route_a_protocol import (
    FrozenPolicyPredictionBundle,
    FrozenProbabilityPrediction,
    OfflineTruthRow,
    OfflineTruthTable,
    freeze_route_a_development_selection,
)

H = "a" * 64
SEED = "b" * 64


def prediction(sample: str, group: str, p_a: float) -> FrozenProbabilityPrediction:
    predicted = "A" if p_a >= 0.5 else "B"
    row = FrozenProbabilityPrediction(
        sample_id=sample,
        group_id=group,
        probabilities={"A": p_a, "B": 1.0 - p_a},
        predicted_id=predicted,
        runtime_trace_sha256=H,
    )
    return row.model_copy(update={"row_sha256": row.compute_sha256()})


def bundle(
    candidate_id: str,
    base_policy_id: str,
    threshold: float,
    probabilities: tuple[float, float, float, float],
) -> FrozenPolicyPredictionBundle:
    rows = (
        prediction("g3-a", "3", probabilities[0]),
        prediction("g3-b", "3", probabilities[1]),
        prediction("g8-a", "8", probabilities[2]),
        prediction("g8-b", "8", probabilities[3]),
    )
    item = FrozenPolicyPredictionBundle(
        candidate_id=candidate_id,
        base_policy_id=base_policy_id,
        confidence_threshold=threshold,
        source_tree_sha256=H,
        model_package_sha256=H,
        temporal_policy_sha256=H,
        evidence_package_sha256=("c" * 64 if base_policy_id != "A0" else H),
        fusion_policy_sha256=("d" * 64 if base_policy_id != "A0" else H),
        development_split_sha256=H,
        prediction_generation_code_sha256=H,
        predictions=rows,
    )
    return item.model_copy(update={"bundle_sha256": item.compute_sha256()})


def plan(candidate_ids: tuple[str, ...], thresholds: tuple[float, ...]) -> PrimaryAnalysisPlanV2:
    return PrimaryAnalysisPlanV2(
        plan_id="route-a-dev-plan",
        status="FROZEN",
        independent_unit="MONTE_CARLO_GROUP",
        development_groups=("3", "8"),
        candidate_ids=candidate_ids,
        baseline_candidate_id="A0",
        candidate_thresholds=dict(zip(candidate_ids, thresholds, strict=True)),
        candidate_fusion_policy_sha256s={candidate_id: (H if candidate_id == "A0" else "d" * 64) for candidate_id in candidate_ids},
        candidate_evidence_package_sha256s={candidate_id: (H if candidate_id == "A0" else "c" * 64) for candidate_id in candidate_ids},
        source_tree_sha256=H,
        model_package_sha256=H,
        temporal_policy_sha256=H,
        development_split_sha256=H,
        prediction_generation_code_sha256=H,
        noninferiority_margins=RouteANonInferiorityMargins(
            max_accuracy_drop=0.05,
            max_coverage_drop=0.10,
            max_selective_risk_increase=0.05,
            max_nll_increase=0.10,
            max_brier_increase=0.10,
            max_ece_increase=0.10,
            max_aurc_increase=0.10,
            max_worst_group_accuracy_drop=0.05,
        ),
        confidence_level=0.95,
        bootstrap_replicates=1000,
        bootstrap_seed_sha256=SEED,
        multiplicity_method=MultiplicityMethod.HIERARCHICAL,
    )


def truth_table(*bundles: FrozenPolicyPredictionBundle) -> OfflineTruthTable:
    table = OfflineTruthTable(
        dataset_id="adcs-dev-3-8",
        development_split_sha256=H,
        eligible_prediction_bundle_sha256s=tuple(
            sorted(item.compute_sha256() for item in bundles)
        ),
        rows=(
            OfflineTruthRow(sample_id="g3-a", group_id="3", true_id="A"),
            OfflineTruthRow(sample_id="g3-b", group_id="3", true_id="B"),
            OfflineTruthRow(sample_id="g8-a", group_id="8", true_id="A"),
            OfflineTruthRow(sample_id="g8-b", group_id="8", true_id="B"),
        ),
    )
    return table.model_copy(update={"truth_table_sha256": table.compute_sha256()})


def test_development_receipt_is_content_addressed_and_selects_feasible_gain() -> None:
    # A0 misses every B case; A7 correctly identifies all cases in both groups.
    a0 = bundle("A0", "A0", 0.0, (0.9, 0.6, 0.9, 0.6))
    a7 = bundle("A7_T080", "A7", 0.8, (0.9, 0.1, 0.9, 0.1))
    receipt = freeze_route_a_development_selection(
        analysis_plan=plan(("A0", "A7_T080"), (0.0, 0.8)),
        prediction_bundles=(a0, a7),
        truth_table=truth_table(a0, a7),
    )
    assert receipt.selected_candidate_id == "A7_T080"
    assert receipt.selected_base_policy_id == "A7"
    assert receipt.development_groups == ("3", "8")
    assert receipt.selection["small_group_warning"] is True
    assert receipt.receipt_sha256 == receipt.compute_sha256()


def test_zero_acceptance_cannot_win_through_artificially_low_risk() -> None:
    a0 = bundle("A0", "A0", 0.0, (0.9, 0.1, 0.9, 0.1))
    reject_all = bundle("A7_T099", "A7", 0.99, (0.9, 0.1, 0.9, 0.1))
    receipt = freeze_route_a_development_selection(
        analysis_plan=plan(("A0", "A7_T099"), (0.0, 0.99)),
        prediction_bundles=(a0, reject_all),
        truth_table=truth_table(a0, reject_all),
    )
    assert receipt.selected_candidate_id == "A0"
    checks = receipt.selection["noninferiority_checks"]["A7_T099"]
    assert checks["accepted_coverage"] is False
    assert checks["selective_risk"] is False


def test_prediction_row_hash_tampering_is_rejected() -> None:
    with pytest.raises(ValueError, match="row_sha256"):
        FrozenProbabilityPrediction(
            sample_id="x",
            group_id="3",
            probabilities={"A": 0.8, "B": 0.2},
            predicted_id="A",
            runtime_trace_sha256=H,
            row_sha256="f" * 64,
        )


def test_truth_table_must_bind_exactly_all_frozen_candidate_bundles() -> None:
    a0 = bundle("A0", "A0", 0.0, (0.9, 0.1, 0.9, 0.1))
    a7 = bundle("A7", "A7", 0.0, (0.9, 0.1, 0.9, 0.1))
    incomplete = OfflineTruthTable(
        dataset_id="x",
        development_split_sha256=H,
        eligible_prediction_bundle_sha256s=(a0.compute_sha256(),),
        rows=truth_table(a0, a7).rows,
    )
    with pytest.raises(ValueError, match="exactly"):
        freeze_route_a_development_selection(
            analysis_plan=plan(("A0", "A7"), (0.0, 0.0)),
            prediction_bundles=(a0, a7),
            truth_table=incomplete,
        )


def test_groups_4_9_remain_impossible_in_frozen_plan() -> None:
    with pytest.raises(ValueError, match="post-hoc"):
        PrimaryAnalysisPlanV2(
            plan_id="bad",
            status="FROZEN",
            independent_unit="MONTE_CARLO_GROUP",
            development_groups=("4", "9"),
            candidate_ids=("A0",),
            candidate_thresholds={"A0": 0.0},
            candidate_fusion_policy_sha256s={"A0": H},
            candidate_evidence_package_sha256s={"A0": H},
            source_tree_sha256=H,
            model_package_sha256=H,
            temporal_policy_sha256=H,
            development_split_sha256=H,
            prediction_generation_code_sha256=H,
            noninferiority_margins=RouteANonInferiorityMargins(
                max_accuracy_drop=0.0,
                max_coverage_drop=0.0,
                max_selective_risk_increase=0.0,
                max_nll_increase=0.0,
                max_brier_increase=0.0,
                max_ece_increase=0.0,
                max_aurc_increase=0.0,
                max_worst_group_accuracy_drop=0.0,
            ),
            bootstrap_seed_sha256=SEED,
        )


def test_candidate_threshold_and_policy_hashes_are_frozen_before_truth_join() -> None:
    a0 = bundle("A0", "A0", 0.0, (0.9, 0.1, 0.9, 0.1))
    a7 = bundle("A7_T080", "A7", 0.8, (0.9, 0.1, 0.9, 0.1))
    frozen_plan = plan(("A0", "A7_T080"), (0.0, 0.8))
    tampered = a7.model_copy(update={"fusion_policy_sha256": "f" * 64})
    with pytest.raises(ValueError, match="fusion policy differs"):
        freeze_route_a_development_selection(
            analysis_plan=frozen_plan,
            prediction_bundles=(a0, tampered),
            truth_table=truth_table(a0, tampered),
        )
