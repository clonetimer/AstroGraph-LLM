from __future__ import annotations

import numpy as np

from scripts.experiments.run_ops_sat_deterministic_bundle_comparator import (
    nested_oof_policy,
    exact_mcnemar_two_sided,
    select_f05_threshold,
)


def test_f05_threshold_prefers_lower_false_positive_on_tie() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=int)
    probabilities = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=float)
    threshold, metrics = select_f05_threshold(labels, probabilities)
    assert threshold > 0.4
    assert metrics["false_positive_count"] == 0
    assert metrics["true_positive_count"] == 2


def test_nested_policy_returns_out_of_fold_decisions_for_every_case() -> None:
    rng = np.random.default_rng(42)
    labels = np.asarray(([0] * 20) + ([1] * 20), dtype=int)
    signal = labels.astype(float) + rng.normal(0.0, 0.1, labels.size)
    matrix = np.column_stack([signal, rng.normal(size=(labels.size, 5))])
    result = nested_oof_policy(matrix, labels, outer_folds=4, inner_folds=3, seed=42)
    assert result["probabilities"].shape == labels.shape
    assert result["predictions"].shape == labels.shape
    assert len(result["fold_receipts"]) == 4
    assert set(result["selected_candidate_ids"]) <= {*OPS_SINGLE_IDS, "ALL_REGISTERED_PROBES"}


def test_nested_policy_accepts_declared_channel_features_without_label_access() -> None:
    rng = np.random.default_rng(7)
    labels = np.asarray(([0] * 20) + ([1] * 20), dtype=int)
    probe = labels.astype(float) + rng.normal(0.0, 0.2, labels.size)
    channel = np.tile(np.asarray([[1.0, 0.0], [0.0, 1.0]]), (20, 1))
    matrix = np.column_stack([probe, channel])
    result = nested_oof_policy(
        matrix,
        labels,
        outer_folds=4,
        inner_folds=3,
        feature_names=("probe", "channel::A", "channel::B"),
        feature_sets={"PROBE_ONLY": [0], "CHANNEL_PLUS_PROBE": [0, 1, 2]},
    )
    assert result["predictions"].shape == labels.shape
    assert set(result["selected_candidate_ids"]) <= {"PROBE_ONLY", "CHANNEL_PLUS_PROBE"}


OPS_SINGLE_IDS = {
    "early_late_shift",
    "log_volatility_ratio",
    "peak_concentration_delta",
    "normalized_linear_trend",
    "lag1_autocorrelation_delta",
    "high_frequency_energy_delta",
}


def test_exact_mcnemar_reports_small_paired_comparison_as_inconclusive() -> None:
    assert exact_mcnemar_two_sided(4, 0) == 0.125
    assert exact_mcnemar_two_sided(0, 0) == 1.0
