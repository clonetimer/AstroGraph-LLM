from __future__ import annotations

import numpy as np
import pytest

from astrograph.evaluation.ops_sat_probe_utility import (
    OPS_SAT_PROBE_IDS,
    OPS_SAT_LENGTH_FREE_FEATURE_IDS,
    measure_ops_sat_probe,
    measure_ops_sat_probe_bundle,
    measure_ops_sat_official_feature_bundle,
    measure_ops_sat_length_free_feature_bundle,
)
from scripts.experiments.run_ops_sat_probe_utility_audit import (
    cross_validated_probabilities,
    score_probabilities,
)


def _points(values: list[float]) -> list[dict]:
    return [
        {
            "channel": "CADC0001",
            "timestamp": f"2026-01-01T00:00:{index:02d}+00:00",
            "value": value,
            "sampling": "1",
            "segment": "7",
        }
        for index, value in enumerate(values)
    ]


@pytest.mark.parametrize("probe_id", OPS_SAT_PROBE_IDS)
def test_probe_library_is_finite_source_bound_and_non_voting(probe_id: str) -> None:
    observation = measure_ops_sat_probe(
        _points([0.0, 0.1, -0.1, 0.0, 1.0, 1.1, 0.9, 1.0]),
        probe_id,
        source_sha256="a" * 64,
    )
    assert observation["observation_id"].startswith("OPSOBS-")
    assert np.isfinite(observation["value"])
    assert observation["source_sha256"] == "a" * 64
    assert observation["creates_new_raw_observation"] is True
    assert observation["formal_evidence_generated"] is False
    assert observation["formal_evidence_eligible"] is False


def test_probe_library_rejects_unknown_probe_and_identity_drift() -> None:
    points = _points([0.0] * 8)
    with pytest.raises(ValueError, match="UNREGISTERED"):
        measure_ops_sat_probe(points, "read_truth", source_sha256="a" * 64)
    points[-1]["segment"] = "8"
    with pytest.raises(ValueError, match="IDENTITY_DRIFT"):
        measure_ops_sat_probe(points, OPS_SAT_PROBE_IDS[0], source_sha256="a" * 64)


def test_probe_bundle_is_one_source_bound_non_voting_observation() -> None:
    bundle = measure_ops_sat_probe_bundle(
        _points([0.0, 0.1, -0.1, 0.0, 1.0, 1.1, 0.9, 1.0]),
        source_sha256="a" * 64,
    )
    assert bundle["observation_id"].startswith("OPSBUNDLE-")
    assert bundle["component_count"] == len(OPS_SAT_PROBE_IDS)
    assert [row["probe_id"] for row in bundle["components"]] == list(OPS_SAT_PROBE_IDS)
    assert bundle["creates_new_raw_observation"] is True
    assert bundle["formal_evidence_generated"] is False


def test_official_feature_bundle_is_derived_source_bound_and_truth_free() -> None:
    row = {
        "segment": "7", "channel": "CADC0001", "sampling": "1",
        "duration": 8.0, "len": 8.0, "mean": 0.5, "var": 0.25, "std": 0.5,
        "kurtosis": 0.0, "skew": 0.0, "n_peaks": 1.0, "smooth10_n_peaks": 1.0,
        "smooth20_n_peaks": 1.0, "diff_peaks": 1.0, "diff2_peaks": 1.0,
        "diff_var": 0.1, "diff2_var": 0.2, "gaps_squared": 0.0,
        "len_weighted": 4.0, "var_div_duration": 0.03125, "var_div_len": 0.03125,
    }
    observation = measure_ops_sat_official_feature_bundle(row, source_sha256="b" * 64)
    assert observation["observation_id"].startswith("OPSFEATURES-")
    assert observation["creates_new_raw_observation"] is False
    assert observation["creates_new_derived_observation"] is True
    assert observation["formal_evidence_eligible"] is False
    assert set(observation["length_free_feature_ids"]) == set(OPS_SAT_LENGTH_FREE_FEATURE_IDS)
    with pytest.raises(ValueError, match="UNADMITTED"):
        measure_ops_sat_official_feature_bundle({**row, "anomaly": 1}, source_sha256="b" * 64)
    length_free = measure_ops_sat_length_free_feature_bundle(row, source_sha256="b" * 64)
    assert length_free["observation_id"].startswith("OPSLENGTHFREE-")
    assert set(length_free["values"]) == set(OPS_SAT_LENGTH_FREE_FEATURE_IDS)
    assert "duration" not in length_free["values"]


def test_cross_validated_information_gain_detects_separable_signal() -> None:
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
    matrix = np.asarray([[-3.0], [-2.0], [2.0], [3.0], [-4.0], [-1.0], [1.0], [4.0]])
    folds = [
        (np.asarray([0, 2, 4, 6]), np.asarray([1, 3, 5, 7])),
        (np.asarray([1, 3, 5, 7]), np.asarray([0, 2, 4, 6])),
    ]
    probabilities, priors = cross_validated_probabilities(matrix, labels, folds, seed=42)
    metrics = score_probabilities(labels, probabilities, priors)
    assert metrics["information_gain_bits_per_case"] > 0.0
    assert metrics["balanced_accuracy_at_0_5"] > 0.5
