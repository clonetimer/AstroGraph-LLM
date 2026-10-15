from __future__ import annotations

import pytest

from astrograph.evaluation.conformal_candidates import (
    apply_group_conformal_candidate_receipt,
    fit_group_conformal_candidate_receipt,
    verify_group_conformal_receipt,
)
from astrograph.evaluation.selective_risk import FrozenPrediction, TruthRecord


def _prediction(sample: str, group: str, a: float) -> FrozenPrediction:
    probabilities = {"A": a, "B": 1.0 - a}
    return FrozenPrediction(
        sample_id=sample,
        group_id=group,
        probabilities=probabilities,
        predicted_id=max(probabilities, key=probabilities.get),
        prediction_sha256=(sample * 64)[:64],
    )


def test_group_conformal_receipt_is_content_bound_and_recommends_abstention() -> None:
    predictions = [
        _prediction("a", "cal-1", 0.9),
        _prediction("b", "cal-2", 0.8),
        _prediction("c", "cal-3", 0.7),
    ]
    truths = [
        TruthRecord("a", "cal-1", "A"),
        TruthRecord("b", "cal-2", "A"),
        TruthRecord("c", "cal-3", "A"),
    ]
    receipt = fit_group_conformal_candidate_receipt(
        predictions,
        truths,
        calibration_group_ids={"cal-1", "cal-2", "cal-3"},
        forbidden_group_ids={"eval-1"},
        alpha=0.5,
    )
    assert verify_group_conformal_receipt(receipt) == receipt
    result = apply_group_conformal_candidate_receipt(
        _prediction("d", "eval-1", 0.55), receipt,
    )
    assert result["formal_evidence_generated"] is False
    assert result["abstain_recommended"] is True
    assert result["candidate_ids"] == []
    assert result["operational_candidate_ids"] == ["A", "B"]
    assert result["empty_set_fallback_applied"] is True


def test_small_calibration_with_strict_alpha_falls_back_to_full_set() -> None:
    receipt = fit_group_conformal_candidate_receipt(
        [_prediction("a", "cal-1", 0.99)],
        [TruthRecord("a", "cal-1", "A")],
        calibration_group_ids={"cal-1"},
        forbidden_group_ids={"eval-1"},
        alpha=0.05,
    )
    assert receipt["minimum_label_probability"] == 0.0
    result = apply_group_conformal_candidate_receipt(
        _prediction("b", "eval-1", 0.99), receipt,
    )
    assert result["candidate_ids"] == ["A", "B"]
    assert result["abstain_recommended"] is True


def test_group_conformal_receipt_rejects_tamper_and_group_reuse() -> None:
    receipt = fit_group_conformal_candidate_receipt(
        [_prediction("a", "cal-1", 0.9), _prediction("b", "cal-2", 0.8)],
        [TruthRecord("a", "cal-1", "A"), TruthRecord("b", "cal-2", "A")],
        calibration_group_ids={"cal-1", "cal-2"},
        forbidden_group_ids={"eval-1"},
        alpha=0.5,
    )
    tampered = {**receipt, "alpha": 0.25}
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_group_conformal_receipt(tampered)
    with pytest.raises(ValueError, match="cannot be reused"):
        apply_group_conformal_candidate_receipt(_prediction("c", "cal-1", 0.7), receipt)
