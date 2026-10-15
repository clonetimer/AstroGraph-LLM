"""Group-aware split-conformal candidate sets for offline shadow decisions.

Calibration uses the maximum nonconformity score within each independent group,
so the finite-sample statement is group-level under exchangeability of groups.
Empty or multi-label sets become an operational abstention; this module never
changes formal evidence, a deterministic diagnosis, or release qualification.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

from .data_exposure import guard_in_memory_development
from .selective_risk import FrozenPrediction, TruthRecord


def _canonical_sha(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def verify_group_conformal_receipt(receipt: Mapping[str, object]) -> dict:
    payload = dict(receipt)
    if payload.get("schema_version") != "astrograph-group-conformal-candidate-receipt.v1":
        raise ValueError("unsupported group conformal receipt schema")
    claimed = str(payload.get("receipt_sha256", ""))
    if len(claimed) != 64:
        raise ValueError("group conformal receipt SHA256 required")
    canonical = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if _canonical_sha(canonical) != claimed:
        raise ValueError("group conformal receipt SHA256 mismatch")
    calibration = set(map(str, payload.get("calibration_group_ids", [])))
    forbidden = set(map(str, payload.get("forbidden_group_ids", [])))
    if not calibration or calibration & forbidden:
        raise ValueError("group conformal receipt requires disjoint calibration/forbidden groups")
    alpha = float(payload.get("alpha", -1.0))
    threshold = float(payload.get("minimum_label_probability", -1.0))
    if not 0.0 < alpha < 1.0 or not 0.0 <= threshold <= 1.0:
        raise ValueError("group conformal receipt has invalid alpha or threshold")
    return payload


def fit_group_conformal_candidate_receipt(
    predictions: Sequence[FrozenPrediction],
    truths: Sequence[TruthRecord],
    *,
    calibration_group_ids: set[str],
    forbidden_group_ids: set[str],
    alpha: float,
) -> dict:
    """Fit a group-maximum split-conformal candidate threshold."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    calibration = set(map(str, calibration_group_ids))
    forbidden = set(map(str, forbidden_group_ids))
    if not calibration or calibration & forbidden:
        raise ValueError("calibration and forbidden groups must be nonempty/disjoint")
    observed_groups = {prediction.group_id for prediction in predictions}
    if observed_groups != calibration:
        raise ValueError("conformal fitting must use exactly the declared calibration groups")
    truth_by_id = {truth.sample_id: truth for truth in truths}
    if len(truth_by_id) != len(truths) or len(predictions) != len(truths):
        raise ValueError("conformal prediction/truth inputs must be one-to-one")
    scores_by_group: dict[str, list[float]] = {}
    seen: set[str] = set()
    label_universe: set[str] | None = None
    for prediction in predictions:
        if prediction.sample_id in seen:
            raise ValueError("duplicate conformal prediction sample_id")
        seen.add(prediction.sample_id)
        truth = truth_by_id.get(prediction.sample_id)
        if truth is None or truth.group_id != prediction.group_id:
            raise ValueError("conformal prediction/truth identity mismatch")
        labels = set(map(str, prediction.probabilities))
        label_universe = labels if label_universe is None else label_universe
        if labels != label_universe or truth.true_id not in prediction.probabilities:
            raise ValueError("conformal label universe drift or truth label absent")
        score = 1.0 - float(prediction.probabilities[truth.true_id])
        scores_by_group.setdefault(prediction.group_id, []).append(score)
    if set(truth_by_id) != seen:
        raise ValueError("truth contains samples without conformal predictions")
    group_scores = sorted(max(values) for values in scores_by_group.values())
    group_count = len(group_scores)
    quantile_rank = math.ceil((group_count + 1) * (1.0 - alpha))
    # The standard split-conformal +infinity correction maps to score=1 for
    # bounded 1-p scores when the requested rank exceeds the calibration size.
    quantile_score = 1.0 if quantile_rank > group_count else group_scores[quantile_rank - 1]
    provenance = guard_in_memory_development("SELECT", group_ids=sorted(calibration))
    payload = {
        "schema_version": "astrograph-group-conformal-candidate-receipt.v1",
        "method": "GROUP_MAXIMUM_SPLIT_CONFORMAL_1_MINUS_TRUE_LABEL_PROBABILITY",
        "alpha": alpha,
        "calibration_group_ids": sorted(calibration),
        "forbidden_group_ids": sorted(forbidden),
        "calibration_group_count": group_count,
        "calibration_sample_count": len(predictions),
        "label_universe": sorted(label_universe or set()),
        "quantile_rank": quantile_rank,
        "quantile_nonconformity_score": quantile_score,
        "minimum_label_probability": 1.0 - quantile_score,
        "selection_provenance": provenance,
        "guarantee_scope": (
            "Group-level simultaneous candidate-set coverage under exchangeability of independent groups; "
            "no guarantee under arbitrary temporal drift or selective labels."
        ),
        "claim_boundary": (
            "Offline shadow candidate-set calibration only. Empty/multiple sets recommend abstention. "
            "Not formal evidence, production qualification, OOD validity or a flight guarantee."
        ),
    }
    payload["receipt_sha256"] = _canonical_sha(payload)
    return payload


def apply_group_conformal_candidate_receipt(
    prediction: FrozenPrediction,
    receipt: Mapping[str, object],
) -> dict:
    verified = verify_group_conformal_receipt(receipt)
    calibration = set(map(str, verified["calibration_group_ids"]))
    forbidden = set(map(str, verified["forbidden_group_ids"]))
    if prediction.group_id in calibration:
        raise ValueError("calibration group cannot be reused for conformal evaluation")
    if forbidden and prediction.group_id not in forbidden:
        raise ValueError("prediction group is not admitted by the conformal receipt")
    labels = sorted(map(str, prediction.probabilities))
    if labels != list(verified["label_universe"]):
        raise ValueError("prediction label universe differs from conformal receipt")
    threshold = float(verified["minimum_label_probability"])
    candidate_set = sorted(
        label for label, probability in prediction.probabilities.items()
        if float(probability) >= threshold
    )
    abstain = len(candidate_set) != 1
    return {
        "schema_version": "astrograph-group-conformal-candidate-result.v1",
        "sample_id": prediction.sample_id,
        "group_id": prediction.group_id,
        "candidate_ids": candidate_set,
        "operational_candidate_ids": candidate_set if candidate_set else labels,
        "empty_set_fallback_applied": not candidate_set,
        "abstain_recommended": abstain,
        "candidate_set_size": len(candidate_set),
        "minimum_label_probability": threshold,
        "receipt_sha256": verified["receipt_sha256"],
        "formal_evidence_generated": False,
        "claim_boundary": verified["claim_boundary"],
    }


__all__ = [
    "apply_group_conformal_candidate_receipt",
    "fit_group_conformal_candidate_receipt",
    "verify_group_conformal_receipt",
]
