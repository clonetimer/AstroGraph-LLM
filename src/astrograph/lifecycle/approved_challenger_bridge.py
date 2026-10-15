"""Bridge reviewed lifecycle improvement candidates to the existing offline challenger evaluator."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from astrograph.data_driven.controlled_evolution import run_controlled_evolution_cycle
from astrograph.data_driven.feedback import FeedbackCandidate
from .knowledge_feedback import KnowledgeFeedbackStore


def run_approved_model_challenger(
    *,
    store: KnowledgeFeedbackStore,
    improvement_candidate_id: str,
    root: str | Path,
    dataset_id: str,
    baseline: FeedbackCandidate,
    challenger: FeedbackCandidate,
    failure_regression_root: str | Path,
    predictor: Callable[[Any], str],
    split_audit_ref: str = "",
) -> dict[str, Any]:
    candidate = store.export_candidate(improvement_candidate_id)
    if candidate.get("status") != "APPROVED_FOR_OFFLINE_EVALUATION":
        raise PermissionError("improvement candidate is not approved for offline evaluation")
    if candidate.get("target_layer") != "model":
        raise ValueError("model challenger bridge accepts model-layer improvement candidates only")
    metadata = challenger.metadata or {}
    if metadata.get("improvement_candidate_id") != improvement_candidate_id:
        raise ValueError("challenger metadata does not bind the approved improvement candidate")
    if metadata.get("feedback_id") != candidate.get("feedback_id"):
        raise ValueError("challenger metadata feedback_id does not match the approved improvement candidate")
    result = run_controlled_evolution_cycle(
        root=root,
        dataset_id=dataset_id,
        baseline=baseline,
        challenger=challenger,
        failure_regression_root=failure_regression_root,
        predictor=predictor,
        split_audit_ref=split_audit_ref,
        evidence_refs=list(candidate.get("evidence_refs") or []),
    )
    return {
        **result,
        "improvement_candidate_id": improvement_candidate_id,
        "improvement_candidate_status": candidate["status"],
        "feedback_id": candidate["feedback_id"],
        "candidate_target_layer": candidate["target_layer"],
        "automatic_production_promotion": False,
        "claim_boundary": (
            "Only an expert-approved model improvement candidate may enter this offline evaluator. "
            "A successful evaluation can create a human-gated upgrade proposal but never promotes production automatically."
        ),
    }


__all__=["run_approved_model_challenger"]
