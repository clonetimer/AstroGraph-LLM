"""Unified governed evaluator dispatcher for reviewed improvement candidates."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from astrograph.agents.mission.storage import sha
from .sealed_model_challenger import run_sealed_approved_model_challenger
from .knowledge_feedback import KnowledgeFeedbackStore
from .sealed_layer_evaluation import qualify_sealed_non_model_candidate
from .evaluation_run_receipt import (
    load_and_validate_evaluation_run_receipt, model_metric_payload, layer_check_payload, evaluation_artifact_ref,
)
from .layer_evaluation import LayerEvaluationArtifact


class GovernedEvolutionReviewPacket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["astrograph-governed-evolution-review-packet.v1"] = "astrograph-governed-evolution-review-packet.v1"
    improvement_candidate_id: str = Field(min_length=1)
    feedback_id: str = Field(min_length=1)
    target_layer: Literal["planning", "memory", "tool", "knowledge", "model", "execution"]
    evaluator_route: Literal["MODEL_CHALLENGER", "LAYER_SPECIFIC_ARTIFACT"]
    native_evaluator_status: str = Field(min_length=1)
    review_status: Literal["ELIGIBLE_FOR_HUMAN_REVIEW", "REJECTED", "STOPPED"]
    evaluator_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_result: dict[str, Any]
    upgrade_proposal: dict[str, Any] | None = None
    requires_human_gate: Literal[True] = True
    automatic_production_promotion: Literal[False] = False
    production_modified: Literal[False] = False
    claim_boundary: str = (
        "Unified routing only. The candidate is evaluated by its target-layer-specific offline evaluator. "
        "No route can automatically promote or deploy a production change."
    )


def evaluate_approved_improvement_candidate(
    *,
    store: KnowledgeFeedbackStore,
    improvement_candidate_id: str,
    model_evaluation_kwargs: dict[str, Any] | None = None,
    model_work_order_path: str | Path | None = None,
    model_expected_work_order_sha256: str | None = None,
    model_challenger_seal_path: str | Path | None = None,
    model_expected_challenger_seal_sha256: str | None = None,
    model_evaluation_receipt_path: str | Path | None = None,
    model_expected_evaluation_receipt_sha256: str | None = None,
    non_model_work_order_path: str | Path | None = None,
    non_model_expected_work_order_sha256: str | None = None,
    non_model_challenger_seal_path: str | Path | None = None,
    non_model_expected_challenger_seal_sha256: str | None = None,
    non_model_evaluation_artifact_path: str | Path | None = None,
    non_model_expected_sha256: str | None = None,
    non_model_evaluation_receipt_path: str | Path | None = None,
    non_model_expected_evaluation_receipt_sha256: str | None = None,
) -> GovernedEvolutionReviewPacket:
    candidate = store.export_candidate(improvement_candidate_id)
    if candidate.get("status") != "APPROVED_FOR_OFFLINE_EVALUATION":
        raise PermissionError("improvement candidate is not approved for offline evaluation")
    layer = candidate.get("target_layer")

    if layer == "model":
        if any(value is not None for value in (
            non_model_work_order_path, non_model_expected_work_order_sha256,
            non_model_challenger_seal_path, non_model_expected_challenger_seal_sha256,
            non_model_evaluation_artifact_path, non_model_expected_sha256,
            non_model_evaluation_receipt_path, non_model_expected_evaluation_receipt_sha256,
        )):
            raise ValueError("non-model sealed evaluation arguments are invalid for a model candidate")
        if not model_evaluation_kwargs:
            raise ValueError("model candidate requires model_evaluation_kwargs")
        if any(value is None for value in (model_work_order_path, model_expected_work_order_sha256,
                                            model_challenger_seal_path, model_expected_challenger_seal_sha256,
                                            model_evaluation_receipt_path, model_expected_evaluation_receipt_sha256)):
            raise ValueError("model candidate requires sealed work order, challenger seal, and evaluation run receipt bindings")
        split_ref=str(model_evaluation_kwargs.get("split_audit_ref") or "").strip()
        if not split_ref:
            raise ValueError("model evaluation requires a non-empty split_audit_ref")
        challenger=model_evaluation_kwargs.get("challenger")
        if challenger is None:
            raise ValueError("model_evaluation_kwargs must include challenger")
        eval_receipt=load_and_validate_evaluation_run_receipt(
            path=model_evaluation_receipt_path,expected_sha256=model_expected_evaluation_receipt_sha256,
            improvement_candidate_id=improvement_candidate_id,feedback_id=candidate["feedback_id"],target_layer="model",
            work_order_sha256=model_expected_work_order_sha256,challenger_seal_sha256=model_expected_challenger_seal_sha256,
            expected_result_kind="MODEL_TASK_METRICS",expected_result_payload=model_metric_payload(challenger),required_source_ref=split_ref,
        )
        result = run_sealed_approved_model_challenger(
            store=store,improvement_candidate_id=improvement_candidate_id,
            work_order_path=model_work_order_path,expected_work_order_sha256=model_expected_work_order_sha256,
            challenger_seal_path=model_challenger_seal_path,expected_challenger_seal_sha256=model_expected_challenger_seal_sha256,
            **model_evaluation_kwargs,
        )
        result={**result,"evaluation_run_id":eval_receipt.run_id,"evaluation_run_receipt_sha256":model_expected_evaluation_receipt_sha256}
        native_status = str(result["status"])
        if native_status == "PENDING_HUMAN_GATE":
            review_status = "ELIGIBLE_FOR_HUMAN_REVIEW"
        elif native_status == "REJECTED":
            review_status = "REJECTED"
        else:
            review_status = "STOPPED"
        route = "MODEL_CHALLENGER"
        proposal = result.get("upgrade_proposal")
    else:
        if model_evaluation_kwargs is not None or any(value is not None for value in (
            model_work_order_path, model_expected_work_order_sha256, model_challenger_seal_path, model_expected_challenger_seal_sha256,
            model_evaluation_receipt_path, model_expected_evaluation_receipt_sha256
        )):
            raise ValueError("model sealed evaluation arguments are invalid for a non-model candidate")
        required_values=(
            non_model_work_order_path, non_model_expected_work_order_sha256,
            non_model_challenger_seal_path, non_model_expected_challenger_seal_sha256,
            non_model_evaluation_artifact_path, non_model_expected_sha256,
            non_model_evaluation_receipt_path, non_model_expected_evaluation_receipt_sha256,
        )
        if any(value is None for value in required_values):
            raise ValueError("non-model candidate requires sealed work order, challenger seal, evaluation artifact, and evaluation run receipt bindings")
        eval_artifact_path=Path(non_model_evaluation_artifact_path).expanduser()
        if eval_artifact_path.is_symlink():
            raise ValueError("non-model evaluation artifact must not be a symlink")
        eval_artifact=LayerEvaluationArtifact.model_validate_json(eval_artifact_path.resolve(strict=True).read_text(encoding="utf-8"))
        eval_receipt=load_and_validate_evaluation_run_receipt(
            path=non_model_evaluation_receipt_path,expected_sha256=non_model_expected_evaluation_receipt_sha256,
            improvement_candidate_id=improvement_candidate_id,feedback_id=candidate["feedback_id"],target_layer=layer,
            work_order_sha256=non_model_expected_work_order_sha256,challenger_seal_sha256=non_model_expected_challenger_seal_sha256,
            expected_result_kind="LAYER_CHECKS",expected_result_payload=layer_check_payload(eval_artifact),
            required_source_ref=evaluation_artifact_ref(non_model_expected_sha256),
            expected_evaluator_id=eval_artifact.evaluator_id,expected_evaluator_version=eval_artifact.evaluator_version,
        )
        qualified = qualify_sealed_non_model_candidate(
            store=store,
            improvement_candidate_id=improvement_candidate_id,
            work_order_path=non_model_work_order_path,
            expected_work_order_sha256=non_model_expected_work_order_sha256,
            challenger_seal_path=non_model_challenger_seal_path,
            expected_challenger_seal_sha256=non_model_expected_challenger_seal_sha256,
            evaluation_artifact_path=non_model_evaluation_artifact_path,
            expected_evaluation_artifact_sha256=non_model_expected_sha256,
        )
        result = {**qualified.model_dump(mode="json"),"evaluation_run_id":eval_receipt.run_id,
                  "evaluation_run_receipt_sha256":non_model_expected_evaluation_receipt_sha256}
        native_status = qualified.status
        review_status = "ELIGIBLE_FOR_HUMAN_REVIEW" if qualified.status == "ELIGIBLE_FOR_GOVERNED_PROMOTION_REVIEW" else "REJECTED"
        route = "LAYER_SPECIFIC_ARTIFACT"
        proposal = qualified.native_qualification.upgrade_proposal

    return GovernedEvolutionReviewPacket(
        improvement_candidate_id=improvement_candidate_id,
        feedback_id=candidate["feedback_id"],
        target_layer=layer,
        evaluator_route=route,
        native_evaluator_status=native_status,
        review_status=review_status,
        evaluator_result_sha256=sha(result),
        evaluator_result=result,
        upgrade_proposal=proposal,
    )


__all__ = ["GovernedEvolutionReviewPacket", "evaluate_approved_improvement_candidate"]
