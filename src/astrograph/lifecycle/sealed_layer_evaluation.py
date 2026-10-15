"""Bind non-model offline evaluation to the exact sandbox work order and challenger seal."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .knowledge_feedback import KnowledgeFeedbackStore
from .layer_evaluation import LayerEvaluationArtifact, LayerEvaluationQualification, qualify_approved_non_model_candidate
from .sandbox_work_order import ChallengerArtifactSeal, ImprovementWorkOrder


def _file_sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def work_order_ref(digest: str) -> str:
    return f"work-order-sha256:{digest}"


def challenger_seal_ref(digest: str) -> str:
    return f"challenger-seal-sha256:{digest}"


class SealedLayerEvaluationQualification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["astrograph-sealed-layer-evaluation-qualification.v1"] = "astrograph-sealed-layer-evaluation-qualification.v1"
    improvement_candidate_id: str = Field(min_length=1)
    target_layer: Literal["planning", "memory", "tool", "knowledge", "execution"]
    work_order_id: str = Field(min_length=1)
    work_order_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    challenger_seal_id: str = Field(min_length=1)
    challenger_seal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    challenger_artifact_sha256: list[str] = Field(min_length=1)
    status: Literal["ELIGIBLE_FOR_GOVERNED_PROMOTION_REVIEW", "REJECTED"]
    native_qualification: LayerEvaluationQualification
    requires_human_gate: Literal[True] = True
    automatic_production_promotion: Literal[False] = False
    production_modified: Literal[False] = False
    claim_boundary: str = (
        "Qualification is bound to the exact work order, challenger seal, current challenger file hashes, and evaluation artifact. "
        "It authorizes only governed promotion review and never deploys a production change."
    )


def qualify_sealed_non_model_candidate(
    *,
    store: KnowledgeFeedbackStore,
    improvement_candidate_id: str,
    work_order_path: str | Path,
    expected_work_order_sha256: str,
    challenger_seal_path: str | Path,
    expected_challenger_seal_sha256: str,
    evaluation_artifact_path: str | Path,
    expected_evaluation_artifact_sha256: str,
) -> SealedLayerEvaluationQualification:
    candidate=store.export_candidate(improvement_candidate_id)
    if candidate.get("status") != "APPROVED_FOR_OFFLINE_EVALUATION":
        raise PermissionError("improvement candidate is not approved for offline evaluation")
    if candidate.get("target_layer") == "model":
        raise ValueError("sealed non-model evaluator does not accept model candidates")

    wp=Path(work_order_path).expanduser()
    sp=Path(challenger_seal_path).expanduser()
    ep=Path(evaluation_artifact_path).expanduser()
    for path,label in ((wp,"work order"),(sp,"challenger seal"),(ep,"evaluation artifact")):
        if path.is_symlink(): raise ValueError(f"{label} must not be a symlink")
        if not path.resolve(strict=True).is_file(): raise ValueError(f"{label} must be a regular file")
    wp=wp.resolve();sp=sp.resolve();ep=ep.resolve()
    if _file_sha(wp) != expected_work_order_sha256: raise ValueError("work order SHA-256 mismatch")
    if _file_sha(sp) != expected_challenger_seal_sha256: raise ValueError("challenger seal SHA-256 mismatch")
    if _file_sha(ep) != expected_evaluation_artifact_sha256: raise ValueError("evaluation artifact SHA-256 mismatch")

    order=ImprovementWorkOrder.model_validate_json(wp.read_text(encoding="utf-8"))
    seal=ChallengerArtifactSeal.model_validate_json(sp.read_text(encoding="utf-8"))
    evaluation=LayerEvaluationArtifact.model_validate_json(ep.read_text(encoding="utf-8"))
    if order.improvement_candidate_id != improvement_candidate_id:
        raise ValueError("work order does not bind the approved improvement candidate")
    if order.feedback_id != candidate.get("feedback_id") or order.target_layer != candidate.get("target_layer"):
        raise ValueError("work order lifecycle binding mismatch")
    if seal.work_order_id != order.work_order_id or seal.work_order_sha256 != expected_work_order_sha256:
        raise ValueError("challenger seal does not bind the supplied work order")

    root=Path(order.challenger_root).resolve(strict=True)
    current_hashes=[]
    for artifact in seal.challenger_artifacts:
        raw=Path(artifact.path)
        if raw.is_symlink(): raise ValueError("sealed challenger artifact became a symlink")
        current=raw.resolve(strict=True)
        if not current.is_file() or not current.is_relative_to(root):
            raise PermissionError("sealed challenger artifact escaped work-order sandbox root")
        digest=_file_sha(current)
        if digest != artifact.sha256:
            raise ValueError("sealed challenger artifact changed after sealing")
        current_hashes.append(digest)

    refs=set(evaluation.evaluated_artifact_refs)
    required_refs={work_order_ref(expected_work_order_sha256),challenger_seal_ref(expected_challenger_seal_sha256)}
    missing=sorted(required_refs-refs)
    if missing:
        raise ValueError(f"evaluation artifact does not bind work order and challenger seal: {missing}")

    native=qualify_approved_non_model_candidate(
        store=store,improvement_candidate_id=improvement_candidate_id,
        evaluation_artifact_path=ep,expected_sha256=expected_evaluation_artifact_sha256,
    )
    return SealedLayerEvaluationQualification(
        improvement_candidate_id=improvement_candidate_id,target_layer=order.target_layer,
        work_order_id=order.work_order_id,work_order_sha256=expected_work_order_sha256,
        challenger_seal_id=seal.seal_id,challenger_seal_sha256=expected_challenger_seal_sha256,
        evaluation_artifact_sha256=expected_evaluation_artifact_sha256,
        challenger_artifact_sha256=current_hashes,status=native.status,native_qualification=native,
    )


__all__=["SealedLayerEvaluationQualification","work_order_ref","challenger_seal_ref","qualify_sealed_non_model_candidate"]
