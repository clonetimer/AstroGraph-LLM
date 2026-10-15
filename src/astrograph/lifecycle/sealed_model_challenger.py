"""Bind model challenger evaluation to the exact sandbox model package seal."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from astrograph.data_driven.feedback import FeedbackCandidate
from .approved_challenger_bridge import run_approved_model_challenger
from .knowledge_feedback import KnowledgeFeedbackStore
from .sandbox_work_order import ChallengerArtifactSeal, ImprovementWorkOrder


def _file_sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def run_sealed_approved_model_challenger(
    *,
    store: KnowledgeFeedbackStore,
    improvement_candidate_id: str,
    work_order_path: str | Path,
    expected_work_order_sha256: str,
    challenger_seal_path: str | Path,
    expected_challenger_seal_sha256: str,
    root: str | Path,
    dataset_id: str,
    baseline: FeedbackCandidate,
    challenger: FeedbackCandidate,
    failure_regression_root: str | Path,
    predictor: Callable[[Any], str],
    split_audit_ref: str = "",
) -> dict[str, Any]:
    candidate=store.export_candidate(improvement_candidate_id)
    if candidate.get("status") != "APPROVED_FOR_OFFLINE_EVALUATION":
        raise PermissionError("improvement candidate is not approved for offline evaluation")
    if candidate.get("target_layer") != "model":
        raise ValueError("sealed model challenger accepts model-layer candidates only")

    wp=Path(work_order_path).expanduser(); sp=Path(challenger_seal_path).expanduser()
    for path,label in ((wp,"work order"),(sp,"challenger seal")):
        if path.is_symlink(): raise ValueError(f"{label} must not be a symlink")
        if not path.resolve(strict=True).is_file(): raise ValueError(f"{label} must be a regular file")
    wp=wp.resolve();sp=sp.resolve()
    if _file_sha(wp) != expected_work_order_sha256: raise ValueError("work order SHA-256 mismatch")
    if _file_sha(sp) != expected_challenger_seal_sha256: raise ValueError("challenger seal SHA-256 mismatch")
    order=ImprovementWorkOrder.model_validate_json(wp.read_text(encoding="utf-8"))
    seal=ChallengerArtifactSeal.model_validate_json(sp.read_text(encoding="utf-8"))
    if order.improvement_candidate_id != improvement_candidate_id or order.feedback_id != candidate.get("feedback_id"):
        raise ValueError("model work order lifecycle binding mismatch")
    if order.target_layer != "model" or order.change_kind != "MODEL_CHALLENGER_PACKAGE":
        raise ValueError("model work order has an invalid target layer or change kind")
    if seal.work_order_id != order.work_order_id or seal.work_order_sha256 != expected_work_order_sha256:
        raise ValueError("challenger seal does not bind the supplied model work order")

    root_dir=Path(order.challenger_root).resolve(strict=True)
    artifact_map={}
    for artifact in seal.challenger_artifacts:
        raw=Path(artifact.path)
        if raw.is_symlink(): raise ValueError("sealed model artifact became a symlink")
        current=raw.resolve(strict=True)
        if not current.is_file() or not current.is_relative_to(root_dir):
            raise PermissionError("sealed model artifact escaped work-order sandbox root")
        digest=_file_sha(current)
        if digest != artifact.sha256: raise ValueError("sealed model artifact changed after sealing")
        artifact_map[artifact.ref]=digest

    metadata=challenger.metadata or {}
    if metadata.get("work_order_sha256") != expected_work_order_sha256:
        raise ValueError("challenger metadata does not bind model work order SHA-256")
    if metadata.get("challenger_seal_sha256") != expected_challenger_seal_sha256:
        raise ValueError("challenger metadata does not bind challenger seal SHA-256")
    if metadata.get("sealed_model_artifacts") != artifact_map:
        raise ValueError("challenger metadata sealed_model_artifacts do not match current sealed package")

    result=run_approved_model_challenger(
        store=store,improvement_candidate_id=improvement_candidate_id,root=root,dataset_id=dataset_id,
        baseline=baseline,challenger=challenger,failure_regression_root=failure_regression_root,predictor=predictor,
        split_audit_ref=split_audit_ref,
    )
    return {**result,"work_order_id":order.work_order_id,"work_order_sha256":expected_work_order_sha256,
            "challenger_seal_id":seal.seal_id,"challenger_seal_sha256":expected_challenger_seal_sha256,
            "sealed_model_artifacts":artifact_map,
            "artifact_lineage_verified":True,"automatic_production_promotion":False,"production_modified":False}


__all__=["run_sealed_approved_model_challenger"]
