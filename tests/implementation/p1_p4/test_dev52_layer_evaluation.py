from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from astrograph.lifecycle import (
    ExpertFeedback,
    KnowledgeFeedbackStore,
    LayerEvaluationArtifact,
    LifecyclePolicy,
    REQUIRED_CHECKS,
    qualify_approved_non_model_candidate,
)

LAYERS=("planning","memory","tool","knowledge","execution")


def _store(tmp_path):
    policy=LifecyclePolicy(approved_reviewers=("expert:dev52",),permitted_document_roots=(str(tmp_path),))
    return KnowledgeFeedbackStore(tmp_path/"lifecycle.sqlite",policy)


def _feedback(layer: str):
    return ExpertFeedback(
        feedback_id=f"feedback:dev52:{layer}",
        task_id=f"episode:dev52:{layer}",
        result_sha256="b"*64,
        feedback_type="DIAGNOSIS_CORRECTION",
        reviewer="expert:dev52",
        reviewed_at=datetime(2026,9,16,6,0,tzinfo=timezone.utc),
        truth_source_ref="truth:independent",
        verified_fault_id="FAULT_B",
        target_layers=(layer,),
        diagnosis_error_summary=f"incorrect {layer} behavior",
        recommended_change=f"evaluate {layer} candidate offline",
        evidence_refs=(f"EVID-{layer.upper()}-1234567890ABCDEF",),
    )


def _approved_candidate(store, layer: str):
    feedback=_feedback(layer)
    receipt=store.submit_feedback(feedback,event_id=f"submit:{layer}")
    cid=receipt["candidate_ids"][0]
    store.review_candidate(
        candidate_id=cid,
        decision="APPROVE_FOR_OFFLINE_EVALUATION",
        reviewer="expert:dev52",
        reason="offline evaluation only",
        event_id=f"approve:{layer}",
    )
    return feedback,cid


def _write_artifact(tmp_path, *, candidate_id: str, feedback: ExpertFeedback, layer: str,
                    checks: dict[str,bool] | None=None, evidence_refs: list[str] | None=None,
                    feedback_id: str | None=None, artifact_layer: str | None=None):
    artifact=LayerEvaluationArtifact(
        evaluation_id=f"eval:dev52:{layer}",
        improvement_candidate_id=candidate_id,
        feedback_id=feedback_id or feedback.feedback_id,
        target_layer=artifact_layer or layer,
        evaluator_id=f"astrograph:{layer}:offline",
        evaluator_version="dev52.v1",
        checks=checks if checks is not None else {name:True for name in REQUIRED_CHECKS[layer]},
        metrics={"regression_cases":12,"trigger_case_fixed":True},
        evidence_refs=evidence_refs if evidence_refs is not None else list(feedback.evidence_refs),
        evaluated_artifact_refs=[f"artifact:{layer}:candidate"],
    )
    path=tmp_path/f"{layer}-evaluation.json"
    path.write_text(artifact.model_dump_json(indent=2),encoding="utf-8")
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    return path,digest


@pytest.mark.parametrize("layer",LAYERS)
def test_each_non_model_layer_has_distinct_required_checks_and_only_reaches_human_gated_review(tmp_path, layer) -> None:
    store=_store(tmp_path);feedback,cid=_approved_candidate(store,layer)
    path,digest=_write_artifact(tmp_path,candidate_id=cid,feedback=feedback,layer=layer)
    result=qualify_approved_non_model_candidate(
        store=store,improvement_candidate_id=cid,evaluation_artifact_path=path,expected_sha256=digest,
    )
    assert result.status=="ELIGIBLE_FOR_GOVERNED_PROMOTION_REVIEW"
    assert set(result.required_checks)==set(REQUIRED_CHECKS[layer])
    assert result.failed_required_checks==[]
    assert result.missing_required_checks==[]
    assert result.upgrade_proposal is not None
    assert result.upgrade_proposal["requires_human_gate"] is True
    assert result.upgrade_proposal["promoted"] is False
    assert result.automatic_production_promotion is False
    assert result.production_modified is False


def test_pending_candidate_is_blocked_before_artifact_evaluation(tmp_path) -> None:
    store=_store(tmp_path);feedback=_feedback("planning")
    cid=store.submit_feedback(feedback,event_id="submit:planning")["candidate_ids"][0]
    path,digest=_write_artifact(tmp_path,candidate_id=cid,feedback=feedback,layer="planning")
    with pytest.raises(PermissionError,match="not approved for offline evaluation"):
        qualify_approved_non_model_candidate(store=store,improvement_candidate_id=cid,evaluation_artifact_path=path,expected_sha256=digest)


def test_model_candidate_is_not_routed_to_non_model_evaluator(tmp_path) -> None:
    store=_store(tmp_path);feedback,cid=_approved_candidate(store,"model")
    # Artifact schema intentionally uses a non-model layer; routing must stop before it is trusted.
    path,digest=_write_artifact(tmp_path,candidate_id=cid,feedback=feedback,layer="planning",evidence_refs=list(feedback.evidence_refs))
    with pytest.raises(ValueError,match="model candidates must use"):
        qualify_approved_non_model_candidate(store=store,improvement_candidate_id=cid,evaluation_artifact_path=path,expected_sha256=digest)


def test_artifact_hash_drift_fails_closed(tmp_path) -> None:
    store=_store(tmp_path);feedback,cid=_approved_candidate(store,"tool")
    path,digest=_write_artifact(tmp_path,candidate_id=cid,feedback=feedback,layer="tool")
    path.write_text(path.read_text(encoding="utf-8")+"\n",encoding="utf-8")
    with pytest.raises(ValueError,match="SHA-256 mismatch"):
        qualify_approved_non_model_candidate(store=store,improvement_candidate_id=cid,evaluation_artifact_path=path,expected_sha256=digest)


@pytest.mark.parametrize("mutation,match",[
    ("candidate","does not bind"),
    ("feedback","feedback_id does not match"),
    ("layer","target layer does not match"),
])
def test_artifact_must_bind_exact_candidate_feedback_and_layer(tmp_path, mutation, match) -> None:
    store=_store(tmp_path);feedback,cid=_approved_candidate(store,"memory")
    kwargs={"candidate_id":cid,"feedback":feedback,"layer":"memory"}
    if mutation=="candidate": kwargs["candidate_id"]="IMPROVE-WRONG"
    elif mutation=="feedback": kwargs["feedback_id"]="feedback:wrong"
    elif mutation=="layer": kwargs["artifact_layer"]="planning"
    path,digest=_write_artifact(tmp_path,**kwargs)
    # For candidate mismatch, we still ask to qualify the approved real candidate.
    with pytest.raises(ValueError,match=match):
        qualify_approved_non_model_candidate(store=store,improvement_candidate_id=cid,evaluation_artifact_path=path,expected_sha256=digest)


def test_candidate_evidence_refs_cannot_be_omitted(tmp_path) -> None:
    store=_store(tmp_path);feedback,cid=_approved_candidate(store,"knowledge")
    path,digest=_write_artifact(tmp_path,candidate_id=cid,feedback=feedback,layer="knowledge",evidence_refs=[])
    with pytest.raises(ValueError,match="omitted candidate evidence references"):
        qualify_approved_non_model_candidate(store=store,improvement_candidate_id=cid,evaluation_artifact_path=path,expected_sha256=digest)


def test_missing_required_check_rejects_without_upgrade_proposal(tmp_path) -> None:
    store=_store(tmp_path);feedback,cid=_approved_candidate(store,"execution")
    checks={name:True for name in REQUIRED_CHECKS["execution"]}
    checks.pop("TIMEOUT_CANCEL_SAFETY")
    path,digest=_write_artifact(tmp_path,candidate_id=cid,feedback=feedback,layer="execution",checks=checks)
    result=qualify_approved_non_model_candidate(store=store,improvement_candidate_id=cid,evaluation_artifact_path=path,expected_sha256=digest)
    assert result.status=="REJECTED"
    assert result.missing_required_checks==["TIMEOUT_CANCEL_SAFETY"]
    assert result.upgrade_proposal is None
    assert result.production_modified is False


def test_failed_required_check_rejects_without_upgrade_proposal(tmp_path) -> None:
    store=_store(tmp_path);feedback,cid=_approved_candidate(store,"tool")
    checks={name:True for name in REQUIRED_CHECKS["tool"]};checks["NO_PERMISSION_ESCALATION"]=False
    path,digest=_write_artifact(tmp_path,candidate_id=cid,feedback=feedback,layer="tool",checks=checks)
    result=qualify_approved_non_model_candidate(store=store,improvement_candidate_id=cid,evaluation_artifact_path=path,expected_sha256=digest)
    assert result.status=="REJECTED"
    assert result.failed_required_checks==["NO_PERMISSION_ESCALATION"]
    assert result.upgrade_proposal is None
    assert result.automatic_production_promotion is False


def test_layer_check_sets_are_not_one_generic_gate() -> None:
    assert set(REQUIRED_CHECKS)==set(LAYERS)
    assert REQUIRED_CHECKS["knowledge"] != REQUIRED_CHECKS["memory"]
    assert "GRAPH_REGRESSION" in REQUIRED_CHECKS["knowledge"]
    assert "MEMORY_ISOLATION" in REQUIRED_CHECKS["memory"]
    assert "TOOL_PERMISSION_BOUNDARY" in REQUIRED_CHECKS["tool"]
    assert "PLAN_SAFETY" in REQUIRED_CHECKS["planning"]
    assert "TIMEOUT_CANCEL_SAFETY" in REQUIRED_CHECKS["execution"]
