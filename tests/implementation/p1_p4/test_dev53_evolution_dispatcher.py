from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from astrograph.data_driven.feedback import FeedbackCandidate, TaskMetrics
from astrograph.lifecycle import (
    ExpertFeedback, KnowledgeFeedbackStore, LayerEvaluationArtifact, LifecyclePolicy, REQUIRED_CHECKS,
    build_improvement_work_order, challenger_seal_ref, evaluate_approved_improvement_candidate,
    seal_sandbox_challenger, work_order_ref, write_evaluation_run_receipt, model_metric_payload,
    layer_check_payload, evaluation_artifact_ref,
)


def _store(tmp_path):
    return KnowledgeFeedbackStore(tmp_path/"lifecycle.sqlite",
        LifecyclePolicy(approved_reviewers=("expert:dev53",),permitted_document_roots=(str(tmp_path),)))


def _feedback(layer: str):
    return ExpertFeedback(feedback_id=f"feedback:dev53:{layer}",task_id=f"episode:dev53:{layer}",result_sha256="c"*64,
        feedback_type="DIAGNOSIS_CORRECTION",reviewer="expert:dev53",reviewed_at=datetime(2026,9,16,7,0,tzinfo=timezone.utc),
        truth_source_ref="truth:independent",verified_fault_id="FAULT_B",target_layers=(layer,),diagnosis_error_summary="wrong diagnosis",
        recommended_change=f"evaluate {layer} change",evidence_refs=(f"EVID-{layer.upper()}-DEV53",))


def _candidate(store, layer: str, *, approve=True):
    fb=_feedback(layer);cid=store.submit_feedback(fb,event_id=f"submit:{layer}")["candidate_ids"][0]
    if approve:
        store.review_candidate(candidate_id=cid,decision="APPROVE_FOR_OFFLINE_EVALUATION",reviewer="expert:dev53",
                               reason="offline only",event_id=f"approve:{layer}")
    return fb,cid


def _sealed_non_model_kwargs(tmp_path, store, fb, cid, layer: str, *, fail: str | None=None):
    baseline=tmp_path/f"baseline-{layer}.json";baseline.write_text('{"v":1}',encoding="utf-8")
    kind={"planning":"PLANNER_POLICY","memory":"MEMORY_CANDIDATE","tool":"TOOL_SELECTION_POLICY",
          "knowledge":"KNOWLEDGE_PROPOSAL","execution":"RUNTIME_POLICY"}[layer]
    order,wp,wsha=build_improvement_work_order(store=store,improvement_candidate_id=cid,sandbox_root=tmp_path/"sandbox",
                                               change_kind=kind,baseline_artifacts={"baseline":baseline})
    challenger=Path(order.challenger_root)/"candidate.json";challenger.write_text('{"v":2}',encoding="utf-8")
    _,sp,ssha=seal_sandbox_challenger(work_order_path=wp,expected_work_order_sha256=wsha,
                                      challenger_artifacts={"candidate":challenger})
    checks={name:True for name in REQUIRED_CHECKS[layer]}
    if fail: checks[fail]=False
    artifact=LayerEvaluationArtifact(evaluation_id=f"eval:dev53:{layer}",improvement_candidate_id=cid,feedback_id=fb.feedback_id,
        target_layer=layer,evaluator_id=f"eval:{layer}",evaluator_version="dev56.v1",checks=checks,
        evidence_refs=list(fb.evidence_refs),evaluated_artifact_refs=[work_order_ref(wsha),challenger_seal_ref(ssha)])
    ep=tmp_path/f"eval-{layer}.json";ep.write_text(artifact.model_dump_json(indent=2),encoding="utf-8")
    esha=hashlib.sha256(ep.read_bytes()).hexdigest()
    _,rp,rsha=write_evaluation_run_receipt(root=tmp_path/"evaluation-runs",improvement_candidate_id=cid,
        feedback_id=fb.feedback_id,target_layer=layer,work_order_sha256=wsha,challenger_seal_sha256=ssha,
        evaluator_id=artifact.evaluator_id,evaluator_version=artifact.evaluator_version,result_kind="LAYER_CHECKS",
        result_payload=layer_check_payload(artifact),source_refs=[evaluation_artifact_ref(esha)])
    return dict(non_model_work_order_path=wp,non_model_expected_work_order_sha256=wsha,
                non_model_challenger_seal_path=sp,non_model_expected_challenger_seal_sha256=ssha,
                non_model_evaluation_artifact_path=ep,non_model_expected_sha256=esha,
                non_model_evaluation_receipt_path=rp,non_model_expected_evaluation_receipt_sha256=rsha)


def _sealed_model_dispatch_args(tmp_path, store, cid, feedback_id, *, predictor=lambda _:"FAULT_B"):
    base_file=tmp_path/"baseline-model.pt";base_file.write_bytes(b"BASELINE")
    order,wp,wsha=build_improvement_work_order(store=store,improvement_candidate_id=cid,sandbox_root=tmp_path/"model-sandbox",
                                               change_kind="MODEL_CHALLENGER_PACKAGE",baseline_artifacts={"baseline-model":base_file})
    package=Path(order.challenger_root)/"challenger.pt";package.write_bytes(b"CHALLENGER")
    seal,sp,ssha=seal_sandbox_challenger(work_order_path=wp,expected_work_order_sha256=wsha,
                                         challenger_artifacts={"model-package":package})
    artifact_map={x.ref:x.sha256 for x in seal.challenger_artifacts}
    baseline=FeedbackCandidate(candidate_id="baseline:v1",model_id="model",model_version="v1",family="TCN",
        metrics=TaskMetrics(detection_macro_f1=.95,localization_macro_f1=.96,classification_macro_f1=.95,end_to_end_macro_f1=.90))
    challenger=FeedbackCandidate(candidate_id="challenger:v2",model_id="model",model_version="v2",family="TCN",
        metrics=TaskMetrics(detection_macro_f1=.955,localization_macro_f1=.965,classification_macro_f1=.956,end_to_end_macro_f1=.92),
        metadata={"improvement_candidate_id":cid,"feedback_id":feedback_id,"work_order_sha256":wsha,
                  "challenger_seal_sha256":ssha,"sealed_model_artifacts":artifact_map})
    _,rp,rsha=write_evaluation_run_receipt(root=tmp_path/"evaluation-runs",improvement_candidate_id=cid,
        feedback_id=feedback_id,target_layer="model",work_order_sha256=wsha,challenger_seal_sha256=ssha,
        evaluator_id="astrograph:model-metrics",evaluator_version="dev59.v1",result_kind="MODEL_TASK_METRICS",
        result_payload=model_metric_payload(challenger),source_refs=["split:held"])
    return {
        "model_work_order_path":wp,"model_expected_work_order_sha256":wsha,
        "model_challenger_seal_path":sp,"model_expected_challenger_seal_sha256":ssha,
        "model_evaluation_receipt_path":rp,"model_expected_evaluation_receipt_sha256":rsha,
        "model_evaluation_kwargs":{"root":tmp_path/"cycles","dataset_id":"ADCS","baseline":baseline,"challenger":challenger,
            "failure_regression_root":tmp_path/"regression","predictor":predictor,"split_audit_ref":"split:held"},
    }


def test_model_candidate_routes_to_model_challenger_and_only_reaches_human_review(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"model")
    packet=evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid,
                                                    **_sealed_model_dispatch_args(tmp_path,store,cid,fb.feedback_id))
    assert packet.evaluator_route=="MODEL_CHALLENGER"
    assert packet.native_evaluator_status=="PENDING_HUMAN_GATE"
    assert packet.review_status=="ELIGIBLE_FOR_HUMAN_REVIEW"
    assert packet.upgrade_proposal is not None
    assert packet.automatic_production_promotion is False and packet.production_modified is False


@pytest.mark.parametrize("layer",("planning","memory","tool","knowledge","execution"))
def test_non_model_candidates_route_to_layer_specific_evaluator(tmp_path, layer) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,layer);kwargs=_sealed_non_model_kwargs(tmp_path,store,fb,cid,layer)
    packet=evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid,**kwargs)
    assert packet.evaluator_route=="LAYER_SPECIFIC_ARTIFACT"
    assert packet.native_evaluator_status=="ELIGIBLE_FOR_GOVERNED_PROMOTION_REVIEW"
    assert packet.review_status=="ELIGIBLE_FOR_HUMAN_REVIEW"
    assert packet.upgrade_proposal is not None and packet.upgrade_proposal["promoted"] is False


def test_dispatcher_blocks_pending_candidate_before_route_execution(tmp_path) -> None:
    store=_store(tmp_path);_,cid=_candidate(store,"planning",approve=False)
    with pytest.raises(PermissionError,match="not approved for offline evaluation"):
        evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid)


def test_dispatcher_rejects_cross_route_arguments(tmp_path) -> None:
    store=_store(tmp_path);_,cid=_candidate(store,"tool")
    with pytest.raises(ValueError,match="model sealed evaluation arguments are invalid"):
        evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid,model_evaluation_kwargs={"bogus":True})
    store2=_store(tmp_path/"m");fb2,cid2=_candidate(store2,"model")
    with pytest.raises(ValueError,match="non-model sealed evaluation arguments are invalid"):
        evaluate_approved_improvement_candidate(store=store2,improvement_candidate_id=cid2,
            **_sealed_model_dispatch_args(tmp_path/"m",store2,cid2,fb2.feedback_id),non_model_work_order_path=tmp_path/"dummy")


def test_non_model_rejection_is_normalized_without_proposal(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"knowledge")
    kwargs=_sealed_non_model_kwargs(tmp_path,store,fb,cid,"knowledge",fail="GRAPH_REGRESSION")
    packet=evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid,**kwargs)
    assert packet.native_evaluator_status=="REJECTED" and packet.review_status=="REJECTED"
    assert packet.upgrade_proposal is None and packet.production_modified is False


def test_dispatcher_requires_route_specific_payload(tmp_path) -> None:
    store=_store(tmp_path);_,cid=_candidate(store,"memory")
    with pytest.raises(ValueError,match="requires sealed work order, challenger seal, evaluation artifact, and evaluation run receipt"):
        evaluate_approved_improvement_candidate(store=store,improvement_candidate_id=cid)
    store2=_store(tmp_path/"m");_,cid2=_candidate(store2,"model")
    with pytest.raises(ValueError,match="requires model_evaluation_kwargs"):
        evaluate_approved_improvement_candidate(store=store2,improvement_candidate_id=cid2)
