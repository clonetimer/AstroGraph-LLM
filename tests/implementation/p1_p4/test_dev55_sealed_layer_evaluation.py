from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from astrograph.lifecycle import (
    ExpertFeedback, KnowledgeFeedbackStore, LayerEvaluationArtifact, LifecyclePolicy, REQUIRED_CHECKS,
    build_improvement_work_order, challenger_seal_ref, qualify_sealed_non_model_candidate,
    seal_sandbox_challenger, work_order_ref,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _store(tmp_path):
    return KnowledgeFeedbackStore(tmp_path/"lifecycle.sqlite",LifecyclePolicy(
        approved_reviewers=("expert:dev55",),permitted_document_roots=(str(tmp_path),)))


def _candidate(store, layer="tool"):
    fb=ExpertFeedback(feedback_id=f"feedback:dev55:{layer}",task_id="episode:dev55",result_sha256="e"*64,
        feedback_type="DIAGNOSIS_CORRECTION",reviewer="expert:dev55",reviewed_at=datetime(2026,9,16,9,0,tzinfo=timezone.utc),
        truth_source_ref="truth:independent",verified_fault_id="FAULT_B",target_layers=(layer,),
        diagnosis_error_summary="wrong diagnosis",recommended_change=f"prepare {layer} change",
        evidence_refs=(f"EVID-{layer.upper()}-DEV55",))
    cid=store.submit_feedback(fb,event_id=f"submit:{layer}")["candidate_ids"][0]
    store.review_candidate(candidate_id=cid,decision="APPROVE_FOR_OFFLINE_EVALUATION",reviewer="expert:dev55",
                           reason="offline only",event_id=f"approve:{layer}")
    return fb,cid


def _sealed(tmp_path, store, fb, cid, layer="tool"):
    base=tmp_path/f"baseline-{layer}.json";base.write_text('{"v":1}',encoding="utf-8")
    kind={"planning":"PLANNER_POLICY","memory":"MEMORY_CANDIDATE","tool":"TOOL_SELECTION_POLICY",
          "knowledge":"KNOWLEDGE_PROPOSAL","execution":"RUNTIME_POLICY","model":"MODEL_CHALLENGER_PACKAGE"}[layer]
    order,wp,wsha=build_improvement_work_order(store=store,improvement_candidate_id=cid,sandbox_root=tmp_path/"sandbox",
                                               change_kind=kind,baseline_artifacts={"baseline":base})
    challenger=Path(order.challenger_root)/"candidate.json";challenger.write_text('{"v":2}',encoding="utf-8")
    seal,sp,ssha=seal_sandbox_challenger(work_order_path=wp,expected_work_order_sha256=wsha,
                                         challenger_artifacts={"candidate":challenger})
    return order,wp,wsha,challenger,seal,sp,ssha


def _evaluation(tmp_path, fb, cid, layer, wsha, ssha, *, failed=None, bind=True):
    checks={name:True for name in REQUIRED_CHECKS[layer]}
    if failed: checks[failed]=False
    refs=["candidate:artifact"]
    if bind: refs += [work_order_ref(wsha),challenger_seal_ref(ssha)]
    artifact=LayerEvaluationArtifact(evaluation_id=f"eval:dev55:{layer}",improvement_candidate_id=cid,
        feedback_id=fb.feedback_id,target_layer=layer,evaluator_id=f"eval:{layer}",evaluator_version="dev55.v1",
        checks=checks,evidence_refs=list(fb.evidence_refs),evaluated_artifact_refs=refs)
    path=tmp_path/f"eval-{layer}.json";path.write_text(artifact.model_dump_json(indent=2),encoding="utf-8")
    return path,_sha(path)


def test_exact_sealed_challenger_and_evaluation_can_reach_governed_review(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"tool")
    order,wp,wsha,challenger,seal,sp,ssha=_sealed(tmp_path,store,fb,cid,"tool")
    ep,esha=_evaluation(tmp_path,fb,cid,"tool",wsha,ssha)
    result=qualify_sealed_non_model_candidate(store=store,improvement_candidate_id=cid,
        work_order_path=wp,expected_work_order_sha256=wsha,challenger_seal_path=sp,expected_challenger_seal_sha256=ssha,
        evaluation_artifact_path=ep,expected_evaluation_artifact_sha256=esha)
    assert result.status=="ELIGIBLE_FOR_GOVERNED_PROMOTION_REVIEW"
    assert result.work_order_id==order.work_order_id and result.challenger_seal_id==seal.seal_id
    assert result.native_qualification.upgrade_proposal is not None
    assert result.automatic_production_promotion is False and result.production_modified is False


def test_challenger_file_drift_after_seal_fails_closed(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"knowledge")
    order,wp,wsha,challenger,seal,sp,ssha=_sealed(tmp_path,store,fb,cid,"knowledge")
    ep,esha=_evaluation(tmp_path,fb,cid,"knowledge",wsha,ssha)
    challenger.write_text('{"v":999}',encoding="utf-8")
    with pytest.raises(ValueError,match="changed after sealing"):
        qualify_sealed_non_model_candidate(store=store,improvement_candidate_id=cid,
            work_order_path=wp,expected_work_order_sha256=wsha,challenger_seal_path=sp,expected_challenger_seal_sha256=ssha,
            evaluation_artifact_path=ep,expected_evaluation_artifact_sha256=esha)


def test_seal_or_work_order_hash_drift_fails_closed(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"memory")
    order,wp,wsha,challenger,seal,sp,ssha=_sealed(tmp_path,store,fb,cid,"memory")
    ep,esha=_evaluation(tmp_path,fb,cid,"memory",wsha,ssha)
    sp.write_text(sp.read_text(encoding="utf-8")+'\n',encoding="utf-8")
    with pytest.raises(ValueError,match="challenger seal SHA-256 mismatch"):
        qualify_sealed_non_model_candidate(store=store,improvement_candidate_id=cid,
            work_order_path=wp,expected_work_order_sha256=wsha,challenger_seal_path=sp,expected_challenger_seal_sha256=ssha,
            evaluation_artifact_path=ep,expected_evaluation_artifact_sha256=esha)


def test_evaluation_must_explicitly_bind_work_order_and_seal(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"planning")
    order,wp,wsha,challenger,seal,sp,ssha=_sealed(tmp_path,store,fb,cid,"planning")
    ep,esha=_evaluation(tmp_path,fb,cid,"planning",wsha,ssha,bind=False)
    with pytest.raises(ValueError,match="does not bind work order and challenger seal"):
        qualify_sealed_non_model_candidate(store=store,improvement_candidate_id=cid,
            work_order_path=wp,expected_work_order_sha256=wsha,challenger_seal_path=sp,expected_challenger_seal_sha256=ssha,
            evaluation_artifact_path=ep,expected_evaluation_artifact_sha256=esha)


def test_failed_layer_check_remains_rejected_even_when_seal_binding_is_valid(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"execution")
    order,wp,wsha,challenger,seal,sp,ssha=_sealed(tmp_path,store,fb,cid,"execution")
    ep,esha=_evaluation(tmp_path,fb,cid,"execution",wsha,ssha,failed="RUNTIME_REGRESSION")
    result=qualify_sealed_non_model_candidate(store=store,improvement_candidate_id=cid,
        work_order_path=wp,expected_work_order_sha256=wsha,challenger_seal_path=sp,expected_challenger_seal_sha256=ssha,
        evaluation_artifact_path=ep,expected_evaluation_artifact_sha256=esha)
    assert result.status=="REJECTED"
    assert result.native_qualification.upgrade_proposal is None


def test_model_candidate_is_not_accepted_by_sealed_non_model_evaluator(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"model")
    order,wp,wsha,challenger,seal,sp,ssha=_sealed(tmp_path,store,fb,cid,"model")
    # Create a syntactically valid non-model evaluation file only to prove routing blocks before trusting it.
    fake=LayerEvaluationArtifact(evaluation_id="eval:fake",improvement_candidate_id=cid,feedback_id=fb.feedback_id,
        target_layer="tool",evaluator_id="fake",evaluator_version="dev55",checks={name:True for name in REQUIRED_CHECKS["tool"]},
        evidence_refs=list(fb.evidence_refs),evaluated_artifact_refs=[work_order_ref(wsha),challenger_seal_ref(ssha)])
    ep=tmp_path/"fake.json";ep.write_text(fake.model_dump_json(indent=2),encoding="utf-8")
    with pytest.raises(ValueError,match="does not accept model candidates"):
        qualify_sealed_non_model_candidate(store=store,improvement_candidate_id=cid,
            work_order_path=wp,expected_work_order_sha256=wsha,challenger_seal_path=sp,expected_challenger_seal_sha256=ssha,
            evaluation_artifact_path=ep,expected_evaluation_artifact_sha256=_sha(ep))


def test_evaluation_artifact_hash_drift_fails_closed(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"tool")
    order,wp,wsha,challenger,seal,sp,ssha=_sealed(tmp_path,store,fb,cid,"tool")
    ep,esha=_evaluation(tmp_path,fb,cid,"tool",wsha,ssha)
    ep.write_text(ep.read_text(encoding="utf-8")+'\n',encoding="utf-8")
    with pytest.raises(ValueError,match="evaluation artifact SHA-256 mismatch"):
        qualify_sealed_non_model_candidate(store=store,improvement_candidate_id=cid,
            work_order_path=wp,expected_work_order_sha256=wsha,challenger_seal_path=sp,expected_challenger_seal_sha256=ssha,
            evaluation_artifact_path=ep,expected_evaluation_artifact_sha256=esha)
