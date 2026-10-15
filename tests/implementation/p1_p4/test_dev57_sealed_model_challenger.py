from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from astrograph.data_driven.feedback import FeedbackCandidate, TaskMetrics
from astrograph.lifecycle import (
    ExpertFeedback, KnowledgeFeedbackStore, LifecyclePolicy,
    build_improvement_work_order, run_sealed_approved_model_challenger, seal_sandbox_challenger,
)


def _store(tmp_path):
    return KnowledgeFeedbackStore(tmp_path/"lifecycle.sqlite",LifecyclePolicy(
        approved_reviewers=("expert:dev57",),permitted_document_roots=(str(tmp_path),)))


def _candidate(store, layer="model"):
    fb=ExpertFeedback(feedback_id=f"feedback:dev57:{layer}",task_id="episode:dev57",result_sha256="f"*64,
        feedback_type="DIAGNOSIS_CORRECTION",reviewer="expert:dev57",reviewed_at=datetime(2026,9,16,10,0,tzinfo=timezone.utc),
        truth_source_ref="truth:independent",verified_fault_id="FAULT_B",target_layers=(layer,),
        diagnosis_error_summary="wrong diagnosis",recommended_change="evaluate sealed challenger package",
        evidence_refs=(f"EVID-{layer.upper()}-DEV57",))
    cid=store.submit_feedback(fb,event_id=f"submit:{layer}")["candidate_ids"][0]
    store.review_candidate(candidate_id=cid,decision="APPROVE_FOR_OFFLINE_EVALUATION",reviewer="expert:dev57",
                           reason="offline only",event_id=f"approve:{layer}")
    return fb,cid


def _sealed_model(tmp_path, store, fb, cid):
    base=tmp_path/"baseline.pt";base.write_bytes(b"BASELINE-MODEL")
    order,wp,wsha=build_improvement_work_order(store=store,improvement_candidate_id=cid,sandbox_root=tmp_path/"sandbox",
                                               change_kind="MODEL_CHALLENGER_PACKAGE",baseline_artifacts={"baseline-model":base})
    package=Path(order.challenger_root)/"challenger.pt";package.write_bytes(b"CHALLENGER-MODEL-V2")
    seal,sp,ssha=seal_sandbox_challenger(work_order_path=wp,expected_work_order_sha256=wsha,
                                         challenger_artifacts={"model-package":package})
    artifact_map={x.ref:x.sha256 for x in seal.challenger_artifacts}
    return order,wp,wsha,package,seal,sp,ssha,artifact_map


def _feedback_candidates(cid, feedback_id, artifact_map, wsha, ssha):
    baseline=FeedbackCandidate(candidate_id="baseline:v1",model_id="model",model_version="v1",family="TCN",
        metrics=TaskMetrics(detection_macro_f1=.95,localization_macro_f1=.96,classification_macro_f1=.95,end_to_end_macro_f1=.90))
    challenger=FeedbackCandidate(candidate_id="challenger:v2",model_id="model",model_version="v2",family="TCN",
        metrics=TaskMetrics(detection_macro_f1=.955,localization_macro_f1=.965,classification_macro_f1=.956,end_to_end_macro_f1=.92),
        metadata={"improvement_candidate_id":cid,"feedback_id":feedback_id,"work_order_sha256":wsha,
                  "challenger_seal_sha256":ssha,"sealed_model_artifacts":artifact_map})
    return baseline,challenger


def _run(tmp_path, store, cid, wp, wsha, sp, ssha, baseline, challenger):
    return run_sealed_approved_model_challenger(store=store,improvement_candidate_id=cid,
        work_order_path=wp,expected_work_order_sha256=wsha,challenger_seal_path=sp,expected_challenger_seal_sha256=ssha,
        root=tmp_path/"cycles",dataset_id="ADCS",baseline=baseline,challenger=challenger,
        failure_regression_root=tmp_path/"regression",predictor=lambda _:"FAULT_B",split_audit_ref="split:held")


def test_sealed_model_package_can_reach_human_gated_proposal(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store)
    order,wp,wsha,package,seal,sp,ssha,artifact_map=_sealed_model(tmp_path,store,fb,cid)
    baseline,challenger=_feedback_candidates(cid,fb.feedback_id,artifact_map,wsha,ssha)
    result=_run(tmp_path,store,cid,wp,wsha,sp,ssha,baseline,challenger)
    assert result["status"]=="PENDING_HUMAN_GATE"
    assert result["artifact_lineage_verified"] is True
    assert result["sealed_model_artifacts"]==artifact_map
    assert result["upgrade_proposal"] is not None
    assert result["automatic_production_promotion"] is False and result["production_modified"] is False


def test_model_package_drift_after_seal_fails_before_evaluation(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store)
    order,wp,wsha,package,seal,sp,ssha,artifact_map=_sealed_model(tmp_path,store,fb,cid)
    baseline,challenger=_feedback_candidates(cid,fb.feedback_id,artifact_map,wsha,ssha)
    package.write_bytes(b"TAMPERED")
    with pytest.raises(ValueError,match="changed after sealing"):
        _run(tmp_path,store,cid,wp,wsha,sp,ssha,baseline,challenger)


@pytest.mark.parametrize("field",("work_order_sha256","challenger_seal_sha256","sealed_model_artifacts"))
def test_challenger_metadata_must_bind_exact_sealed_package(tmp_path, field) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store)
    order,wp,wsha,package,seal,sp,ssha,artifact_map=_sealed_model(tmp_path,store,fb,cid)
    baseline,challenger=_feedback_candidates(cid,fb.feedback_id,artifact_map,wsha,ssha)
    metadata=dict(challenger.metadata)
    metadata[field] = ({"model-package":"0"*64} if field=="sealed_model_artifacts" else "0"*64)
    bad=challenger.model_copy(update={"metadata":metadata})
    with pytest.raises(ValueError,match="metadata"):
        _run(tmp_path,store,cid,wp,wsha,sp,ssha,baseline,bad)


def test_non_model_candidate_cannot_use_sealed_model_evaluator(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store,"tool")
    # Build only enough dummy paths to prove routing stops before trusting model assets.
    dummy=tmp_path/"dummy.json";dummy.write_text('{}',encoding="utf-8")
    baseline=FeedbackCandidate(candidate_id="b",model_id="m",model_version="v1",metrics=TaskMetrics(end_to_end_macro_f1=.9))
    challenger=FeedbackCandidate(candidate_id="c",model_id="m",model_version="v2",metrics=TaskMetrics(end_to_end_macro_f1=.92))
    with pytest.raises(ValueError,match="model-layer candidates only"):
        run_sealed_approved_model_challenger(store=store,improvement_candidate_id=cid,
            work_order_path=dummy,expected_work_order_sha256="0"*64,challenger_seal_path=dummy,expected_challenger_seal_sha256="0"*64,
            root=tmp_path/"cycles",dataset_id="ADCS",baseline=baseline,challenger=challenger,
            failure_regression_root=tmp_path/"regression",predictor=lambda _:"FAULT_B")


def test_model_work_order_or_seal_hash_drift_fails_closed(tmp_path) -> None:
    store=_store(tmp_path);fb,cid=_candidate(store)
    order,wp,wsha,package,seal,sp,ssha,artifact_map=_sealed_model(tmp_path,store,fb,cid)
    baseline,challenger=_feedback_candidates(cid,fb.feedback_id,artifact_map,wsha,ssha)
    wp.write_text(wp.read_text(encoding="utf-8")+'\n',encoding="utf-8")
    with pytest.raises(ValueError,match="work order SHA-256 mismatch"):
        _run(tmp_path,store,cid,wp,wsha,sp,ssha,baseline,challenger)
