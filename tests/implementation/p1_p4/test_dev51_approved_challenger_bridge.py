from __future__ import annotations

from datetime import datetime, timezone

import pytest

from astrograph.data_driven.failure_regression import FailureCase, FailureRegressionSet
from astrograph.data_driven.feedback import FeedbackCandidate, TaskMetrics
from astrograph.lifecycle import ExpertFeedback, KnowledgeFeedbackStore, LifecyclePolicy, run_approved_model_challenger
from astrograph.agents.mission.storage import sha


def _store(tmp_path):
    policy=LifecyclePolicy(approved_reviewers=("expert:dev51",),permitted_document_roots=(str(tmp_path),))
    return KnowledgeFeedbackStore(tmp_path/"lifecycle.sqlite",policy)


def _feedback(layer="model"):
    return ExpertFeedback(feedback_id=f"feedback:dev51:{layer}",task_id="episode:dev51",result_sha256="a"*64,
        feedback_type="DIAGNOSIS_CORRECTION",reviewer="expert:dev51",reviewed_at=datetime(2026,9,16,5,0,tzinfo=timezone.utc),
        truth_source_ref="truth:independent",verified_fault_id="FAULT_B",target_layers=(layer,),
        diagnosis_error_summary="incorrect diagnosis",recommended_change="evaluate challenger",
        evidence_refs=("EVID-MODEL-1234567890ABCDEF1234",))


def _candidates(candidate_id, feedback_id, *, e2e=.92):
    baseline=FeedbackCandidate(candidate_id="baseline:v1",model_id="model",model_version="v1",family="TCN",
        metrics=TaskMetrics(detection_macro_f1=.95,localization_macro_f1=.96,classification_macro_f1=.95,end_to_end_macro_f1=.90))
    challenger=FeedbackCandidate(candidate_id="challenger:v2",model_id="model",model_version="v2",family="TCN",
        metrics=TaskMetrics(detection_macro_f1=.955,localization_macro_f1=.965,classification_macro_f1=.956,end_to_end_macro_f1=e2e),
        metadata={"improvement_candidate_id":candidate_id,"feedback_id":feedback_id})
    return baseline,challenger


def _submit(store, layer="model"):
    feedback=_feedback(layer)
    receipt=store.submit_feedback(feedback,event_id=f"submit:{layer}")
    return feedback,receipt["candidate_ids"][0]


def test_pending_improvement_candidate_cannot_enter_offline_challenger_evaluation(tmp_path) -> None:
    store=_store(tmp_path);feedback,cid=_submit(store)
    baseline,challenger=_candidates(cid,feedback.feedback_id)
    with pytest.raises(PermissionError,match="not approved for offline evaluation"):
        run_approved_model_challenger(store=store,improvement_candidate_id=cid,root=tmp_path/"cycles",dataset_id="ADCS",
            baseline=baseline,challenger=challenger,failure_regression_root=tmp_path/"regression",predictor=lambda _:"FAULT_B")


def test_approved_model_candidate_can_reach_human_gated_upgrade_proposal(tmp_path) -> None:
    store=_store(tmp_path);feedback,cid=_submit(store)
    store.review_candidate(candidate_id=cid,decision="APPROVE_FOR_OFFLINE_EVALUATION",reviewer="expert:dev51",reason="run offline only",event_id="approve:model")
    baseline,challenger=_candidates(cid,feedback.feedback_id)
    result=run_approved_model_challenger(store=store,improvement_candidate_id=cid,root=tmp_path/"cycles",dataset_id="ADCS",
        baseline=baseline,challenger=challenger,failure_regression_root=tmp_path/"regression",predictor=lambda _:"FAULT_B",split_audit_ref="split:held")
    assert result["status"]=="PENDING_HUMAN_GATE"
    assert result["upgrade_proposal"] is not None
    assert result["upgrade_proposal"]["requires_human_gate"] is True
    assert result["production_modified"] is False
    assert result["automatic_production_promotion"] is False


def test_challenger_must_bind_exact_approved_candidate_and_feedback(tmp_path) -> None:
    store=_store(tmp_path);feedback,cid=_submit(store)
    store.review_candidate(candidate_id=cid,decision="APPROVE_FOR_OFFLINE_EVALUATION",reviewer="expert:dev51",reason="offline",event_id="approve:model")
    baseline,challenger=_candidates(cid,feedback.feedback_id)
    bad=challenger.model_copy(update={"metadata":{"improvement_candidate_id":"IMPROVE-WRONG","feedback_id":feedback.feedback_id}})
    with pytest.raises(ValueError,match="does not bind"):
        run_approved_model_challenger(store=store,improvement_candidate_id=cid,root=tmp_path/"cycles",dataset_id="ADCS",
            baseline=baseline,challenger=bad,failure_regression_root=tmp_path/"regression",predictor=lambda _:"FAULT_B")


def test_unfixed_expert_failure_case_rejects_otherwise_good_challenger(tmp_path) -> None:
    store=_store(tmp_path);feedback,cid=_submit(store)
    store.review_candidate(candidate_id=cid,decision="APPROVE_FOR_OFFLINE_EVALUATION",reviewer="expert:dev51",reason="offline",event_id="approve:model")
    FailureRegressionSet(tmp_path/"regression").add(FailureCase(case_id="FAIL-DEV51",episode_id="episode:dev51",failure_type="MISDIAGNOSIS",
        truth_fault_id="FAULT_B",predicted_fault_id="FAULT_A",feedback_id=feedback.feedback_id,evidence_refs=list(feedback.evidence_refs),recorded_at=feedback.reviewed_at.isoformat()))
    baseline,challenger=_candidates(cid,feedback.feedback_id)
    result=run_approved_model_challenger(store=store,improvement_candidate_id=cid,root=tmp_path/"cycles",dataset_id="ADCS",
        baseline=baseline,challenger=challenger,failure_regression_root=tmp_path/"regression",predictor=lambda _:"FAULT_A")
    assert result["status"]=="REJECTED"
    assert result["upgrade_proposal"] is None
    assert result["failure_regression"]["fixed_count"]==0
    assert any("remain unfixed" in reason for reason in result["selection"]["reasons"])


def test_non_model_improvement_candidate_is_not_routed_to_model_evaluator(tmp_path) -> None:
    store=_store(tmp_path);feedback,cid=_submit(store,"knowledge")
    store.review_candidate(candidate_id=cid,decision="APPROVE_FOR_OFFLINE_EVALUATION",reviewer="expert:dev51",reason="offline",event_id="approve:knowledge")
    baseline,challenger=_candidates(cid,feedback.feedback_id)
    with pytest.raises(ValueError,match="model-layer"):
        run_approved_model_challenger(store=store,improvement_candidate_id=cid,root=tmp_path/"cycles",dataset_id="ADCS",
            baseline=baseline,challenger=challenger,failure_regression_root=tmp_path/"regression",predictor=lambda _:"FAULT_B")
