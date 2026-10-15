from datetime import datetime, timezone
from pathlib import Path

from astrograph.agents.mission.storage import sha
from astrograph.data_driven.controlled_evolution import run_controlled_evolution_cycle
from astrograph.data_driven.feedback import FeedbackCandidate, TaskMetrics
from astrograph.lifecycle.feedback_regression_bridge import bind_feedback_to_failure_regression
from astrograph.lifecycle.knowledge_feedback import ExpertFeedback


def test_stage88_feedback_is_bound_to_original_result_and_failure_set(tmp_path:Path):
    result={'status':'SUCCEEDED','diagnosis':{'selected_fault_id':'F_BAD','candidate_fault_ids':['F_BAD','F_TRUE']}}
    feedback=ExpertFeedback(feedback_id='FB-1',task_id='CASE-1',result_sha256=sha(result),
        feedback_type='DIAGNOSIS_CORRECTION',reviewer='expert',reviewed_at=datetime.now(timezone.utc),
        truth_source_ref='truth://case-1',verified_fault_id='F_TRUE',target_layers=('model','planning'),
        diagnosis_error_summary='wrong top fault',recommended_change='improve separation',evidence_refs=('EV-1',))
    receipt=bind_feedback_to_failure_regression(feedback=feedback,original_result=result,
        regression_root=tmp_path/'reg',input_data_sha256='a'*64,model_refs=['model:v1'],knowledge_version='kg:v1')
    assert receipt['status']=='RECORDED'
    case=receipt['failure_case'];assert case['truth_fault_id']=='F_TRUE' and case['predicted_fault_id']=='F_BAD'
    # idempotent replay of the exact same feedback does not duplicate the case
    bind_feedback_to_failure_regression(feedback=feedback,original_result=result,regression_root=tmp_path/'reg',
        input_data_sha256='a'*64,model_refs=['model:v1'],knowledge_version='kg:v1')
    assert len((tmp_path/'reg'/'failure_regression_set.jsonl').read_text(encoding="utf-8").splitlines())==1


def test_stage89_offline_cycle_auto_evaluates_but_keeps_production_gate(tmp_path:Path):
    # Seed one expert-corrected regression case through Stage 88.
    result={'status':'SUCCEEDED','diagnosis':{'selected_fault_id':'F_BAD'}}
    feedback=ExpertFeedback(feedback_id='FB-2',task_id='CASE-2',result_sha256=sha(result),
        feedback_type='DIAGNOSIS_CORRECTION',reviewer='expert',reviewed_at=datetime.now(timezone.utc),
        truth_source_ref='truth://case-2',verified_fault_id='F_TRUE',target_layers=('model',),
        diagnosis_error_summary='wrong',recommended_change='candidate',evidence_refs=())
    bind_feedback_to_failure_regression(feedback=feedback,original_result=result,regression_root=tmp_path/'reg')
    baseline=FeedbackCandidate(candidate_id='base',model_id='m',model_version='1',metrics=TaskMetrics(
        detection_macro_f1=.90,localization_macro_f1=.90,classification_macro_f1=.90,end_to_end_macro_f1=.90))
    challenger=FeedbackCandidate(candidate_id='chall',model_id='m',model_version='2',metrics=TaskMetrics(
        detection_macro_f1=.91,localization_macro_f1=.91,classification_macro_f1=.92,end_to_end_macro_f1=.92))
    run=run_controlled_evolution_cycle(root=tmp_path/'cycles',dataset_id='dev-set',baseline=baseline,challenger=challenger,
        failure_regression_root=tmp_path/'reg',predictor=lambda case:'F_TRUE')
    assert run['offline_evaluation_completed'] is True
    assert run['status']=='PENDING_HUMAN_GATE'
    assert run['selection']['decision']=='PROPOSE'
    assert run['production_modified'] is False
    assert run['production_promotion_allowed_without_gate'] is False
