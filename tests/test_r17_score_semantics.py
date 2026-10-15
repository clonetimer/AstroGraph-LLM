from astrograph.data_driven import DataDrivenModelManifest,DataDrivenTaskType,ModelRuntime,ModelOutputSemantics,ModelCapability,ModelInferenceRequest,RawModelOutput,adapt_raw_model_output
from astrograph.fusion import FaultRanking,RankedFault,RankingSource,ScoreSemantics,BiRRFConfig
from astrograph.evaluation.score_semantics import audit_score_semantics
import pytest

def test_nonnegative_scores_become_normalized_ranking_mass_not_probability():
 m=DataDrivenModelManifest(model_id='h',model_version='1',display_name='h',task_type=DataDrivenTaskType.MULTICLASS_FAULT_CLASSIFICATION,runtime=ModelRuntime.CUSTOM_CALLABLE,output_semantics=ModelOutputSemantics.NONNEGATIVE_SCORES,capabilities=[ModelCapability.FAULT_RANKING],supported_fault_ids=['F'])
 r=ModelInferenceRequest(episode_id='e',model_id='h',model_version='1',feature_schema_id='x',input_data_sha256='0'*64,registered_fault_ids=['F'])
 p=adapt_raw_model_output(m,r,RawModelOutput(episode_id='e',model_id='h',model_version='1',output_semantics=ModelOutputSemantics.NONNEGATIVE_SCORES,class_values={'F':3,'NORMAL':1}))
 assert p.class_value_semantics=='NORMALIZED_RANKING_MASS'
 assert p.to_fault_ranking(['F']).score_semantics==ScoreSemantics.NORMALIZED_RANKING_MASS

def test_probability_semantics_reject_out_of_range_source_score():
 with pytest.raises(ValueError,match='within'):
  FaultRanking(ranking_id='r',episode_id='e',source=RankingSource.DATA_DRIVEN,score_semantics=ScoreSemantics.CALIBRATED_CLASS_PROBABILITY,calibration_id='c',producer_id='p',producer_version='1',candidates=[RankedFault(fault_id='F',rank=1,score=1.2,source=RankingSource.DATA_DRIVEN)])

def test_content_bound_calibration_receipt_is_required_for_probability_threshold_eligibility():
 from astrograph.data_driven import ProbabilityCalibrationReceipt
 rec=ProbabilityCalibrationReceipt(calibration_id='c',model_id='m',model_version='1',calibration_dataset_id='d',calibration_split_sha256='1'*64,calibration_method='TEMPERATURE_SCALING',calibration_report_sha256='2'*64,validation_expected_calibration_error=0.05,calibration_sample_count=20,calibration_group_count=10)
 m=DataDrivenModelManifest(model_id='m',model_version='1',display_name='m',task_type=DataDrivenTaskType.MULTICLASS_FAULT_CLASSIFICATION,runtime=ModelRuntime.CUSTOM_CALLABLE,output_semantics=ModelOutputSemantics.PROBABILITIES,capabilities=[ModelCapability.FAULT_RANKING,ModelCapability.CALIBRATED_PROBABILITY],supported_fault_ids=['F'],calibration_id='c',calibration_receipt_sha256=rec.receipt_sha256)
 r=ModelInferenceRequest(episode_id='e',model_id='m',model_version='1',feature_schema_id='x',input_data_sha256='0'*64,registered_fault_ids=['F'])
 p=adapt_raw_model_output(m,r,RawModelOutput(episode_id='e',model_id='m',model_version='1',output_semantics=ModelOutputSemantics.PROBABILITIES,class_values={'F':.8,'NORMAL':.2}))
 ranking=p.to_fault_ranking(['F'])
 declared=audit_score_semantics(prediction=p,manifest=m,ranking=ranking)
 verified=audit_score_semantics(prediction=p,manifest=m,ranking=ranking,calibration_receipt=rec)
 assert declared.probability_threshold_eligible is False
 assert declared.calibration_evidence_status=='CONTENT_BOUND_RECEIPT_DECLARED_NOT_VERIFIED'
 assert verified.probability_threshold_eligible is True
 assert verified.calibration_evidence_status=='CONTENT_BOUND_RECEIPT_VERIFIED_LEGACY_NO_EXTERNAL_VALIDATION'
 assert verified.external_validation_qualified is False
