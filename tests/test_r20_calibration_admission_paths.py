import pytest
from tests.test_r20_task_identity import TASK
from tests.test_rw_jam_promotion import test_complete_evidence_can_reach_formal_ready as _complete_fixture_check
from astrograph.integrations.rw_jam_promotion import RWJamPromotionGate,PromotionStatus
from astrograph.data_driven.contracts import ProbabilityCalibrationQualificationReceipt

def test_formal_promotion_main_path_cannot_reuse_bound_external_task(monkeypatch):
    original=RWJamPromotionGate.assess;requests=[]
    def capture(self,request):
        requests.append(request);return original(self,request)
    monkeypatch.setattr(RWJamPromotionGate,'assess',capture)
    _complete_fixture_check()
    request=requests[0];model=request.model
    receipt=ProbabilityCalibrationQualificationReceipt.model_validate({**model.calibration_receipt.model_dump(),**TASK})
    manifest=model.manifest.model_copy(update={'calibration_receipt_sha256':receipt.receipt_sha256})
    bound=model.model_copy(update={'manifest':manifest,'calibration_receipt':receipt})
    result=original(RWJamPromotionGate(),request.model_copy(update={'model':bound}))
    assert result.status==PromotionStatus.BLOCKED and 'CROSS_TASK_CALIBRATION_REUSE_BLOCKED' in result.blockers
    matching=bound.model_copy(update=TASK)
    assert original(RWJamPromotionGate(),request.model_copy(update={'model':matching})).status==PromotionStatus.FORMAL_READY
    wrong=matching.model_copy(update={'evaluation_task_id':'NASA_ADAPT_DXC10_DPII'})
    assert 'CROSS_TASK_CALIBRATION_REUSE_BLOCKED' in original(RWJamPromotionGate(),request.model_copy(update={'model':wrong})).blockers

def test_score_audit_does_not_accept_foreign_qualification():
    from tests.test_r19_calibration_fitting import _fixture,_fit,_qualify
    from astrograph.data_driven import DataDrivenModelManifest,DataDrivenTaskType,ModelRuntime,ModelOutputSemantics,ModelCapability,ModelInferenceRequest,RawModelOutput,adapt_raw_model_output
    from astrograph.evaluation.score_semantics import audit_score_semantics
    p,t,c=_fixture();_,old=_qualify(p,t,c,_fit(p,t,c));receipt=ProbabilityCalibrationQualificationReceipt.model_validate({**old.model_dump(),**TASK})
    m=DataDrivenModelManifest(model_id=receipt.model_id,model_version=receipt.model_version,display_name='test',task_type=DataDrivenTaskType.MULTICLASS_FAULT_CLASSIFICATION,
       runtime=ModelRuntime.CUSTOM_CALLABLE,output_semantics=ModelOutputSemantics.PROBABILITIES,capabilities=[ModelCapability.FAULT_RANKING,ModelCapability.CALIBRATED_PROBABILITY],
       supported_fault_ids=['A','B'],calibration_id=receipt.calibration_id,calibration_receipt_sha256=receipt.receipt_sha256)
    r=ModelInferenceRequest(episode_id='e',model_id=m.model_id,model_version=m.model_version,feature_schema_id='f',input_data_sha256='0'*64,registered_fault_ids=['A','B'])
    prediction=adapt_raw_model_output(m,r,RawModelOutput(episode_id='e',model_id=m.model_id,model_version=m.model_version,output_semantics=ModelOutputSemantics.PROBABILITIES,class_values={'A':.9,'B':.1}))
    args=dict(prediction=prediction,manifest=m,ranking=prediction.to_fault_ranking(['A','B']),calibration_receipt=receipt)
    with pytest.raises(ValueError,match='CROSS_TASK'):audit_score_semantics(**args)
    with pytest.raises(ValueError,match='CROSS_TASK'):audit_score_semantics(**args,external_task_identity={**TASK,'evaluation_task_id':'ESA'})
    assert audit_score_semantics(**args,external_task_identity=TASK).calibration_evidence_status.startswith('EXTERNAL_VALIDATION_RECEIPT')
