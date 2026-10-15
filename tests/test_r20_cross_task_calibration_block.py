import pytest
from astrograph.evaluation.calibration_fitting import fit_temperature_scaler_receipt,apply_temperature_scaler_receipt
from astrograph.evaluation.data_exposure import dataset_use
from tests.test_r19_calibration_fitting import _fixture, _fit, _qualify
from astrograph.data_driven.contracts import ProbabilityCalibrationQualificationReceipt
TASK=dict(evaluation_task_id='OPS',label_semantics_sha256='1'*64,target_universe_sha256='2'*64)

def test_existing_temperature_main_path_requires_bound_matching_task():
    p,t,c=_fixture()
    with dataset_use('SYNTHETIC_PROTOCOL','CALIBRATION_FIT','CALIBRATE'):
        fit=fit_temperature_scaler_receipt(p,t,c,calibration_id='r20',model_id='m',model_version='1',calibration_dataset_id='d',
          calibration_dataset_sha256='3'*64,model_artifact_sha256='4'*64,evaluation_task_identity=TASK)
    assert apply_temperature_scaler_receipt(p,fit,evaluation_task_identity=TASK)
    for other in ({},{**TASK,'evaluation_task_id':'NASA_ADAPT_DXC10_DPII'},{**TASK,'target_universe_sha256':'5'*64}):
        with pytest.raises(ValueError,match='CROSS_TASK'):apply_temperature_scaler_receipt(p,fit,evaluation_task_identity=other)
    with pytest.raises(ValueError,match='CROSS_TASK'):apply_temperature_scaler_receipt(p,fit)
    with pytest.raises(ValueError,match='CROSS_TASK'):_qualify(p,t,c,fit)

def test_legacy_fit_cannot_enter_external_task_and_qualification_requires_identity():
    p,t,c=_fixture();fit=_fit(p,t,c)
    with pytest.raises(ValueError,match='CROSS_TASK'):apply_temperature_scaler_receipt(p,fit,evaluation_task_identity=TASK)
    _,legacy=_qualify(p,t,c,fit)
    assert 'evaluation_task_id' not in legacy.model_dump()
    with pytest.raises(ValueError,match='CROSS_TASK'):legacy.require_external_task(TASK)
    bound=ProbabilityCalibrationQualificationReceipt.model_validate({**legacy.model_dump(),**TASK})
    bound.require_external_task(TASK)
    with pytest.raises(ValueError,match='CROSS_TASK'):bound.require_external_task({**TASK,'evaluation_task_id':'ESA'})
    with pytest.raises(ValueError,match='TASK_IDENTITY_REQUIRED'):ProbabilityCalibrationQualificationReceipt.model_validate({**legacy.model_dump(),'evaluation_task_id':'OPS'})
