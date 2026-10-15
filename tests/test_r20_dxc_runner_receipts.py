import hashlib
import json
import zipfile
import pytest
from astrograph.evaluation.dxc_receipt_writer import bind_role_evaluation
from astrograph.evaluation.external_dataset_contract import digest, seal, ExternalValidationReceipt, require_same_task


def inputs(tmp_path):
    archive=tmp_path/'input.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('truth.xls',b'synthetic truth artifact')
    case={'case_id':'group','group_id':'group','status':'SCORED','truth':{'true_fault_ids':[]}}
    cases=seal({'cases':[case],'workbook_member':'truth.xls','workbook_sha256':hashlib.sha256(b'synthetic truth artifact').hexdigest()})
    predictions=seal({'predictions':{'group':{'predicted_fault_ids':[]}}})
    report=seal({'evaluation_task_id':'DXC_TEST','role':'EXTERNAL_VALIDATION','status':'EXECUTED','predictions_sha256':digest(predictions['predictions']),'case_receipt_sha256':digest(cases['cases'])},'report_sha256')
    for name,obj in [('case_evaluation_receipt',cases),('frozen_predictions',predictions),('external_evaluation_report',report)]:
        (tmp_path/('test_'+name+'.json')).write_text(json.dumps(obj),encoding='utf-8')
    return archive,{'target_universe':['sensor:Offset'],'xml_sha256':'a'*64}


def test_runner_emits_bound_receipts_and_foreign_task_cannot_reuse_them(tmp_path):
    archive,catalog=inputs(tmp_path)
    report=bind_role_evaluation(archive,tmp_path,'test',catalog)
    receipt=json.loads((tmp_path/'test_external_validation_receipt.json').read_text())
    ExternalValidationReceipt.model_validate(receipt)
    require_same_task(report,receipt)
    with pytest.raises(ValueError,match='CROSS_TASK'):
        require_same_task({**report,'evaluation_task_id':'OTHER'},receipt)


def test_resealed_changed_predictions_still_fail_report_binding(tmp_path):
    archive,catalog=inputs(tmp_path)
    path=tmp_path/'test_frozen_predictions.json'
    path.write_text(json.dumps(seal({'predictions':{'group':{'predicted_fault_ids':['sensor:Offset']}}})),encoding='utf-8')
    with pytest.raises(ValueError,match='PREDICTION_REPORT_BINDING_DRIFT'):
        bind_role_evaluation(archive,tmp_path,'test',catalog)
