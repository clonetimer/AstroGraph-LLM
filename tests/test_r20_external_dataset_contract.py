import copy
import pytest
from astrograph.evaluation.external_dataset_contract import *

def receipt():
    semantics={'0':'NOMINAL','1':'ANOMALY'};universe=['NOMINAL','ANOMALY']
    return seal(dict(schema_version='astrograph-external-evaluation-dataset-receipt.v1',dataset_id='test',dataset_family='OPS-SAT',
      source_artifact_sha256='1'*64,source_artifact_size_bytes=1,evaluation_task_id='OPS',task_kind='BINARY_ANOMALY_DETECTION',
      label_semantics=semantics,label_semantics_sha256=digest(semantics),target_universe=universe,target_universe_sha256=digest(universe),
      group_policy={'group':'segment'},truth_provenance={'field':'anomaly'},payload_manifest_sha256='2'*64,
      allowed_claims=['anomaly'],forbidden_claims=['fault'],receipt_sha256='0'*64))

def test_content_sealed_admission():
    r=receipt();ExternalDatasetReceipt.model_validate(r);r['target_universe'].append('FAULT')
    with pytest.raises(ValueError,match='TASK_SEMANTICS_DRIFT'):ExternalDatasetReceipt.model_validate(r)

def test_duplicate_universe_blocked_even_resealed():
    r=receipt();r['target_universe'].append('NOMINAL');r['target_universe_sha256']=digest(r['target_universe'])
    with pytest.raises(ValueError,match='DUPLICATE'):ExternalDatasetReceipt.model_validate(seal(r))
