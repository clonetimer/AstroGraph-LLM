import pytest
from astrograph.evaluation.external_dataset_contract import require_same_task,TASK_FIELDS,task_identity
TASK=dict(evaluation_task_id='OPS',label_semantics_sha256='1'*64,target_universe_sha256='2'*64)

@pytest.mark.parametrize('key',TASK_FIELDS)
def test_each_identity_dimension_required(key):
    other=dict(TASK);other.pop(key)
    with pytest.raises(ValueError,match='CROSS_TASK'):require_same_task(TASK,other)

@pytest.mark.parametrize('key',TASK_FIELDS)
def test_each_identity_dimension_must_match(key):
    other=dict(TASK);other[key]='DXC' if key=='evaluation_task_id' else '3'*64
    with pytest.raises(ValueError,match='CROSS_TASK'):require_same_task(TASK,other)
