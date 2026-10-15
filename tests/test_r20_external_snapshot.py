import shutil
import pytest
from astrograph.evaluation.external_dataset_contract import *
from tests.test_r20_external_dataset_contract import receipt

def test_snapshot_relocation_and_data_truth_drift(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();(a/'data.csv').write_text('1');(a/'truth.csv').write_text('nominal')
    kw=dict(dataset_receipt=receipt(),partitions={'s':'EXTERNAL_VALIDATION'},groups=['s'])
    s=build_snapshot(a,[a/'data.csv'],truth_paths=[a/'truth.csv'],**kw)
    shutil.copytree(a,b);assert s==build_snapshot(b,[b/'data.csv'],truth_paths=[b/'truth.csv'],**kw);verify_snapshot(b,s,**kw)
    with pytest.raises(ValueError,match='PARTITION_MANIFEST_DRIFT'):verify_snapshot(b,s,partitions={'s':'MODEL_TRAIN'})
    with pytest.raises(ValueError,match='GROUP_IDENTITY_DRIFT'):verify_snapshot(b,s,groups=['other'])
    (b/'truth.csv').write_text('fault')
    with pytest.raises(ValueError,match='BYTES_DRIFT'):verify_snapshot(b,s)
    shutil.copyfile(a/'truth.csv',b/'truth.csv');(b/'data.csv').write_text('2')
    with pytest.raises(ValueError,match='BYTES_DRIFT'):verify_snapshot(b,s)

def test_snapshot_cannot_escape_data_root(tmp_path):
    a=tmp_path/'a';a.mkdir();p=tmp_path/'truth';p.write_text('1')
    with pytest.raises(ValueError,match='ESCAPES'):build_snapshot(a,[p],dataset_receipt=receipt(),truth_paths=[],partitions={},groups=[])
