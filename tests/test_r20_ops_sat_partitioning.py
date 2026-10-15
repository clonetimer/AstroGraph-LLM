import json,copy
from pathlib import Path
import pytest
from astrograph.evaluation.ops_sat_adapter import build_partitions,verify_partitions

def fixture():
    truth={str(i):dict(segment=str(i),anomaly=i%2,train=0 if i<200 else 1) for i in range(2000)}
    config=json.loads((Path(__file__).parents[1]/'configs/evaluation/r20_ops_sat.json').read_text())
    return truth,config

def test_group_disjoint_repeatable_official_test_untouched():
    truth,c=fixture();a=build_partitions(truth,c);assert a==build_partitions(dict(reversed(list(truth.items()))),c)
    assert {r['group_id'] for r in a['groups'] if r['partition']=='EXTERNAL_VALIDATION'}=={str(i) for i in range(200)}

def test_duplicate_segment_cross_partition_blocked():
    truth,c=fixture();a=build_partitions(truth,c);a['groups'].append({'group_id':'0','partition':'MODEL_TRAIN'})
    with pytest.raises(ValueError,match='GROUP_PARTITION_LEAKAGE'):verify_partitions(a,truth)

def test_class_coverage_failure_blocks():
    truth,c=fixture()
    for t in truth.values():t['anomaly']=0
    with pytest.raises(ValueError,match='CLASS_COVERAGE'):build_partitions(truth,c)
