import pytest
from astrograph.evaluation.ops_sat_adapter import *
from astrograph.evaluation.ops_sat_external_validation import feature_fit,feature_score
from astrograph.evaluation.data_exposure import dataset_use

@pytest.mark.parametrize('track', ['raw','feature'])
@pytest.mark.parametrize('field', ['label','anomaly','train','fault_truth','prompt','EvidenceFact','candidate_selection','rag','kg','calibration_transform'])
def test_online_predictor_rejects_truth_and_unadmitted_sinks(track,field):
    p={k:1. for k in FEATURE_FIELDS};p.update(segment='s',channel='c',sampling='1')
    raw=dict(segment='s',channel='c',sampling='1',timestamp='2020-01-01T00:00:00Z',value=1.)
    p=p if track=='feature' else raw;p[field]=1
    with pytest.raises(ValueError,match='LABEL_FIELD_OR_UNADMITTED'):validate_online_fields(p,track=track)

def test_feature_prediction_has_no_truth_argument():
    p={k:1. for k in FEATURE_FIELDS};p.update(segment='s',channel='c',sampling='1')
    with dataset_use('SYNTHETIC_PROTOCOL','DEVELOPMENT','FIT'):
        fitted=feature_fit({'s':p},['s'])
    assert feature_score(p,fitted)==0.
    for field in ('anomaly','label','train'):
        contaminated={**p,field:'1'}
        with pytest.raises(ValueError):feature_score(contaminated,fitted)
