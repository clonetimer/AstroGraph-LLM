import pytest
from astrograph.evaluation.esa_statistical_predictor import fit_reference,score
from astrograph.evaluation.data_exposure import dataset_use

def test_observation_only_model_input_rejects_truth():
    with dataset_use('SYNTHETIC_PROTOCOL','DEVELOPMENT','FIT'):
        reference=fit_reference([1.,2.,3.])
        payload=dict(time_ns=[0,1,2],value=[1.,2.,float('nan')],observed=[True,True,False],channel='c')
        assert score(payload,reference)[2]==0
    for field in ('label','anomaly','train','fault_truth','event_id','category'):
        with dataset_use('SYNTHETIC_PROTOCOL','DEVELOPMENT','OPTIMIZE'):
            with pytest.raises(ValueError,match='ESA_TRUTH'): score({**payload,field:1},reference)
