import json,copy
from pathlib import Path
import pytest
from astrograph.evaluation.dxc_statistical_predictor import predict
from astrograph.evaluation.data_exposure import dataset_use

def fixture():
    catalog={'components':[{'component_id':'E1','kind':'sensor','unit':'VDC','unit_status':'VERIFIED_FROM_XML','fault_modes':['Stuck','Offset']}],'connections':[]}
    rows=[{'kind':'sensors','time_tick':t*1000,'values':{'E1':10.+(t%2)*.1 if t<30 else 3.}} for t in range(100)]
    config=json.loads((Path(__file__).parents[1]/'configs/evaluation/r20_dxc.json').read_text())
    return rows,catalog,config

def test_independent_set_output_and_no_truth_fields():
    rows,catalog,c=fixture()
    with dataset_use('SYNTHETIC_PROTOCOL','DEVELOPMENT','OPTIMIZE'):
        p=predict(rows,catalog,c)
    assert p['predicted_fault_ids']==['E1:Stuck']
    assert p['candidate_sets']==[['E1:Stuck']]
    for field in ('anomaly','train','faultInjection','truth','scenario_name'):
        bad=copy.deepcopy(rows);bad[0][field]='Nominal'
        with dataset_use('SYNTHETIC_PROTOCOL','DEVELOPMENT','OPTIMIZE'):
            with pytest.raises(ValueError,match='DXC_TRUTH'): predict(bad,catalog,c)

def test_unknown_unit_channel_is_audited_not_formal_evidence():
    rows,catalog,c=fixture();catalog['components'][0]['unit_status']='UNIT_UNVERIFIED'
    with dataset_use('SYNTHETIC_PROTOCOL','DEVELOPMENT','OPTIMIZE'):
        p=predict(rows,catalog,c)
    assert p['predicted_fault_ids']==[] and p['unverified_unit_channels']==['E1']
