from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from tests.test_rw_window_mechanisms import frames
from astrograph.oag import physical_native as native

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize("side,field,expected",[
    ("source","adcs.rw.measured_speed_rad_s_0","UNAVAILABLE_MISSING_OBSERVATIONS"),
    ("source","adcs.control.applied_torque_nm_0","NOT_EXECUTED_EVIDENCE_UNAVAILABLE"),
    ("nominal","adcs.control.applied_torque_nm_0","NOT_EXECUTED_EVIDENCE_UNAVAILABLE"),
])
def test_entry_missing_values_reach_physical_route(tmp_path,monkeypatch,side,field,expected):
    source,nominal=frames();source['adcs.control.applied_torque_nm_0']=0
    (source if side=="source" else nominal).loc[10,field]=np.nan
    source.to_csv(tmp_path/'source.csv',index=False);nominal.to_csv(tmp_path/'nominal.csv',index=False)
    model=tmp_path/'model.pt';model.write_bytes(b'unit-test-provider')
    provider=SimpleNamespace(selected_channels=('adcs.rw.measured_speed_rad_s_0',),
                             manifest=SimpleNamespace(input_contract={'window_length':20}))
    monkeypatch.setattr(native,'load_adcs_three_task_torch_package',lambda *a,**k:provider)
    def forbidden(**kwargs):raise AssertionError('incomplete mechanism branch must not run')
    monkeypatch.setattr(native,'from_native_frames',forbidden)
    result=native.diagnose_files(source_path=tmp_path/'source.csv',nominal_path=tmp_path/'nominal.csv',
                                model_path=model,project_root=ROOT,output_path=tmp_path/'result.json')
    assert result['model_status']==expected
    assert result['original_oag_result'] is None
    assert result['route_readiness']['imputation_performed'] is False
    assert result['diagnosis']['physical_evidence']['receipt']['valid_pairs']==19


def test_complete_input_keeps_routes_available():
    s,n=frames()
    readiness=native.paired_route_readiness(s,n,['adcs.rw.measured_speed_rad_s_0'])
    assert readiness['model_inputs_ready'] and readiness['paired_mechanism_ready']


def test_time_mismatch_not_hidden_as_missing_data():
    s,n=frames();n.time_s+=.1
    with pytest.raises(ValueError,match='aligned'):native.paired_route_readiness(s,n,['adcs.rw.measured_speed_rad_s_0'])


def test_missing_column_not_imputed():
    s,n=frames();s=s.drop(columns=['adcs.control.applied_torque_nm_0'])
    with pytest.raises(ValueError,match='fields'):native.paired_route_readiness(s,n,['adcs.rw.measured_speed_rad_s_0'])
