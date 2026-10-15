from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from astrograph.agents.mission.generic_series import GenericSeriesAdapter
from astrograph.agents.mission.model_capabilities import ModelCapabilityQuery,ModelCapabilityRegistry

ROOT=Path(__file__).resolve().parents[3]

def _reader(roots):
    roots=[Path(x).resolve() for x in roots]
    def read(value,directory=False):
        p=Path(value).resolve()
        if not any(p.is_relative_to(r) for r in roots): raise PermissionError(p)
        if directory and not p.is_dir(): raise FileNotFoundError(p)
        if not directory and not p.is_file(): raise FileNotFoundError(p)
        return p
    return read

def _write(tmp:Path,name:str):
    data=tmp/(name+'.csv')
    pd.DataFrame({'time_s':range(100),name:[0.0]*50+[100.0]*50}).to_csv(data,index=False)
    manifest=tmp/(name+'.json')
    manifest.write_text(json.dumps({'schema_version':'astrograph-generic-series-manifest.v1','data_path':data.name,
        'entity_id':'case','timestamp_column':'time_s','channels':[{'name':name,'unit':'V'}],
        'baseline_fraction':0.3}),encoding='utf-8')
    return manifest

def test_registered_single_series_detects_and_reasons(tmp_path:Path):
    m=_write(tmp_path,'sensor.eps.battery_voltage_v')
    adapter=GenericSeriesAdapter(project_root=ROOT,workspace_root=tmp_path/'run',read_path=_reader([ROOT,tmp_path]))
    out=adapter.run(m,request_id='p1-known')
    assert out['diagnosis']['detections'][0]['mapped'] is True
    assert out['diagnosis']['conclusion']['fault_id']=='EPS_BATTERY_VOLTAGE_FAULT'
    assert out['capability_boundary']['classification']=='NOT_EXECUTED_WITHOUT_COMPATIBLE_MODEL'

def test_unknown_single_series_detects_but_physical_cause_is_unknown(tmp_path:Path):
    m=_write(tmp_path,'mystery_signal')
    adapter=GenericSeriesAdapter(project_root=ROOT,workspace_root=tmp_path/'run',read_path=_reader([ROOT,tmp_path]))
    out=adapter.run(m,request_id='p1-unknown')
    assert out['diagnosis']['detections'][0]['mapped'] is False
    assert out['diagnosis']['conclusion']['level']=='unknown'
    assert out['diagnosis']['conclusion']['fault_id'] is None

def test_capability_registry_reports_heads_separately():
    reg=ModelCapabilityRegistry.from_json(ROOT/'configs/agent/model_capabilities.default.json',project_root=ROOT)
    det=reg.query(ModelCapabilityQuery(domain='generic',channel_names=('mystery_signal',),tasks=('detection',)))
    cls=reg.query(ModelCapabilityQuery(domain='generic',channel_names=('mystery_signal',),tasks=('classification',)))
    assert det['compatible'] and det['compatible'][0]['runtime_kind']=='BUILTIN_GENERIC'
    assert not cls['compatible']
    phm=reg.query(ModelCapabilityQuery(domain='phm',tasks=('classification',),online_required=False,allow_retrospective=True,bound_artifact_contract=True))
    assert any(x['runtime_kind']=='PHM_RETROSPECTIVE_BUNDLE' for x in phm['compatible'])

def test_domain_registry_does_not_leak_generic_detector_into_phm_replay():
    reg=ModelCapabilityRegistry.from_json(ROOT/'configs/agent/model_capabilities.default.json',project_root=ROOT)
    phm=reg.query(ModelCapabilityQuery(domain='phm',tasks=('detection',),online_required=False,allow_retrospective=True,bound_artifact_contract=True))
    assert phm['compatible']
    assert all(x['runtime_kind']=='PHM_RETROSPECTIVE_BUNDLE' for x in phm['compatible'])
