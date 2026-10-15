from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from astrograph.agents.mission.models import AgentRequest, AgentSettings
from astrograph.agents.mission.service import MissionAgent
from astrograph.agents.mission.storage import file_sha

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'tests/fixtures/mission_three_task_candidate'


def _package(tmp:Path,*,admitted:bool=True):
    out=tmp/'package';out.mkdir()
    shutil.copy2(BASE/'weights.pt',out/'weights.pt')
    payload=json.loads((BASE/'manifest.json').read_text(encoding="utf-8"))
    payload['production_admitted']=admitted
    (out/'manifest.json').write_text(json.dumps(payload),encoding='utf-8')
    return out/'manifest.json'


def _configs(tmp:Path,package:Path):
    capability=tmp/'capabilities.json'
    capability.write_text(json.dumps({
        'schema_version':'astrograph-model-capability-catalog.v1','entries':[
            {'capability_id':'fixture-three-task','model_id':'fixture-three-task','model_version':'1','domain':'generic',
             'tasks':['detection','localization','classification'],'runtime_kind':'EXTERNAL_ADMITTED','channel_match':'EXACT',
             'required_channels':['sensor.x','sensor.y'],'units':{'sensor.x':'1','sensor.y':'1'},'artifact_manifest':None,
             'online_eligible':True,'retrospective_only':False,'priority':500,
             'claim_boundary':'fixture exact-channel three-task model'}]}),encoding='utf-8')
    bindings=tmp/'bindings.json'
    bindings.write_text(json.dumps({'schema_version':'astrograph-model-execution-bindings.v1','bindings':[
        {'binding_id':'fixture-binding','capability_id':'fixture-three-task','model_id':'fixture-three-task','model_version':'1',
         'provider_kind':'MISSION_THREE_TASK_TCN_PACKAGE','package_manifest':str(package),
         'package_manifest_sha256':file_sha(package),'enabled':True,'online_allowed':True,
         'claim_boundary':'test strict admitted execution'}]}),encoding='utf-8')
    return capability,bindings


def _series(tmp:Path):
    csv=tmp/'series.csv';pd.DataFrame({'time_s':range(10),'sensor.x':[0.1*i for i in range(10)],'sensor.y':[1.0-0.05*i for i in range(10)]}).to_csv(csv,index=False)
    manifest=tmp/'series.json';manifest.write_text(json.dumps({'schema_version':'astrograph-generic-series-manifest.v1','data_path':csv.name,
        'entity_id':'fixture','timestamp_column':'time_s','channels':[{'name':'sensor.x','unit':'1'},{'name':'sensor.y','unit':'1'}],
        'baseline_fraction':0.3}),encoding='utf-8')
    return manifest


def _settings(tmp:Path,capability:Path,bindings:Path):
    return AgentSettings(project_root=str(ROOT),read_roots=[str(ROOT),str(tmp)],output_root=str(tmp/'runs'),
        model_capability_registry=str(capability),model_execution_bindings=str(bindings),max_optional_actions=0)


def test_compatible_capability_executes_only_through_strict_admission(tmp_path:Path):
    package=_package(tmp_path,admitted=True);cap,binding=_configs(tmp_path,package);series=_series(tmp_path)
    result=MissionAgent(_settings(tmp_path,cap,binding)).run_inline(AgentRequest(task_id='bridge-ok',artifacts={'series_manifest':str(series)}))
    execution=result.output['diagnosis_result']['model_execution']
    assert result.status=='SUCCEEDED'
    assert execution['status']=='SUCCEEDED' and execution['execution_authorized'] is True
    assert set(execution['heads'])=={'detection','localization','classification'}
    assert execution['admission']['receipt_id']
    assert execution['heads']['localization']['root_cause_supported'] is False


def test_compatible_but_unadmitted_package_fails_closed(tmp_path:Path):
    package=_package(tmp_path,admitted=False);cap,binding=_configs(tmp_path,package);series=_series(tmp_path)
    result=MissionAgent(_settings(tmp_path,cap,binding)).run_inline(AgentRequest(task_id='bridge-denied',artifacts={'series_manifest':str(series)}))
    assert result.status in {'PARTIAL','BLOCKED'}
    assert any('PermissionError' in x for x in result.limitations)


def test_no_execution_binding_does_not_turn_discovery_into_authorization(tmp_path:Path):
    package=_package(tmp_path,admitted=True);cap,_=_configs(tmp_path,package);series=_series(tmp_path)
    empty=tmp_path/'empty.json';empty.write_text(json.dumps({'schema_version':'astrograph-model-execution-bindings.v1','bindings':[]}),encoding='utf-8')
    result=MissionAgent(_settings(tmp_path,cap,empty)).run_inline(AgentRequest(task_id='bridge-none',artifacts={'series_manifest':str(series)}))
    execution=result.output['diagnosis_result']['model_execution']
    assert execution['status']=='NOT_AVAILABLE' and execution['execution_authorized'] is False
    assert result.output['diagnosis_result']['model_capabilities']['execution_authorized'] is False
