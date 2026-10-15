from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from astrograph.agents.mission.identifiability import screen_single_signal_identifiability
from astrograph.agents.mission.models import AgentRequest, AgentSettings
from astrograph.agents.mission.service import MissionAgent
from astrograph.agents.mission.storage import file_sha

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'tests/fixtures/mission_three_task_candidate'


def test_stage82_identifiability_separates_knowledge_uniqueness_from_empirical_classification():
    report=screen_single_signal_identifiability(
        ['sensor.adcs.rw.motor_current_a_0','sensor.comm.rf_output_power_w','unknown.signal'],
        fault_catalog_path=ROOT/'configs/fault_modes/lab_fault_catalog.json')
    rows={x.channel:x for x in report.channels}
    assert rows['sensor.adcs.rw.motor_current_a_0'].knowledge_status=='KNOWLEDGE_AMBIGUOUS'
    assert set(rows['sensor.adcs.rw.motor_current_a_0'].catalog_candidate_fault_ids)=={'ADCS_RW_FRICTION_INCREASE','ADCS_RW_JAM'}
    assert rows['sensor.comm.rf_output_power_w'].knowledge_status=='KNOWLEDGE_UNIQUE'
    assert rows['sensor.comm.rf_output_power_w'].formal_single_signal_classification_eligible is False
    assert rows['unknown.signal'].knowledge_status=='NO_CATALOG_MAPPING'
    assert report.eligible_channel_count==0


def _fixture(tmp:Path):
    pkg=tmp/'package';pkg.mkdir();shutil.copy2(BASE/'weights.pt',pkg/'weights.pt')
    manifest=json.loads((BASE/'manifest.json').read_text(encoding="utf-8"));manifest['production_admitted']=True
    (pkg/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    cap=tmp/'cap.json';cap.write_text(json.dumps({'schema_version':'astrograph-model-capability-catalog.v1','entries':[
        {'capability_id':'fixture-three-task','model_id':'fixture-three-task','model_version':'1','domain':'generic',
         'tasks':['detection','localization','classification'],'runtime_kind':'EXTERNAL_ADMITTED','channel_match':'EXACT',
         'required_channels':['sensor.x','sensor.y'],'units':{'sensor.x':'1','sensor.y':'1'},'artifact_manifest':None,
         'online_eligible':True,'retrospective_only':False,'priority':500,'claim_boundary':'fixture'}]}),encoding='utf-8')
    bind=tmp/'bind.json';bind.write_text(json.dumps({'schema_version':'astrograph-model-execution-bindings.v1','bindings':[
        {'binding_id':'fixture-binding','capability_id':'fixture-three-task','model_id':'fixture-three-task','model_version':'1',
         'provider_kind':'MISSION_THREE_TASK_TCN_PACKAGE','package_manifest':str(pkg/'manifest.json'),
         'package_manifest_sha256':file_sha(pkg/'manifest.json'),'enabled':True,'online_allowed':True,'claim_boundary':'fixture'}]}),encoding='utf-8')
    csv=tmp/'series.csv';pd.DataFrame({'time_s':range(10),'sensor.x':[0.1*i for i in range(10)],'sensor.y':[1-.05*i for i in range(10)]}).to_csv(csv,index=False)
    series=tmp/'series.json';series.write_text(json.dumps({'schema_version':'astrograph-generic-series-manifest.v1','data_path':csv.name,
        'entity_id':'fixture','timestamp_column':'time_s','channels':[{'name':'sensor.x','unit':'1'},{'name':'sensor.y','unit':'1'}],
        'baseline_fraction':0.3}),encoding='utf-8')
    settings=AgentSettings(project_root=str(ROOT),read_roots=[str(ROOT),str(tmp)],output_root=str(tmp/'runs'),
        model_capability_registry=str(cap),model_execution_bindings=str(bind),max_optional_actions=0)
    return settings,series


def _run_three_task(tmp_path:Path, task_id:str):
    settings,series=_fixture(tmp_path)
    result=MissionAgent(settings).run_inline(AgentRequest(task_id=task_id,artifacts={'series_manifest':str(series)}))
    return result.output['diagnosis_result']['model_execution']


def test_stage83_three_task_contract_is_explicit_about_affected_component(tmp_path:Path):
    execution=_run_three_task(tmp_path,'dev19-stage83')
    three=execution['three_task_contract']
    assert three['detection']['status']=='EXECUTED'
    assert three['localization']['status']=='EXECUTED_AFFECTED_COMPONENT_ONLY'
    assert three['localization']['root_cause_supported'] is False
    assert three['classification']['status']=='EXECUTED_SHADOW_UNCALIBRATED'


def test_stage84_generic_model_rejoins_birrf_only_in_shadow(tmp_path:Path):
    execution=_run_three_task(tmp_path,'dev19-stage84')
    shadow=execution['shadow_fusion']
    assert shadow['status']=='SUCCEEDED_SHADOW'
    assert shadow['diagnostic_vote_allowed'] is False
    assert shadow['forward_context_ranking']['metadata']['diagnostic_vote_allowed'] is False
    assert shadow['birrf_result']['metadata']['configured_branch_weights']['KG_FORWARD']==0.0


def test_stage85_uncalibrated_or_unassessed_model_abstains(tmp_path:Path):
    execution=_run_three_task(tmp_path,'dev19-stage85')
    assert execution['decision_eligibility']['status']=='ABSTAIN'
    assert execution['decision_eligibility']['reason']=='UNCALIBRATED_MODEL_OUTPUT'
    assert execution['decision_eligibility']['decision_eligible'] is False
