from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from astrograph.agents.mission.generic_series import GenericSeriesManifest
from astrograph.agents.mission.model_capabilities import ModelCapabilityQuery, ModelCapabilityRegistry
from astrograph.agents.mission.model_execution import ModelExecutionResolver
from astrograph.agents.mission.three_task_contract import normalize_three_task_execution

ROOT=Path(__file__).resolve().parents[3]


def _read_path(value:str, directory:bool=False):
    p=Path(value).resolve(strict=True)
    if directory and not p.is_dir(): raise ValueError("expected directory")
    if not directory and not p.exists(): raise ValueError("missing path")
    return p


def _native_manifest(tmp_path:Path, *, unit_override:tuple[int,str]|None=None, channel_limit:int|None=None):
    binding=json.loads((ROOT/'configs/agent/model_execution_bindings.default.json').read_text(encoding="utf-8"))['bindings'][0]
    channels=list(binding['required_channels']); units=list(binding['channel_units'])
    if channel_limit is not None:
        channels=channels[:channel_limit];units=units[:channel_limit]
    if unit_override is not None:
        units[unit_override[0]]=unit_override[1]
    frame=pd.DataFrame({name:[0.0]*20 for name in channels})
    data=tmp_path/'series.csv';frame.to_csv(data,index=False)
    manifest={
        'schema_version':'astrograph-generic-series-manifest.v1','data_path':str(data),'entity_id':'dev63-adcs',
        'channels':[{'name':name,'unit':unit} for name,unit in zip(channels,units,strict=True)],
        'baseline_fraction':0.2,'z_threshold':6.0,'minimum_baseline_points':10,
    }
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return mp,GenericSeriesManifest.model_validate(manifest)


def _resolver():
    return ModelExecutionResolver.from_json(ROOT/'configs/agent/model_execution_bindings.default.json',project_root=ROOT,read_path=_read_path)


def test_dev63_exact_33_channel_adcs_profile_executes_frozen_native_three_task_provider(tmp_path) -> None:
    mp,manifest=_native_manifest(tmp_path)
    resolver=_resolver();hint=resolver.profile_hint(manifest)
    assert hint['domain']=='adcs' and hint['bound_artifact_contract'] is True
    capabilities=ModelCapabilityRegistry.from_json(ROOT/'configs/agent/model_capabilities.default.json',project_root=ROOT)
    profile=capabilities.profile(ModelCapabilityQuery(domain='adcs',channel_names=tuple(x.name for x in manifest.channels),
        units={x.name:x.unit for x in manifest.channels if x.unit},tasks=('detection','localization','classification'),
        online_required=True,allow_retrospective=False,bound_artifact_contract=True))
    assert profile['coverage_tier']=='FULL_THREE_TASK'
    result=resolver.execute_generic(series_manifest_path=mp,capability_profile={'tasks':profile['heads']},episode_id='episode:dev63')
    assert result['status']=='SUCCEEDED' and result['provider_id']=='ADCS_NATIVE_THREE_TASK_V2'
    contract=normalize_three_task_execution(result)
    assert contract['detection']['status']=='EXECUTED'
    assert contract['localization']['status']=='EXECUTED_AFFECTED_COMPONENT_ONLY'
    assert contract['classification']['status']=='EXECUTED_SHADOW_UNCALIBRATED'
    assert contract['localization']['root_cause_supported'] is False
    assert contract['formal_diagnostic_vote_allowed'] is False


def test_dev63_partial_adcs_profile_does_not_auto_select_native_three_task_model(tmp_path) -> None:
    _,manifest=_native_manifest(tmp_path,channel_limit=10)
    hint=_resolver().profile_hint(manifest)
    assert hint['domain']=='generic' and hint['bound_artifact_contract'] is False


def test_dev63_unit_drift_breaks_host_bound_profile_match(tmp_path) -> None:
    _,manifest=_native_manifest(tmp_path,unit_override=(0,'rad'))
    hint=_resolver().profile_hint(manifest)
    assert hint['domain']=='generic' and hint['bound_artifact_contract'] is False


def test_dev63_default_binding_is_hash_pinned_to_current_approved_adcs_model() -> None:
    payload=json.loads((ROOT/'configs/agent/model_execution_bindings.default.json').read_text(encoding="utf-8"))
    assert len(payload['bindings'])==1
    b=payload['bindings'][0]
    assert b['provider_kind']=='ADCS_NATIVE_THREE_TASK_PACKAGE'
    assert b['package_manifest_sha256']=='39062457c1ed6d68c1478afa1d125ba54b80a6e07936de0a4078dca865852732'
    assert len(b['required_channels'])==33 and len(b['channel_units'])==33


def test_dev63_release_readiness_matrix_is_explicit_about_partial_and_external_boundaries() -> None:
    payload=json.loads((ROOT/'governance/implementation/post48_dev63/RELEASE_READINESS.json').read_text(encoding="utf-8"))
    assert payload['release_state']=='RC_READY_WITH_DECLARED_PROFILE_GATES_AND_EXTERNAL_DEPENDENCIES'
    assert [x['id'] for x in payload['scenarios']]==list(range(1,9))
    by_id={x['id']:x for x in payload['scenarios']}
    assert by_id[3]['status']=='PROFILE_GATED_PARTIAL'
    assert 'No universal single-signal classifier' in by_id[3]['boundary']
    assert by_id[4]['status']=='SUPPORTED_EXACT_ADCS_PROFILE'
    assert 'not root-cause proof' in by_id[4]['boundary']
    assert by_id[6]['status']=='SUPPORTED_NONVOTING_JOINT_EVIDENCE'
    assert payload['automatic_production_promotion'] is False
    deps={x['id']:x for x in payload['external_dependencies']}
    assert 'ADCS_NEW_INDEPENDENT_TRAJECTORY_GROUP' in deps
    assert 'REAL_MINERU_A2O_DOMAIN_QUALITY' in deps
    assert 'GENERIC_CLASSIFIER_LOCALIZER_PROFILE_COVERAGE' in deps


def test_dev63_default_mission_enables_only_the_strict_execution_binding_catalog() -> None:
    mission=json.loads((ROOT/'configs/agent/mission.default.json').read_text(encoding="utf-8"))
    assert mission['model_execution_bindings']=='configs/agent/model_execution_bindings.default.json'
    binding=json.loads((ROOT/mission['model_execution_bindings']).read_text(encoding="utf-8"))['bindings'][0]
    assert binding['enabled'] is True and binding['online_allowed'] is True
    assert binding['provider_kind']=='ADCS_NATIVE_THREE_TASK_PACKAGE'
