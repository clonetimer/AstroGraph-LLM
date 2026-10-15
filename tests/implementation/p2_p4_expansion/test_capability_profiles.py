from pathlib import Path
from astrograph.agents.mission.model_capabilities import ModelCapabilityQuery, ModelCapabilityRegistry

ROOT=Path(__file__).resolve().parents[3]

def registry():
    return ModelCapabilityRegistry.from_json(ROOT/'configs/agent/model_capabilities.default.json',project_root=ROOT)

def test_generic_profile_is_detection_only_and_never_authorizes_execution():
    p=registry().profile(ModelCapabilityQuery(domain='generic',channel_names=('sensor.eps.battery_voltage_v',),units={'sensor.eps.battery_voltage_v':'V'},tasks=('detection','localization','classification')))
    assert p['coverage_tier']=='DETECTION_ONLY'
    assert p['available_heads']==['detection']
    assert p['missing_heads']==['localization','classification']
    assert p['execution_authorized'] is False

def test_adcs_requires_host_bound_contract():
    r=registry()
    unbound=r.profile(ModelCapabilityQuery(domain='adcs',tasks=('detection','localization','classification')))
    bound=r.profile(ModelCapabilityQuery(domain='adcs',tasks=('detection','localization','classification'),bound_artifact_contract=True))
    assert unbound['coverage_tier']=='NO_COMPATIBLE_MODEL'
    assert unbound['recommended_next_step']=='USE_ADCS_NATIVE_ADAPTER_WITH_APPROVED_MODEL_CONTRACT'
    assert bound['coverage_tier']=='FULL_THREE_TASK'
    assert bound['shared_three_task_model_id']=='adcs-p2-innovation-residual-tcn'

def test_phm_profile_is_retrospective_only():
    r=registry()
    online=r.profile(ModelCapabilityQuery(domain='phm',tasks=('detection','localization','classification'),bound_artifact_contract=True))
    replay=r.profile(ModelCapabilityQuery(domain='phm',tasks=('detection','localization','classification'),online_required=False,allow_retrospective=True,bound_artifact_contract=True))
    assert online['coverage_tier']=='NO_COMPATIBLE_MODEL'
    assert 'REPLAY_TASK' in online['recommended_next_step']
    assert replay['coverage_tier']=='FULL_THREE_TASK'
    assert replay['shared_three_task_model_id']=='phm-v60-extra-trees-retrospective-screening'

def test_profile_preserves_raw_head_results_for_audit():
    p=registry().profile(ModelCapabilityQuery(domain='generic',channel_names=('x',),units={'x':'1'},tasks=('detection','localization','classification')))
    assert set(p['heads'])=={'detection','localization','classification'}
    assert p['heads']['detection']['compatible'][0]['model_id']=='astrograph-robust-baseline-deviation'
    assert p['heads']['classification']['compatible']==[]
