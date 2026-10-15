from pathlib import Path
import json
import numpy as np
import pytest
from tests.test_rw_window_mechanisms import frames
from astrograph.oag.window_features import derive_window_statistics, derive_terminal_statistics
from astrograph.oag.degraded import mechanism_only_fallback
from astrograph.evidence import ConfiguredRuntimeEvidenceBuilder
from astrograph.simulation import SimulationRunCompleted

ROOT=Path(__file__).resolve().parents[1]
def event():
    return SimulationRunCompleted(event_id='EV-test',run_id='opaque',run_dir='opaque',nominal_run_dir='reference',runtime_id='unit-test')
def builder(terminal=False):
    return ConfiguredRuntimeEvidenceBuilder(ROOT/('configs/research/terminal8_zero_authority.v1.json' if terminal else 'configs/research/zero_authority.v1.json'))
def targets(s,n,terminal=False):
    a,b=(derive_terminal_statistics if terminal else derive_window_statistics)(s,n)
    bundle,gate=builder(terminal).build_from_frames(event(),source=a,nominal=b)
    assert gate.passed
    return {f.target_fault_id for f in bundle.evidence_facts if f.status.value=='SATISFIED'}

@pytest.mark.parametrize('wheel',range(4))
def test_zero_torque_effect_does_not_claim_motor_cause(wheel):
    s,n=frames();s[f'adcs.control.applied_torque_nm_{wheel}']=0
    ids=targets(s,n)
    assert 'ADCS_RW_TORQUE_AUTHORITY_LOSS' in ids
    assert 'ADCS_RW_MOTOR_FAILURE' not in ids


def test_zero_torque_at_rest_keeps_jam_alternative():
    s,n=frames();s['adcs.control.applied_torque_nm_0']=0;s['adcs.rw.measured_speed_rad_s_0']=.2
    assert {x for x in targets(s,n) if x.startswith('ADCS_RW_')}=={'ADCS_RW_JAM'}


def test_zero_torque_without_excitation_is_not_supported():
    s,n=frames();s['adcs.control.applied_torque_nm_0']=0;s['adcs.rw.command_torque_nm_0']=0
    assert 'ADCS_RW_TORQUE_AUTHORITY_LOSS' not in targets(s,n)

@pytest.mark.parametrize('kind',['jam','bias','torque'])
def test_terminal_persistence_after_transition(kind):
    s,n=frames()
    if kind=='jam':
        s.loc[12:,'adcs.control.applied_torque_nm_0']=0
        s.loc[12:,'adcs.rw.measured_speed_rad_s_0']=.2
        target='ADCS_RW_JAM'
    elif kind=='bias':
        s.loc[12:,'adcs.rw.measured_speed_rad_s_0']+=20
        target='ADCS_RW_SPEED_SENSOR_BIAS'
    else:
        s.loc[12:,'adcs.control.applied_torque_nm_0']*=.5
        target='ADCS_RW_TORQUE_AUTHORITY_LOSS'
    assert target not in targets(s,n)
    assert target in targets(s,n,True)
    assert len(s)==20 and not any('derived.' in c for c in s)


def test_one_point_spike_is_not_persistent_bias():
    s,n=frames();s.loc[19,'adcs.rw.measured_speed_rad_s_0']+=20
    assert 'ADCS_RW_SPEED_SENSOR_BIAS' not in targets(s,n,True)


def test_terminal_rejects_reverse_time_and_misalignment():
    s,n=frames()
    with pytest.raises(ValueError):derive_terminal_statistics(s.iloc[::-1],n.iloc[::-1])
    with pytest.raises(ValueError):derive_terminal_statistics(s,n.iloc[:-1])


def test_missing_packet_produces_candidates_without_model_or_imputation():
    s,n=frames();s['adcs.control.applied_torque_nm_0']=0
    fields=[c for c in s if c.startswith(('adcs.rw.','adcs.control.applied_torque'))]
    s.loc[14,fields]=np.nan;n.loc[14,fields]=np.nan
    schema=json.loads((ROOT/'configs/datasets/adcs_v5_2_2/telemetry_projection.json').read_text(encoding="utf-8"))['canonical_sha256']
    d=mechanism_only_fallback(builder=builder(),event=event(),source=s,nominal=n,schema_sha256=schema)
    assert d['candidate_fault_ids']==['ADCS_RW_TORQUE_AUTHORITY_LOSS']
    assert not d['model_invoked'] and not d['resampling_performed'] and d['confirmed_fault_id'] is None
    assert d['missing_times_s']==[14.] and d['valid_observations']==19
    s.loc[13,fields]=np.nan;s.loc[12,fields]=np.nan
    with pytest.raises(ValueError,match='90 percent'):mechanism_only_fallback(builder=builder(),event=event(),source=s,nominal=n,schema_sha256=schema)


def test_effect_scope_never_confirms_motor():
    from astrograph.oag.diagnostic_scope import diagnostic_scope
    d=diagnostic_scope('ADCS_RW_TORQUE_AUTHORITY_LOSS')
    assert d['diagnostic_level']=='FUNCTIONAL_EFFECT'
    assert not d['root_cause_confirmed'] and 'MOTOR' in d['root_cause_candidates']
    assert len(d['required_observations'])>=3


def test_dual_level_metrics_preserve_original_errors_and_abstention():
    from scripts.evaluate_diagnostic_enhancement import measure
    rows={0:{'target':'ADCS_RW_MOTOR_FAILURE','family':1,'prediction':{'kind':'CLASS','ids':['ADCS_RW_TORQUE_AUTHORITY_LOSS']},'degraded':None},
          1:{'target':'ADCS_RW_MOTOR_FAILURE','family':1,'prediction':{'kind':'SET','ids':['ADCS_RW_TORQUE_AUTHORITY_LOSS']},'degraded':None}}
    assert measure(rows)['wrong']==1
    m=measure(rows,effect=True)
    assert m['correct']==1 and m['candidate_hit']==2 and m['decided']==1 and m['windows']==2


def test_zero_package_preserves_qualified_legacy_observation_operators():
    old=json.loads((ROOT/'configs/datasets/adcs_v5_2_2/runtime_window_integrated_evidence.json').read_text(encoding="utf-8"))
    new=json.loads((ROOT/'configs/research/zero_authority.v1.json').read_text(encoding="utf-8"))
    for key in ['channels','feature_definitions','source_schema_sha256','package_id']:
        assert old[key]==new[key]
    original={x['evidence_definition_id']:x for x in old['evidence_definitions']}
    added={x['evidence_definition_id']:x for x in new['evidence_definitions']}
    assert all(added[key]==value for key,value in original.items())


def test_zero_rule_cannot_grant_detector_override():
    from astrograph.oag.complementary import mechanism_confirmation_candidate
    import hashlib
    fid='ADCS_RW_TORQUE_AUTHORITY_LOSS'
    d={'status':'CANDIDATE_SET','detector_mechanism_disagreement':True,'blocking_conflicts':[],
       'candidate_fault_ids':[fid],'data_candidate_count':1,'data_candidate_source':'PRE_HIERARCHY_CLASS_DISTRIBUTION',
       'support_by_fault':{fid:{'complete_mechanism_support':True,'data_candidate':True}},
       'evidence_state_matrix':[{'fault_id':fid,'state':'SATISFIED','definition_id':'ED-RW-WINDOW-0-ADCS_RW_TORQUE_AUTHORITY_LOSS-ZERO-V1'}]}
    sha=hashlib.sha256((ROOT/'configs/research/zero_authority.v1.json').read_bytes()).hexdigest()
    assert mechanism_confirmation_candidate(d,evidence_package_sha256=sha) is None
    d['evidence_state_matrix'][0]['definition_id']='ED-RW-WINDOW-0-ADCS_RW_TORQUE_AUTHORITY_LOSS-V1'
    assert mechanism_confirmation_candidate(d,evidence_package_sha256=sha)==fid
    assert mechanism_confirmation_candidate(d,evidence_package_sha256='0'*64) is None
