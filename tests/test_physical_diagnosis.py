import numpy as np
import pandas as pd
import pytest
from tests.test_rw_window_mechanisms import frames
from astrograph.oag.physical_diagnosis import physical_signatures,diagnose_physical_complement,RW_FIELDS


def history():
    _,f=frames();a=f.copy();a['time_s']+=20
    n=pd.concat([f,a],ignore_index=True)
    return n.copy(),n

@pytest.mark.parametrize('wheel',range(4))
@pytest.mark.parametrize('kind',['jam','partial','offset'])
def test_specific_signature_locates_each_wheel(wheel,kind):
    s,n=history();c=f'adcs.rw.command_torque_nm_{wheel}';a=f'adcs.control.applied_torque_nm_{wheel}';w=f'adcs.rw.measured_speed_rad_s_{wheel}'
    if kind=='jam':s.loc[20:,a]=0;s.loc[20:,w]=.2;target='ADCS_RW_JAM'
    elif kind=='partial':s.loc[20:,a]=s.loc[20:,c]*.5;target='ADCS_RW_TORQUE_AUTHORITY_LOSS'
    else:s.loc[20:,w]+=20;target='ADCS_RW_SPEED_SENSOR_BIAS'
    d=diagnose_physical_complement(s,n)
    assert d['prediction']=={'kind':'CLASS','ids':[target]}
    assert d['localization']['wheel_index']==wheel


def test_short_persistent_segment_and_not_single_sample_spike():
    s,n=history();c='adcs.rw.command_torque_nm_0';a='adcs.control.applied_torque_nm_0'
    s.loc[36:,a]=s.loc[36:,c]*.5
    assert diagnose_physical_complement(s,n)['prediction']['ids']==['ADCS_RW_TORQUE_AUTHORITY_LOSS']
    s,n=history();s.loc[39,a]=0
    assert not physical_signatures(s,n)['hypotheses']


def test_history_distinguishes_motion_offset_from_sensor_offset():
    s,n=history();s.loc[20:,'adcs.rw.measured_speed_rad_s_0']+=20
    assert physical_signatures(s,n)['hypotheses']
    s.loc[5,'adcs.rw.command_torque_nm_1']+=.1
    assert not physical_signatures(s,n)['hypotheses']


def test_uncommanded_braking_is_not_proportional_torque_loss():
    s,n=history();s.loc[20:,'adcs.rw.measured_speed_rad_s_0']=np.linspace(50,3,20)
    s.loc[20:,'adcs.control.applied_torque_nm_0']=-.18
    d=diagnose_physical_complement(s,n)
    assert d['prediction']['ids']==['ADCS_RW_JAM']
    assert d['physical_evidence']['hypotheses'][0]['signature']=='UNCOMMANDED_BRAKING_AND_SPEED_DECAY'


def test_missing_frame_does_not_need_model_imputation():
    s,n=history();s.loc[20:,'adcs.control.applied_torque_nm_0']=0
    s.loc[30,RW_FIELDS]=np.nan;n.loc[30,RW_FIELDS]=np.nan
    d=diagnose_physical_complement(s,n)
    assert d['prediction']['kind']=='CLASS'
    assert d['physical_evidence']['receipt']['valid_pairs']==19
    assert not d['physical_evidence']['receipt']['imputation_performed']


def test_latest_missing_frame_has_explicit_age_and_too_stale_rejects():
    s,n=history();s.loc[20:,'adcs.control.applied_torque_nm_0']=0
    s.loc[39,RW_FIELDS]=np.nan;n.loc[39,RW_FIELDS]=np.nan
    d=physical_signatures(s,n);assert d['hypotheses'] and d['receipt']['observation_age_s']==1
    s.loc[38,RW_FIELDS]=np.nan;n.loc[38,RW_FIELDS]=np.nan
    assert physical_signatures(s,n)['status']=='INSUFFICIENT_OBSERVATIONS'


def test_multiple_wheel_faults_are_not_forced_to_a_single_location():
    s,n=history();s.loc[20:,'adcs.control.applied_torque_nm_0']=0;s.loc[20:,'adcs.control.applied_torque_nm_1']=0
    d=diagnose_physical_complement(s,n);assert d['prediction']['kind']=='SET' and d['localization'] is None


def test_gain_error_bound_blocks_unexcited_and_healthy_noisy_data():
    s,n=history();rng=np.random.default_rng(73)
    for j in range(4):
        f=f'adcs.control.applied_torque_nm_{j}';s[f]+=rng.uniform(-1e-6,1e-6,len(s));n[f]+=rng.uniform(-1e-6,1e-6,len(n))
    assert not physical_signatures(s,n)['hypotheses']
    s,n=history()
    for j in range(4):
        s[f'adcs.rw.command_torque_nm_{j}']=1e-8;s[f'adcs.control.applied_torque_nm_{j}']=0
    assert not physical_signatures(s,n)['hypotheses']

@pytest.mark.parametrize('bias',[0.,.001])
def test_time_alignment_removes_acceleration_not_true_gyro_bias(bias):
    s,n=history()
    for f in ['adcs.star_tracker.valid','adcs.star_tracker.solution_fresh']:s[f]=1;n[f]=1
    s['adcs.sensor.gyro_measured_rad_s_x']=.001*s.time_s+bias
    s['adcs.innovation.star_gyro_rate_residual_rad_s_x']=-.0005-bias
    ids={h['fault_id'] for h in physical_signatures(s,n)['hypotheses']}
    assert ('ADCS_GYRO_BIAS_DRIFT' in ids)==bool(bias)


def test_alignment_rejects_future_or_wrong_reference_and_short_input():
    s,n=history()
    with pytest.raises(ValueError):physical_signatures(s.iloc[:3],n)
    with pytest.raises(ValueError):physical_signatures(s,n[n.time_s!=30])
    with pytest.raises(ValueError):physical_signatures(s.iloc[::-1],n)
