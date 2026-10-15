from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from astrograph.oag.sensor_checks import sensor_statistics,channel_zero_count,quaternion_spread,ZERO_ATOL
from astrograph.evidence.executable import FeatureResult,EvidenceDefinition,evaluate_evidence
from astrograph.oag.agent import AutonomousDiagnosisService
from astrograph.oag.kernel import EvidenceDiagnosisKernel
from astrograph.oag.three_task import ThreeTaskResult
from astrograph.oag.evidence_tools import FeatureSeries,MeasurementBank,tools_from_measurements
from astrograph.orchestration.models import WorkflowContext,PermissionScope


def frames():
    t=np.arange(20,dtype=float);g=np.column_stack((1e-3+1e-4*np.sin(t),2e-3+1e-4*np.cos(t),1e-3+1e-5*t))
    q=np.column_stack((np.cos(.001*t),np.sin(.001*t),np.zeros(20),np.zeros(20)))
    f=pd.DataFrame({'time_s':t,'adcs.star_tracker.valid':1.,'adcs.star_tracker.solution_fresh':1.})
    for j,x in enumerate('xyz'):
        f['adcs.sensor.gyro_measured_rad_s_'+x]=g[:,j]
        f['adcs.sensor.accel_measured_m_s2_'+x]=g[:,j]*.1
        f['adcs.innovation.star_gyro_rate_residual_rad_s_'+x]=0.
    for j in range(4):f['adcs.star_tracker.q'+str(j)]=q[:,j]
    return f.copy(),f.copy()


def test_paired_nominal_does_not_generate_positive_sensor_effects():
    a,b=frames();s=sensor_statistics(a,b)
    assert s['gyro_zero_axes']==0 and s['gyro_aligned_bias_norm']==0
    assert s['star_output_frozen']==0 and s['imu_all_output_zero']==0


def test_exact_axis_and_total_zero_are_distinct_observations():
    a,b=frames();a['adcs.sensor.gyro_measured_rad_s_x']=0
    assert sensor_statistics(a,b)['gyro_zero_axes']==1
    for x in 'xyz':a['adcs.sensor.gyro_measured_rad_s_'+x]=0
    assert sensor_statistics(a,b)['imu_all_output_zero']==0
    for x in 'xyz':a['adcs.sensor.accel_measured_m_s2_'+x]=0
    assert sensor_statistics(a,b)['imu_all_output_zero']==1


def test_quaternion_sign_changes_do_not_look_like_new_attitude():
    q=np.tile([1.,0,0,0],(20,1));q[1::2]*=-1
    assert quaternion_spread(q)==0


def test_frozen_source_requires_excited_reference():
    a,b=frames()
    for j in range(4):a['adcs.star_tracker.q'+str(j)]=b['adcs.star_tracker.q'+str(j)].iloc[0]
    assert sensor_statistics(a,b)['star_output_frozen']==1
    assert 'star_output_frozen' not in sensor_statistics(a,a)


def test_bad_star_quality_blocks_gyro_bias_statistics():
    a,b=frames();a['adcs.star_tracker.valid']=0
    s=sensor_statistics(a,b)
    assert 'gyro_aligned_bias_norm' not in s
    assert 'star_output_frozen' not in s


def test_noisy_residual_not_constant_bias():
    a,b=frames();a['adcs.innovation.star_gyro_rate_residual_rad_s_x']=np.tile([1e-3,-1e-3],10)
    assert sensor_statistics(a,b)['gyro_aligned_spread_ratio']>.25


def test_constant_interval_bias_is_retained():
    a,b=frames();a['adcs.innovation.star_gyro_rate_residual_rad_s_x']=1e-4
    s=sensor_statistics(a,b)
    assert s['gyro_aligned_bias_norm']==pytest.approx(1e-4)
    assert s['gyro_aligned_spread_ratio']<1e-9


@pytest.mark.parametrize('kind',['missing','nan','time','short'])
def test_missing_or_invalid_data_never_converted_to_zero(kind):
    a,b=frames()
    if kind=='missing':a=a.drop(columns='adcs.sensor.gyro_measured_rad_s_x')
    if kind=='nan':a.loc[0,'adcs.sensor.gyro_measured_rad_s_x']=np.nan
    if kind=='time':a.loc[0,'time_s']=3
    if kind=='short':a=a.iloc[:4];b=b.iloc[:4]
    if kind in {'time','short'}:
        with pytest.raises(ValueError):sensor_statistics(a,b)
    else:assert 'gyro_zero_axes' not in sensor_statistics(a,b)


def test_missing_value_keeps_episode_provenance():
    f=FeatureResult(feature_result_id='F',feature_definition_id='FD',episode_id='E',run_id='R',paired_nominal_run_id='N',resolved_field='x',component_instance='sensor',value=None,unit='1',window_start_s=0,window_end_s=2,sample_count=0,algorithm_version='v',input_data_sha256='a'*64)
    d=EvidenceDefinition(evidence_definition_id='ED',target_fault_id='A',name='a',feature_definition_id='FD',role='REQUIRED',comparison='GREATER_THAN',threshold=1,threshold_unit='1')
    fact=evaluate_evidence(d,f)
    assert fact.episode_id=='E' and fact.feature_result_id=='F' and fact.status.value=='MISSING_DATA'
    assert evaluate_evidence(d,None).episode_id=='UNKNOWN'


def test_negative_detector_can_still_trigger_available_check(make_diagnostic_case):
    data,graph=make_diagnostic_case(a=0,b=None,counter=None)
    head=ThreeTaskResult(episode_id=data.bundle.episode.episode_id,prediction_id='p',model_id='test-fixture',model_version='1',input_data_sha256='a'*64,window_start_s=0,window_end_s=2,detection_probability=.1,detection_threshold=.5,detected=False,component_probabilities={'sensor':1.},predicted_component='sensor',classification_ranking=data.diagnostic_ranking)
    data=data.model_copy(update={'three_task_result':head})
    bank=MeasurementBank(episode_id=data.bundle.episode.episode_id,series=(FeatureSeries(feature_definition_id='F_b',field='sensor.b',unit='1',time_s=(0,1,2),current=(2,2,2),available_at_s=2),))
    context=WorkflowContext(workflow_id='w',run_id='r',actor='agent',permissions=[PermissionScope.RUN_DIAGNOSIS,PermissionScope.READ_DATA,PermissionScope.READ_KNOWLEDGE])
    out=AutonomousDiagnosisService(EvidenceDiagnosisKernel(graph),tools=tools_from_measurements(bank,data.bundle)).run(data,context=context)
    assert out['initial_diagnosis']['status']=='NO_DETECTED_ANOMALY'
    assert out['checks_executed']==1
    assert out['diagnosis']['detector_mechanism_disagreement']
    assert out['diagnosis']['status']=='CANDIDATE_SET'
    assert out['diagnosis']['selected_fault_id'] is None


def test_independent_actuator_fault_blocks_specific_gyro_attribution():
    a,b=frames()
    for j in range(4):
        for f in (a,b):
            f[f'adcs.rw.command_torque_nm_{j}']=.01
            f[f'adcs.control.applied_torque_nm_{j}']=.01
    assert sensor_statistics(a,b)['actuator_response_consistent']==1
    a['adcs.control.applied_torque_nm_0']=0.
    assert sensor_statistics(a,b)['actuator_response_consistent']==0


def test_missing_actuator_response_is_unknown_not_consistent():
    a,b=frames()
    assert 'actuator_response_consistent' not in sensor_statistics(a,b)
