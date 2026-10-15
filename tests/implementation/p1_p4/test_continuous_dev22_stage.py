from pathlib import Path

from astrograph.agents.mission.continuous_diagnosis import CaseTimelineStore, make_timeline_event, recommend_next_observations

ROOT=Path(__file__).resolve().parents[3]


def test_stage90_timeline_tracks_diagnosis_revision_and_is_idempotent(tmp_path:Path):
    store=CaseTimelineStore(tmp_path/'timeline')
    first=make_timeline_event(case_id='CASE-A',task_id='T1',observed_at='2026-09-15T01:00:00Z',
        diagnosis={'diagnosis':{'selected_fault_id':'F1','candidate_fault_ids':['F1','F2'],'detected':True}})
    second=make_timeline_event(case_id='CASE-A',task_id='T2',observed_at='2026-09-15T01:05:00Z',
        diagnosis={'diagnosis':{'selected_fault_id':'F2','candidate_fault_ids':['F2'],'detected':True}})
    store.append(first);store.append(first);store.append(second)
    trajectory=store.trajectory('CASE-A')
    assert trajectory['event_count']==2
    assert trajectory['diagnosis_revision_count']==1
    assert trajectory['latest']['selected_fault_id']=='F2'


def test_stage90_next_best_observation_discriminates_rw_faults():
    result=recommend_next_observations(candidate_fault_ids=['ADCS_RW_FRICTION_INCREASE','ADCS_RW_JAM'],
        observed_channels=['sensor.adcs.rw.motor_current_a_0'],fault_catalog_path=ROOT/'configs/fault_modes/lab_fault_catalog.json')
    assert result['status']=='AVAILABLE'
    channels=[x['channel'] for x in result['recommendations']]
    assert 'sensor.adcs.rw.speed_rad_s_0' in channels
    assert result['control_action_authorized'] is False
