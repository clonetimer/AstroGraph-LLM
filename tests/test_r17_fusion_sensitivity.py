from astrograph.fusion import *
from astrograph.evaluation.fusion_sensitivity import *
from astrograph.evaluation.data_exposure import dataset_use

def r(src, vals):
 return FaultRanking(ranking_id=src.value,episode_id='e',source=src,producer_id='p',producer_version='1',candidates=[RankedFault(fault_id=f,rank=i+1,score=float(len(vals)-i),source=src) for i,f in enumerate(vals)])

def test_weight_sensitivity_is_exposed_not_called_calibration():
 d=r(RankingSource.DATA_DRIVEN,['A','B','C']); f=r(RankingSource.KG_FORWARD,['B','C','A']); b=r(RankingSource.KG_BACKWARD,['C','B','A'])
 scenarios=[FusionSensitivityScenario(scenario_id='base',config=BiRRFConfig(collapse_identical_kg_rankings=False)),FusionSensitivityScenario(scenario_id='data-heavy',config=BiRRFConfig(data_weight=5,collapse_identical_kg_rankings=False))]
 with dataset_use('SYNTHETIC_PROTOCOL','DEVELOPMENT','TUNE', group_ids=['r17-fusion-synth']):
  out=audit_birrf_sensitivity(d,f,b,scenarios,baseline_scenario_id='base')
 assert out.scenario_count==2 and out.calibration_status=='NOT_CALIBRATED_PERFORMANCE_STUDY'
 assert out.top1_stability_rate in {0.5,1.0}
