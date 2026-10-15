#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from astrograph.evaluation.fusion_sensitivity import FusionSensitivityScenario, audit_birrf_sensitivity
from astrograph.fusion import BiRRFConfig, FaultRanking

def load(path:Path)->dict:
    return json.loads(path.read_text(encoding='utf-8'))

def main()->int:
    ap=argparse.ArgumentParser(description='Offline BiRRF engineering-parameter sensitivity audit; not calibration.')
    ap.add_argument('--data-ranking',required=True,type=Path);ap.add_argument('--forward-ranking',required=True,type=Path);ap.add_argument('--backward-ranking',required=True,type=Path)
    ap.add_argument('--design',type=Path,default=Path('configs/evaluation/birrf_sensitivity_r17.v1.json'));ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args(); design=load(a.design)
    scenarios=[]
    for row in design['scenarios']:
        cfg={k:v for k,v in row.items() if k!='scenario_id'}
        scenarios.append(FusionSensitivityScenario(scenario_id=row['scenario_id'],config=BiRRFConfig(**cfg)))
    report=audit_birrf_sensitivity(FaultRanking.model_validate(load(a.data_ranking)),FaultRanking.model_validate(load(a.forward_ranking)),FaultRanking.model_validate(load(a.backward_ranking)),scenarios,baseline_scenario_id=design['baseline_scenario_id'])
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(report.model_dump_json(indent=2)+'\n',encoding='utf-8'); return 0
if __name__=='__main__': raise SystemExit(main())
