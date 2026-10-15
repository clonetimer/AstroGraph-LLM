"""Recompute full three-task metrics from persisted, truth-joined records."""
import argparse
import json
from pathlib import Path
from astrograph.evaluation.three_task_replay import evaluate_three_task_replay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--branches',nargs='+',default=['data_only','data_plus_reference','dev2_single','sensor_initial','sensor_eager','sensor_agent','sensor_working'])
    a=p.parse_args();config=a.project_root/'configs/datasets/adcs_v5_2_2'
    mapping=json.loads((config/'fault_class_map.json').read_text())['mappings']
    hierarchy=json.loads((config/'diagnosis_hierarchy.json').read_text())['fault_to_component']
    out=evaluate_three_task_replay(json.loads(a.records.read_text()),fault_to_component={mapping[k]:v for k,v in hierarchy.items()},dataset_class_to_fault=mapping,branches=a.branches)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')

if __name__=='__main__':main()
