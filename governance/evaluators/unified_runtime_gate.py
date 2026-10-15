#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--trusted-gate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    e=json.loads(a.evaluation.read_text())
    t=json.loads(a.trusted_gate.read_text())
    checks=[]
    def add(cid,obs,expected,ok): checks.append({'criterion_id':cid,'observed':obs,'expected':expected,'status':'PASS' if ok else 'FAIL','severity':'HARD'})
    add('TRUSTED_DATA_PREREQUISITE',t.get('status'),'PASS',t.get('status')=='PASS')
    add('MINIMUM_CASE_COUNT',e.get('case_count'),'>=6',e.get('case_count',0)>=6)
    add('NO_DATA_ERRORS',e.get('data_error_count'),0,e.get('data_error_count')==0)
    add('NO_FALSE_DIAGNOSIS',e.get('false_diagnosis_count'),0,e.get('false_diagnosis_count')==0)
    add('MINIMUM_EXACT_DIAGNOSES',e.get('exact_diagnosis_count'),'>=4',e.get('exact_diagnosis_count',0)>=4)
    add('SAFE_UNRESOLVED_CASES',e.get('safe_non_conclusion_count'),'>=2',e.get('safe_non_conclusion_count',0)>=2)
    add('NO_TRUTH_USE',e.get('label_or_truth_use_count'),0,e.get('label_or_truth_use_count')==0)
    add('CLI_API_CONSISTENCY',e.get('cli_api_mismatch_count'),0,e.get('cli_api_mismatch_count')==0)
    add('BATCH_API_CONSISTENCY',e.get('batch_api_mismatch_count'),0,e.get('batch_api_mismatch_count')==0)
    add('DETERMINISTIC_REPLAY',e.get('deterministic_replay_mismatch_count'),0,e.get('deterministic_replay_mismatch_count')==0)
    status='PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL'
    out={'schema_version':'astrograph-unified-runtime-gate.v1','status':status,'checks':checks,'blockers':[c['criterion_id'] for c in checks if c['status']!='PASS'],'claim_boundary':'Unified runtime interoperability, fail-closed routing, deterministic replay and trusted minimum campaign behavior only; not broad accuracy or flight qualification.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2));return 0 if status=='PASS' else 2
if __name__=='__main__': raise SystemExit(main())
