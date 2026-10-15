#!/usr/bin/env python3
"""Independent Round-2 trusted-data acceptance gate.

This evaluator is intentionally outside the candidate source tree. It verifies a
candidate-produced audit against the frozen Round-2 policy and emits PASS, REJECT,
or BLOCKED without importing AstroGraph.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--manifest-jsonl',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    checks=[]; blockers=[]; rejections=[]
    try: audit=json.loads(a.audit.read_text(encoding='utf-8'))
    except Exception as exc:
        audit={};blockers.append(f'AUDIT_UNREADABLE:{type(exc).__name__}')
    rows=[]
    try:
        rows=[json.loads(line) for line in a.manifest_jsonl.read_text(encoding='utf-8').splitlines() if line.strip()]
    except Exception as exc: blockers.append(f'MANIFEST_UNREADABLE:{type(exc).__name__}')
    def ck(cid,ok,observed,expected,hard=True):
        checks.append({'criterion_id':cid,'status':'PASS' if ok else 'FAIL','observed':observed,'expected':expected,'severity':'HARD' if hard else 'SOFT'})
        if not ok: (blockers if cid in {'AUDIT_PROVENANCE','MANIFEST_INTEGRITY','NO_LABEL_LEAKAGE','NO_FALSE_ACCEPTANCE'} else rejections).append(cid)
    ck('AUDIT_PROVENANCE', bool(audit.get('dataset_tree_sha256') and audit.get('sat_agent_tree_sha256')), str(bool(audit.get('dataset_tree_sha256'))), 'source hashes present')
    ck('MANIFEST_INTEGRITY', len(rows)==audit.get('total_run_count')==108, f"rows={len(rows)};audit={audit.get('total_run_count')}", '108')
    ck('NO_LABEL_LEAKAGE', audit.get('public_trace_leakage_count')==0, str(audit.get('public_trace_leakage_count')), '0')
    false_accept=[r.get('run_id') for r in rows if r.get('acceptance_level')=='ACCEPTANCE_VALID' and r.get('full_gate_status')!='PASS']
    ck('NO_FALSE_ACCEPTANCE', not false_accept, str(false_accept[:5]), '[]')
    invalid_usage=[r.get('run_id') for r in rows if r.get('acceptance_level')!='ACCEPTANCE_VALID' and any(x in {'training','calibration','validation','acceptance'} for x in r.get('allowed_uses',[]))]
    ck('USE_RESTRICTIONS', not invalid_usage, str(invalid_usage[:5]), '[]')
    # Round-2 exit gate: a small but real regenerated acceptance-valid campaign is mandatory.
    a_count=int(audit.get('acceptance_valid_count',0)); pair_count=int(audit.get('acceptance_valid_pair_count',0))
    ck('ACCEPTANCE_VALID_CAMPAIGN', a_count>=12 and pair_count>=3, f'acceptance_valid={a_count};pairs={pair_count}', '>=12 runs and >=3 nominal/effect pairs')
    ck('REPRODUCTION_READY', audit.get('reproduction_ready') is True, str(audit.get('reproduction_ready')), 'true')
    if blockers: status='BLOCKED'; reasons=['Integrity/provenance violation prevents acceptance.']
    elif rejections: status='REJECT'; reasons=['Trusted-data protocol is implemented, but the frozen Round-2 data exit criteria are not met.']
    else: status='PASS'; reasons=['All frozen Round-2 trusted-data criteria passed.']
    out={'schema_version':'astrograph-trusted-data-gate.v1','status':status,'audit_sha256':sha256(a.audit) if a.audit.is_file() else '', 'manifest_sha256':sha256(a.manifest_jsonl) if a.manifest_jsonl.is_file() else '', 'checks':checks,'blockers':blockers,'rejections':rejections,'reasons':reasons,'claim_boundary':'Covers data provenance, usage classification, leakage and reproducibility only; not diagnosis accuracy or flight qualification.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2));return 0 if status=='PASS' else 2
if __name__=='__main__': raise SystemExit(main())
