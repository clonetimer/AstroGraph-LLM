"""Independent archived-live paired ablation: no new model or HTTP requests."""
from __future__ import annotations
import sys,argparse,zipfile,json,statistics
from pathlib import Path,PurePosixPath
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from common import *

def main(args):
    from astrograph.evaluation.agent_quality.metrics import inspect_task
    from astrograph.agents.mission.storage import canonical as runtime_canonical
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    plan={'scope':'ARCHIVED_PAIRED_LIVE_COMPARISON_NOT_NEW_LLM_RUN','source_zip_sha256':sha_file(args.archive),
      'workspaces':['quality02','quality-checks02'],'new_generation_requests':0,'chosen_before_scoring':True,
      'selection':'quality02 is the completed 40-slot protocol; quality01 partial predecessor excluded, not pooled',
      'core_runtime_modified':False}
    write(out/'plan.json',plan);event(out,'PLAN_FROZEN',sha256=sha_file(out/'plan.json'))
    allrows=[];groups={};errors=[]
    with zipfile.ZipFile(args.archive) as z:
      names=set(z.namelist())
      for workspace in plan['workspaces']:
        prefix='AstroGraph-local/'+workspace+'/'
        exp=json.loads(z.read(prefix+'experiment.json'))
        assert sha_obj(exp) # use byte checksum below, not current order
        assert __import__('hashlib').sha256(z.read(prefix+'experiment.json')).hexdigest()==json.loads(z.read(prefix+'experiment-receipt.json'))['sha256']
        write(out/(workspace+'.experiment.json'),exp)
        for rel,dig in exp['input_files'].items():
          assert __import__('hashlib').sha256(z.read(prefix+rel)).hexdigest()==dig,rel
        reports=[]
        for cell in exp['cells']:
          cid=cell['cell_id'];base=prefix+'trials/'+cid+'/'
          try:
            trial=json.loads(z.read(base+'trial.json'))
            assert __import__('hashlib').sha256(z.read(base+'trial.json')).hexdigest()==json.loads(z.read(base+'receipt.json'))['sha256']
            tp=base+'tasks/'+cid+'/'
            dest=out/'verified_tasks'/workspace/cid
            dest.mkdir(parents=True)
            for n in sorted(names):
              if not n.startswith(tp) or n.endswith('/'):continue
              rel=n[len(tp):];pp=PurePosixPath(rel)
              if '..' in pp.parts or pp.is_absolute():raise ValueError('UNSAFE_ARCHIVE_PATH')
              path=dest.joinpath(*pp.parts);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(n))
            quality=inspect_task(dest)
            archived=json.loads(z.read(base+'quality.json'))
            assert quality==archived,'RESCORE_MISMATCH'
            write(out/'quality'/workspace/(cid+'.json'),quality)
            reports.append((cell,quality,trial))
            allrows.append({'workspace':workspace,**cell,'mission_status':quality['mission_status'],'diagnosis':quality['diagnosis'],
             'checks':quality['checks']['executed'],'admitted':quality['checks']['admitted_feature_result_count'],
             'retrieval_calls':quality['retrieval']['executed_calls'],'new_texts':[r['new_texts'] for r in quality['retrieval']['calls']],
             'http_attempts':quality['http_attempts'],'elapsed_s':quality['elapsed_s'],'end_to_end_elapsed_s':trial['elapsed_s'],'wire':quality['wire_status'],'full_wire':quality['full_wire_pairs'],
             'llm_status':quality['llm_status'],'fallbacks':quality['fallbacks'],'quality_recomputed_equal':True,'result_sha256':quality['result_sha256']})
          except Exception as exc:errors.append({'workspace':workspace,'cell':cid,'type':type(exc).__name__,'error':str(exc)})
        bucket=defaultdict(list)
        for cell,q,t in reports:bucket[(cell['case_id'],cell['mode'],cell['planner'])].append(q)
        summary=[]
        import numpy as np
        for key,rs in sorted(bucket.items()):
          end_to_end=[t['elapsed_s'] for c,q,t in reports if (c['case_id'],c['mode'],c['planner'])==key]
          summary.append({'case':key[0],'mode':key[1],'planner':key[2],'n':len(rs),'mission':dict(Counter(r['mission_status'] for r in rs)),
             'final':dict(Counter(str((r['diagnosis']['selected'],r['diagnosis']['status'])) for r in rs)),
             'mean_retrieval_calls':statistics.mean(r['retrieval']['executed_calls'] for r in rs),
             'mean_checks':statistics.mean(r['checks']['executed'] for r in rs),'mean_admitted_features':statistics.mean(r['checks']['admitted_feature_result_count'] for r in rs),
             'http_attempts':sum(r['http_attempts'] for r in rs),'end_to_end_p50_s':float(np.quantile(end_to_end,.5)),'end_to_end_p95_s':float(np.quantile(end_to_end,.95)),'event_clock_p50_s':float(np.quantile([r['elapsed_s'] for r in rs],.5)),'p95_s':float(np.quantile([r['elapsed_s'] for r in rs],.95))})
        pairs=[];lookup={(c['case_id'],c['mode'],c['repeat'],c['planner']):q for c,q,_ in reports}
        for key,d in lookup.items():
          if key[-1]!='deterministic':continue
          l=lookup.get((*key[:3],'llm_assisted'))
          if l is None:continue
          pairs.append({'case':key[0],'mode':key[1],'repeat':key[2],
           'diagnosis_equal':all(d['diagnosis'].get(k)==l['diagnosis'].get(k) for k in ['status','selected','candidate_fault_ids','root_cause_confirmed']),
           'extra_checks':l['checks']['executed']-d['checks']['executed'],
           'extra_admitted':l['checks']['admitted_feature_result_count']-d['checks']['admitted_feature_result_count'],
           'extra_retrieval':l['retrieval']['executed_calls']-d['retrieval']['executed_calls'],
           'delta_s':l['elapsed_s']-d['elapsed_s'],'corrected_errors':None,'induced_errors':None})
        groups[workspace]={'planned':len(exp['cells']),'verified':len(reports),'arms':summary,'pairs':pairs,
           'declared_model_parameters':exp['declared_generation_parameters'],'model_identity':exp['backend_identity']}
    write(out/'rows.json',allrows);write(out/'report.json',{'scope':plan['scope'],'groups':groups,'errors':errors,'new_generation_requests':0,
      'quality_interpretation':'Matched archived systems; LLM affects preparation, check selector and prose together, not a single isolated planning phase; repeated trials are not independent faults.'})
    event(out,'REVIEW_COMPLETED',verified=len(allrows),errors=len(errors));print(json.dumps({'verified':len(allrows),'errors':errors[:4]}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
