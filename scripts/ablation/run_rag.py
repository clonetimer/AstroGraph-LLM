"""RAG-budget × planner factorial on a pinned already-used ADCS slice.

Fresh LLM variants require the user's reachable, privately configured service.
Deterministic arms are executable offline. No recorded response is reused as a
counterfactual LLM output. Memory, training and publication stay disabled.
"""
from __future__ import annotations
import sys,argparse,zipfile,time
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from common import *


def init(args):
    from astrograph.agents.mission.models import AgentSettings,AgentRequest
    from astrograph.agents.mission.modes import resolve_mode
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    settings=AgentSettings.model_validate(read(args.settings)) if args.settings else AgentSettings(project_root=str(ROOT))
    req=None;prefix='AstroGraph-local/quality02/inputs/adcs-slice/'
    with zipfile.ZipFile(args.archive) as z:
        req=read_zip_request=__import__('json').loads(z.read(prefix+'request.json'))
        paths={}
        for n in z.namelist():
            if not n.startswith(prefix) or n.endswith('/') or n.endswith('/request.json'):continue
            rel=n[len(prefix):];p=PurePosixPath(rel)
            if p.is_absolute() or '..' in p.parts:raise ValueError('UNSAFE_PATH')
            dest=out/'inputs'/Path(*p.parts);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(n));paths[rel]=sha_file(dest)
        req['artifacts']={k:str(out/'inputs'/('graph' if k=='graph_dir' else Path(v).name)) for k,v in req['artifacts'].items()}
        req['documents']=[str(out/'inputs'/Path(v).name) for v in req['documents']]
        req.update(task_id='rag-factorial',case=None,work_mode='thinking')
        AgentRequest.model_validate(req);write(out/'request.json',req)
    cells=[]
    for repeat in range(args.repeats):
        policies=['deterministic','llm_assisted'] if repeat%2==0 else ['llm_assisted','deterministic']
        for policy in policies:
            for cap in (0,1,2):cells.append({'id':f'{policy}-rag{cap}-r{repeat+1:02d}','planner':policy,'max_optional_actions':cap,'repeat':repeat+1})
    resolved,_=resolve_mode(settings,AgentRequest.model_validate(req),backend_available=bool(settings.llm_config))
    if resolved.max_optional_actions<2:raise PermissionError('HOST_MUST_ALLOW_TWO_OPTIONAL_ACTIONS_FOR_THIS_MATRIX')
    plan={'scope':'ONE_ALREADY_USED_ADCS_SLICE_RAG_PLANNER_FACTORIAL_NOT_ACCURACY','created_at':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
      'project':str(ROOT),'settings_path':str(args.settings.resolve()) if args.settings else None,'settings_sha256':sha_file(args.settings) if args.settings else None,
      'llm_config_sha256':sha_file(settings.llm_config) if settings.llm_config else None,
      'source_archive_sha256':sha_file(args.archive),'inputs':paths,'request_sha256':sha_file(out/'request.json'),
      'repeats':args.repeats,'cells':cells,'frozen_runtime':{str(p.relative_to(ROOT)):sha_file(p) for p in sorted((ROOT/'src/astrograph').rglob('*.py'))},
      'budgets':{k:getattr(resolved,k) for k in ('max_checks','max_llm_calls','max_http_attempts','max_plan_revisions','task_timeout_s')},
      'scope_control':'same documents processed in all arms; only retrieval action allowance 0/1/2 and planner change; fixed checks, memory off; actual execution may use less than allowance',
      'live':'explicit --allow-live and frozen private settings required; unavailable LLM slots kept NOT_RUN',
      'labels':None,'output_truth':'No new accuracy estimate from a single input or no-label runs'}
    write(out/'plan.json',plan);write(out/'plan-receipt.json',{'sha256':sha_file(out/'plan.json')});event(out,'PLAN_FROZEN',sha256=sha_file(out/'plan.json'))
    print(canonical({'status':'INITIALIZED','cells':len(cells),'live_configured':bool(settings.llm_config)}))


def verify(out):
    p=read(out/'plan.json')
    if sha_file(out/'plan.json')!=read(out/'plan-receipt.json')['sha256']:raise ValueError('PLAN_CHANGED')
    if sha_file(out/'request.json')!=p['request_sha256']:raise ValueError('REQUEST_CHANGED')
    for name,digest in p['inputs'].items():
        if sha_file(out/'inputs'/name)!=digest:raise ValueError('INPUT_CHANGED')
    for name,digest in p['frozen_runtime'].items():
        if sha_file(ROOT/name)!=digest:raise ValueError('RUNTIME_CHANGED')
    if p['settings_path'] and sha_file(p['settings_path'])!=p['settings_sha256']:raise ValueError('SETTINGS_CHANGED')
    return p


def execute(args):
    from astrograph.agents.mission.models import AgentSettings,AgentRequest
    from astrograph.agents.mission.service import MissionAgent
    from astrograph.evaluation.agent_quality.metrics import inspect_task
    out=args.output.resolve();plan=verify(out)
    settings=AgentSettings.model_validate(read(plan['settings_path'])) if plan['settings_path'] else AgentSettings(project_root=str(ROOT))
    if settings.llm_config and sha_file(settings.llm_config)!=plan['llm_config_sha256']:raise ValueError('PRIVATE_LLM_CONFIG_CHANGED')
    started=0
    for cell in plan['cells']:
        p=out/'trials'/cell['id'];receipt=p/'receipt.json'
        if receipt.exists():
            if sha_file(p/'trial.json')!=read(receipt)['sha256']:raise ValueError('TRIAL_CHANGED')
            continue
        if cell['planner']=='llm_assisted' and (not args.allow_live or not settings.llm_config):continue
        if p.exists():raise RuntimeError('INCOMPLETE_SLOT_REQUIRES_REVIEW_NOT_AUTOREPLAY')
        p.mkdir(parents=True);event(out,'TRIAL_STARTED',cell=cell['id'])
        req=AgentRequest.model_validate(read(out/'request.json')).model_copy(update={'task_id':cell['id']})
        conf=settings.model_copy(update={**plan['budgets'],'project_root':str(ROOT),'read_roots':list(dict.fromkeys([*settings.read_roots,str(out/'inputs'),str(ROOT)])),
           'output_root':str(p/'tasks'),'planner':cell['planner'],'max_optional_actions':cell['max_optional_actions'],
           'work_mode_budgets':{},'memory_db':None,'memory_policy':None,'allow_external_literature':False,
           'approved_training_manifest_sha256':[]})
        t=time.monotonic();record={'cell':cell}
        try:
            r=MissionAgent(conf).run(req);q=inspect_task(p/'tasks'/cell['id']);write(p/'quality.json',q)
            record.update(status=r.status,quality=q)
        except Exception as exc:record.update(status='FAILED',error={'type':type(exc).__name__,'message':str(exc)})
        record['elapsed_including_launch_s']=time.monotonic()-t;write(p/'trial.json',record);write(receipt,{'sha256':sha_file(p/'trial.json')})
        event(out,'TRIAL_COMMITTED',cell=cell['id'],status=record['status']);started+=1
        if args.max_runs and started>=args.max_runs:break
    report=[]
    for cell in plan['cells']:
        p=out/'trials'/cell['id']/'trial.json';r=read(p) if p.exists() else {'status':'NOT_RUN','reason':'LIVE_BACKEND_REQUIRED_OR_RUN_LIMIT'}
        report.append({'cell':cell,**r})
    # Append a numbered summary, never overwrite a previous result.
    seq=len(list(out.glob('report-*.json')))+1;write(out/f'report-{seq:03d}.json',{'scope':plan['scope'],'rows':report,'new_generation':'Only indicated completed llm_assisted slots send live generation requests','semantic_relevance':None})
    print(canonical({'cells':len(report),'status':dict(__import__('collections').Counter(r['status'] for r in report))}))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
 q=sub.add_parser('init');q.add_argument('--archive',type=Path,required=True);q.add_argument('--settings',type=Path);q.add_argument('--output',type=Path,required=True);q.add_argument('--repeats',type=int,default=3)
 q=sub.add_parser('run');q.add_argument('--output',type=Path,required=True);q.add_argument('--allow-live',action='store_true');q.add_argument('--max-runs',type=int)
 a=p.parse_args()
 if a.cmd=='init':
  if not 1<=a.repeats<=20:p.error('repeats must be 1..20')
  init(a)
 else:execute(a)
