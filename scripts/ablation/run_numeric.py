"""Synthetic numeric and memory ablations through the unmodified core."""
from __future__ import annotations
import sys,argparse
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from common import *

def main(args):
    from astrograph.evaluation.agent_quality.fixtures import build_numeric_cases
    from astrograph.oag.contracts import OAGInput
    from astrograph.oag.evidence_tools import MeasurementBank,tools_from_measurements
    from astrograph.oag.kernel import EvidenceDiagnosisKernel
    from astrograph.oag.agent import AutonomousDiagnosisService,CaseContext,_symptoms
    from astrograph.memory.service import CaseMemoryService,MemoryPolicy,PendingCase,CheckExperience
    from astrograph.orchestration.models import WorkflowContext,PermissionScope
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cases=build_numeric_cases(ROOT,out/'fixtures');write(out/'cases.json',cases)
    arms=['budget0','budget1','budget2','budget3','common_first_budget2','reverse_order_budget2','without_counter_tool','without_b_tool',
          'memory_empty','memory_counter','memory_common','memory_pending','memory_future','memory_same_source']
    write(out/'plan.json',{'scope':'SYNTHETIC_FIXED_SCENARIO_MECHANISM_ABLATION_NOT_ADCS_ACCURACY','arms':arms,'cases':cases,
       'labels':'evaluator-only file never read by selectors','no_real_LLM':True,'memory':'isolated SQLite, explicitly synthetic historical confirmations, no production DB or PPL',
       'memory_identity':'source-trace producer and synthetic-evaluator distinct; reviewer is simulated not a real human',
       'known_previous_results':'existing three fixtures previously used; new arms frozen before execution, not new holdout'})
    event(out,'PLAN_FROZEN',sha256=sha_file(out/'plan.json'));rows=[]
    T=datetime(2026,9,1,tzinfo=timezone.utc)
    for c in cases:
        req=read(c['request_path']);data=OAGInput.model_validate(read(req['artifacts']['input']));root_sha=sha_obj(data.model_dump(mode='json'))
        bank=MeasurementBank.model_validate(read(req['artifacts']['measurements']));kernel=EvidenceDiagnosisKernel(req['artifacts']['graph_dir'])
        for arm in arms:
            directory=out/'runs'/c['case_id']/arm;directory.mkdir(parents=True)
            ctx=WorkflowContext(workflow_id='ablation',run_id=c['case_id']+'-'+arm,actor='evaluation-agent',
                permissions=[PermissionScope.RUN_DIAGNOSIS,PermissionScope.READ_DATA,PermissionScope.READ_KNOWLEDGE,PermissionScope.READ_MEMORY,PermissionScope.WRITE_MEMORY])
            tools=tools_from_measurements(bank,data.bundle);selector=None;memory=None;case=None;budget=2
            if arm.startswith('budget'):budget=int(arm[-1])
            if arm=='without_counter_tool':tools=tuple(t for t in tools if not t.tool_id.endswith('F_counter'))
            if arm=='without_b_tool':tools=tuple(t for t in tools if not t.tool_id.endswith('F_b'))
            if arm=='common_first_budget2':
                selector=lambda **p:next((x['tool_id'] for x in p['options'] if x['tool_id'].endswith('F_common')),p['options'][0]['tool_id'])
            if arm=='reverse_order_budget2':selector=lambda **p:p['options'][-1]['tool_id']
            if arm.startswith('memory'):
                pol=MemoryPolicy(profile_sha256='0'*64,approved_reviewers=('synthetic-evaluator',))
                memory=CaseMemoryService(directory/'memory.sqlite',pol)
                case=CaseContext(case_id='current-'+c['case_id'],source_group_id='CURRENT-INDEPENDENT-GROUP',operating_mode='tracking',observed_end=T+timedelta(days=2))
                if arm!='memory_empty':
                    fid='F_common' if arm=='memory_common' else 'F_counter'
                    source_group=case.source_group_id if arm=='memory_same_source' else 'SYNTHETIC-PRIOR-GROUP'
                    prior=PendingCase(case_id='synthetic-history',source_group_id=source_group,producer_actor='history-producer',
                        symptom_ids=_symptoms(data.bundle),operating_mode='tracking',proposed_fault_id='B',evidence_ids=('SYNTHETIC-HISTORY-EVIDENCE',),
                        observed_end=T,end_reason='completed',output_valid=True,knowledge_version=kernel.knowledge_version,
                        checks=(CheckExperience(tool_id='telemetry.compute:'+fid,feature_definition_ids=(fid,),outcome='ADMITTED'),))
                    producer=ctx.model_copy(update={'actor':'history-producer'})
                    memory.stage_case(prior,event_id='stage-history',context=producer)
                    if arm!='memory_pending':
                        review=ctx.model_copy(update={'actor':'synthetic-evaluator','permissions':[PermissionScope.REVIEW_MEMORY]})
                        memory.confirm_case(case_id='synthetic-history',event_id='confirm-history',verified_fault_id='B',
                           truth_source_ref='SYNTHETIC_FIXTURE_EVALUATOR_NOT_HUMAN_OR_FLIGHT_EVIDENCE',verified_at=T+timedelta(hours=1),
                           available_at=T+timedelta(days=3) if arm=='memory_future' else T+timedelta(hours=2),expected_snapshot=0,context=review)
            try:
                result=AutonomousDiagnosisService(kernel,tools=tools,memory=memory,max_checks=budget,plan_selector=selector).run(data,context=ctx,case=case)
                assert sha_obj(data.model_dump(mode='json'))==root_sha
                write(directory/'result.json',result)
                r={'case':c['case_id'],'arm':arm,'run_status':'SUCCEEDED','budget':budget,'checks':result['checks_executed'],
                   'sequence':[t['tool_id'] for t in result['planning_trace']],
                   'admitted':sum(len(t.get('admitted_evidence_ids',[])) for t in result['planning_trace']),
                   'status':result['diagnosis']['status'],'selected':result['diagnosis']['selected_fault_id'],
                   'knowledge_verified':result['diagnosis']['knowledge_verified'],'stop':result['stopping_reason'],'memory':result['memory'],
                   'result_sha256':sha_file(directory/'result.json'),'correctness':None}
                rows.append(r);event(out,'CELL_COMPLETE',case=c['case_id'],arm=arm,result_sha256=r['result_sha256'])
            except Exception as exc:
                rows.append({'case':c['case_id'],'arm':arm,'run_status':'FAILED','error':type(exc).__name__+':'+str(exc)})
                event(out,'CELL_FAILED',**rows[-1])
    write(out/'report.json',{'scope':'SYNTHETIC_NUMERIC_AND_MEMORY_ABLATION','rows':rows,'new_llm_requests':0,'original_runtime_changed':False,
                           'label_metrics':None,'judgment':'Structural outputs only; no synthetic percentage called spacecraft accuracy; MODEL_ONLY is a hypothesis not a confirmed diagnosis.'})
    print(canonical({'runs':len(rows),'failed':[r for r in rows if r['run_status']=='FAILED']}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
