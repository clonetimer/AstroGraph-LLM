"""P1-P4 acceptance stages.  Each stage is isolated and fail-closed."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from scripts.acceptance.common import Context, check, digest, save, ROOT
from astrograph.agents.mission.models import AgentRequest
from astrograph.agents.mission.service import MissionAgent
from astrograph.agents.mission.model_capabilities import ModelCapabilityQuery, ModelCapabilityRegistry
from astrograph.agents.mission.storage import file_sha, sha
from astrograph.lifecycle import (
    DocumentProposal, ExpertFeedback, KnowledgeFeedbackStore, KnowledgeProposal, LifecyclePolicy,
)


def generic_series(c: Context) -> dict:
    settings=c.settings()
    # P1 acceptance explicitly opts into the host-owned capability catalogue.
    settings=settings.model_copy(update={"model_capability_registry":"configs/agent/model_capabilities.default.json"})
    rows=100
    known=c.path('known.csv'); unknown=c.path('unknown.csv')
    pd.DataFrame({'time_s':range(rows),'sensor.eps.battery_voltage_v':[0.0]*50+[100.0]*50}).to_csv(known,index=False)
    pd.DataFrame({'time_s':range(rows),'mystery_signal':[0.0]*50+[100.0]*50}).to_csv(unknown,index=False)
    def manifest(data:Path, channel:str, unit:str, name:str):
        p=c.path(name)
        save(p,{"schema_version":"astrograph-generic-series-manifest.v1","data_path":str(data),
                "entity_id":name,"timestamp_column":"time_s","channels":[{"name":channel,"unit":unit}],
                "baseline_fraction":0.3})
        return p
    m1=manifest(known,'sensor.eps.battery_voltage_v','V','known-manifest.json')
    m2=manifest(unknown,'mystery_signal','1','unknown-manifest.json')
    agent=MissionAgent(settings)
    multi=c.path('multi.csv')
    pd.DataFrame({'time_s':range(rows),'sensor.eps.battery_voltage_v':[0.0]*50+[100.0]*50,
                  'mystery_signal':[0.0]*50+[-100.0]*50}).to_csv(multi,index=False)
    mm=c.path('multi-manifest.json');save(mm,{"schema_version":"astrograph-generic-series-manifest.v1","data_path":str(multi),
        "entity_id":"multi","timestamp_column":"time_s","channels":[{"name":"sensor.eps.battery_voltage_v","unit":"V"},{"name":"mystery_signal","unit":"1"}],"baseline_fraction":0.3})
    r1=agent.run_inline(AgentRequest(task_id='p1-known',artifacts={'series_manifest':str(m1)}))
    r2=agent.run_inline(AgentRequest(task_id='p1-unknown',artifacts={'series_manifest':str(m2)}))
    r3=agent.run_inline(AgentRequest(task_id='p1-multi',artifacts={'series_manifest':str(mm)}))
    c.write('known-result.json',r1.model_dump(mode='json')); c.write('unknown-result.json',r2.model_dump(mode='json'));c.write('multi-result.json',r3.model_dump(mode='json'))
    d1=r1.output['diagnosis_result']['diagnosis']; d2=r2.output['diagnosis_result']['diagnosis'];d3=r3.output['diagnosis_result']['diagnosis']
    check(r1.status=='SUCCEEDED' and d1['detected'] and d1['selected_fault_id']=='EPS_BATTERY_VOLTAGE_FAULT','mapped generic series did not produce the expected provisional cause')
    check(d1['status']=='CANDIDATE_SET' and d1['knowledge_verified'] is False,'provisional generic knowledge was overclaimed')
    check(r2.status=='SUCCEEDED' and d2['detected'] and d2['selected_fault_id'] is None and d2['status']=='INSUFFICIENT_EVIDENCE','unmapped series invented a physical cause')
    check(r3.status=='SUCCEEDED' and d3['detected'] and d3['selected_fault_id'] is None and d3['status']=='INSUFFICIENT_EVIDENCE','partially mapped multichannel input was overclaimed')
    multi_layered=r3.output['diagnosis_result']['generic_series']
    check(any(x.get('fault_id')=='EPS_BATTERY_VOLTAGE_FAULT' for x in multi_layered.get('fault_candidates',[])),'mapped multichannel cause candidate was lost')
    return {'mapped_case':{'status':d1['status'],'fault_id':d1['selected_fault_id'],'knowledge_verified':d1['knowledge_verified']},
            'unmapped_case':{'status':d2['status'],'fault_id':d2['selected_fault_id']},
            'mixed_multichannel_case':{'status':d3['status'],'fault_id':d3['selected_fault_id'],'candidate_retained':True},
            'labels_used':False,'production_model_modified':False,'active_kg_modified':False}


def capability_registry(c: Context) -> dict:
    path=ROOT/'configs/agent/model_capabilities.default.json'
    reg=ModelCapabilityRegistry.from_json(path,project_root=ROOT)
    queries={
        'generic_detection':ModelCapabilityQuery(domain='generic',channel_names=('mystery_signal',),tasks=('detection',)),
        'generic_classification':ModelCapabilityQuery(domain='generic',channel_names=('mystery_signal',),tasks=('classification',)),
        'adcs_bound_classification':ModelCapabilityQuery(domain='adcs',tasks=('classification',),bound_artifact_contract=True),
        'phm_replay_classification':ModelCapabilityQuery(domain='phm',tasks=('classification',),online_required=False,allow_retrospective=True,bound_artifact_contract=True),
    }
    results={k:reg.query(v) for k,v in queries.items()};c.write('capabilities.json',results)
    check(results['generic_detection']['compatible'],'generic detection must be discoverable')
    check(not results['generic_classification']['compatible'],'arbitrary generic classification must not be invented')
    check(any(x['runtime_kind']=='OAG_NATIVE' for x in results['adcs_bound_classification']['compatible']),'bound ADCS model missing')
    check(any(x['runtime_kind']=='PHM_RETROSPECTIVE_BUNDLE' for x in results['phm_replay_classification']['compatible']),'PHM replay bundle missing')
    check(all(not value['execution_authorized'] for value in results.values()),'capability discovery granted execution')
    return {'catalog_sha256':digest(path),'queries':{k:len(v['compatible']) for k,v in results.items()},'execution_authorized':False}


def phm_replay(c: Context) -> dict:
    c.long_task()
    value=c.config.get('phm_dataset_root')
    if not value: raise RuntimeError('Initialize a new workspace with --phm-dataset-root PATH for P3 replay')
    dataset=Path(value)
    if not (dataset/'sample_index.csv').is_file(): raise RuntimeError('PHM dataset root lacks sample_index.csv')
    replay=c.path('replay-manifest.json');save(replay,{"schema_version":"astrograph-replay-manifest.v1","dataset_kind":"phm_v6_retrospective","model_strategy":"promoted_extra_trees","skip_llm":True})
    settings=c.settings().model_copy(update={'domain':'phm','evidence_profile':'legacy','max_optional_actions':0,
        'model_capability_registry':'configs/agent/model_capabilities.default.json',
        'subprocess_timeout_s':min(c.config['long_timeout_s'],3600),'task_timeout_s':min(c.config['long_timeout_s']+60,7200)})
    request=AgentRequest(task_id='p3-phm-replay',task='replay',goal='Offline retrospective PHM replay acceptance.',
        artifacts={'replay_manifest':str(replay),'dataset_root':str(dataset)})
    result=MissionAgent(settings).run_inline(request);c.write('result.json',result.model_dump(mode='json'))
    output=result.output['task_output']; metrics=output['metrics']
    check(result.status=='SUCCEEDED','PHM Mission replay failed')
    check(all(metrics[k] and metrics[k].get('macro_f1') is not None for k in ('detection','localization','classification','end_to_end')),'PHM Mission replay omitted metrics')
    check(output['offline_evaluation_truth_used'] is True and output['online_diagnosis_claim_allowed'] is False,'offline PHM replay claim boundary was lost')
    return {'model_strategy':output['model_strategy'],'metrics':{k:metrics[k]['macro_f1'] for k in metrics},
            'offline_evaluation_truth_used':True,'online_diagnosis_claim_allowed':False,
            'dataset_identity_sha256':digest(dataset/'sample_index.csv')}


def lifecycle(c: Context) -> dict:
    root=c.out.resolve(); doc=c.path('manual.md');doc.write_text('Battery undervoltage requires evidence and expert-reviewed mechanism context.\n',encoding='utf-8')
    policy=LifecyclePolicy(approved_reviewers=('acceptance-expert',),permitted_document_roots=(str(root),))
    policy_path=c.path('policy.json');policy_path.write_text(policy.model_dump_json(),encoding='utf-8')
    store=KnowledgeFeedbackStore(c.path('lifecycle.sqlite'),policy)
    proposal=DocumentProposal(document_id='DOC-P4',source_path=str(doc),source_sha256=file_sha(doc),title='P4 fixture',media_type='md',parser='UTF8_NATIVE_TEXT',proposer_actor='acceptance',proposed_at=datetime.now(timezone.utc))
    staged=store.stage_document(proposal,event_id='p4-stage-doc');check(not staged['library_visible'] and not store.approved_documents(),'unreviewed document became visible')
    approved=store.review_document(document_id='DOC-P4',decision='APPROVE',reviewer='acceptance-expert',reason='acceptance fixture',event_id='p4-review-doc')
    check(approved['library_visible'] and len(store.approved_documents())==1,'approved document not visible')
    candidate={'status':'REVIEW_REQUIRED','active_kg_modified':False,'edges':[{'subject':'battery_voltage','predicate':'may_indicate','object':'eps_fault'}]}
    kp=KnowledgeProposal(proposal_id='KPROP-P4',task_id='P4',candidate_sha256=sha(candidate),candidate_payload=candidate,source_bindings=(),proposer_actor='acceptance',proposed_at=datetime.now(timezone.utc))
    kstaged=store.stage_knowledge_proposal(kp,event_id='p4-stage-kg');check(kstaged['active_kg_write_allowed'] is False,'candidate gained KG write permission')
    kreview=store.review_knowledge_proposal(proposal_id='KPROP-P4',decision='APPROVE_FOR_OFFLINE_EVALUATION',reviewer='acceptance-expert',reason='run graph regression',event_id='p4-review-kg')
    check(kreview['active_kg_modified'] is False and kreview['status']=='APPROVED_FOR_OFFLINE_EVALUATION','expert review mutated active KG')
    feedback=ExpertFeedback(feedback_id='FB-P4',task_id='T-P4',result_sha256='a'*64,feedback_type='DIAGNOSIS_CORRECTION',reviewer='acceptance-expert',reviewed_at=datetime.now(timezone.utc),truth_source_ref='fixture://independent-expert-label',verified_fault_id='FAULT-B',target_layers=('planning','memory','tool','knowledge','model','execution'),diagnosis_error_summary='fixture diagnosis A was incorrect',recommended_change='evaluate a B-discriminating check before release',evidence_refs=('fixture-evidence',))
    freceipt=store.submit_feedback(feedback,event_id='p4-feedback');check(len(freceipt['candidate_ids'])==6 and freceipt['production_modified'] is False,'feedback mutated production')
    first=freceipt['candidate_ids'][0]; reviewed=store.review_candidate(candidate_id=first,decision='APPROVE_FOR_OFFLINE_EVALUATION',reviewer='acceptance-expert',reason='A/B only',event_id='p4-review-improvement')
    check(reviewed['confirmed_memory_modified'] is False and reviewed['active_model_modified'] is False,'offline candidate changed production state')
    # Verify the approved library can be read by Mission/RAG without becoming a diagnostic vote.
    settings=c.settings().model_copy(update={'lifecycle_db':str(c.path('lifecycle.sqlite')),'lifecycle_policy':str(policy_path),
        'use_approved_document_library':True,'max_optional_actions':1})
    research=MissionAgent(settings).run_inline(AgentRequest(task_id='p4-research',task='research',goal='battery undervoltage evidence'))
    c.write('research-result.json',research.model_dump(mode='json'))
    check(research.status=='SUCCEEDED' and research.output['task_output']['sources'],'approved library was not retrievable')
    check(all(x.get('diagnostic_vote_allowed') is False for x in research.output['task_output']['sources']),'library source gained diagnostic vote')
    return {'document':approved,'knowledge_proposal':kreview,'feedback_candidate_count':len(freceipt['candidate_ids']),
            'research_sources':len(research.output['task_output']['sources']),'active_kg_modified':False,'active_model_modified':False,'confirmed_memory_modified':False}


HANDLERS={'66-generic-series':generic_series,'67-model-capabilities':capability_registry,'68-phm-mission-replay':phm_replay,'69-knowledge-feedback-lifecycle':lifecycle}
