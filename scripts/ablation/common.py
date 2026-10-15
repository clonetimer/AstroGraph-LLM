"""Read-only, evaluator-side ablations. Never imported by the runtime."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from datetime import datetime, timezone


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def sha_obj(obj):return hashlib.sha256(canonical(obj).encode('utf-8')).hexdigest()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: f.write(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def event(root,kind,**fields):
    p=Path(root)/'ledger.jsonl';prev='0'*64
    if p.exists():
        lines=p.read_text(encoding='utf-8').splitlines()
        if lines:prev=json.loads(lines[-1])['event_sha256']
    item={'timestamp':datetime.now(timezone.utc).isoformat(),'kind':kind,'previous_sha256':prev,**fields}
    item['event_sha256']=sha_obj(item)
    with p.open('a',encoding='utf-8') as f:f.write(canonical(item)+'\n')


def pack_result(result):
    d=result['diagnosis'];f=result['fusion']['birrf_result']
    from evaluate_core_fusion_zip import prediction
    return {'prediction':prediction(d),'status':d['status'],'selected':d['selected_fault_id'],
        'knowledge_verified':d['knowledge_verified'],'requires_discriminating_evidence':d['requires_discriminating_evidence'],
        'support_ids':sorted({i for c in d['candidate_explanations'] for i in c['supporting_evidence_ids']}),
        'counter_ids':sorted({i for c in d['candidate_explanations'] for i in c['contradicting_evidence_ids']}),
        'graph_path_count':sum(len(c['graph_paths']) for c in d['candidate_explanations']),
        'top':f['top_fault_id'],
        'ranking':[{k:c[k] for k in ('fault_id','final_rank','fused_score','raw_reciprocal_contributions','reciprocal_contributions','dependency_adjusted')} for c in f['candidates']],
        'reason_codes':d['reason_codes'],'root_cause_confirmed':d['root_cause_confirmed']}


SCORE_ARMS={
 'forward_score_off':{'forward_weight':0.0},
 'backward_score_off':{'backward_weight':0.0},
 'both_kg_scores_off':{'forward_weight':0.0,'backward_weight':0.0},
 'data_score_off':{'data_weight':0.0},
 'lineage_cap_off':{'collapse_shared_correlation_groups':False},
 'kg_redundancy_cap_off':{'collapse_identical_kg_rankings':False,'collapse_highly_correlated_kg_rankings':False},
}


def intervene(data,bundle,kernel,*,arm):
    """Fixed-evidence component intervention, NOT an end-to-end LLM rerun.

    Score masks retain all evidence and nominations. Knowledge removal explicitly
    removes definitions/facts/associations on an evaluator copy, not the source
    evidence, model receipts, active graph, runtime policy or published model.
    """
    from astrograph.oag.contracts import OAGInput
    from astrograph.oag.kernel import validate_causal_evidence,apply_fusion_decision
    from astrograph.oag.summary import build_diagnostic_summary
    from astrograph.fusion import RuntimeFusionService
    from astrograph.knowledge import associations_from_query_result,build_forward_evidence_query_plan,build_backward_fault_query_plan
    from astrograph.fusion.decision import render_decision
    data=OAGInput.model_validate_json(data.model_dump_json())
    validate_causal_evidence(bundle)
    if bundle.evaluation_metadata: raise ValueError('TRUTH_IN_RUNTIME_BUNDLE')
    b=bundle.model_copy(deep=True)
    cfg=data.fusion_config.model_copy(deep=True,update=SCORE_ARMS.get(arm,{}))
    if arm not in {*SCORE_ARMS,'no_executable_knowledge','explicit_counter_rules_off'}:raise ValueError('UNREGISTERED_ABLATION')
    if arm in {'no_executable_knowledge','explicit_counter_rules_off'}:
        keep=[d for d in b.evidence_definitions if arm!='no_executable_knowledge' and d.role.value!='COUNTER']
        ids={d.evidence_definition_id for d in keep}
        b=b.model_copy(update={'evidence_definitions':keep,'evidence_facts':[f for f in b.evidence_facts if f.evidence_definition_id in ids]})
    definitions=sorted(d.evidence_definition_id for d in b.evidence_definitions)
    faults=sorted({d.target_fault_id for d in b.evidence_definitions})
    queries=[kernel.executor.execute(build_forward_evidence_query_plan(definitions),knowledge_version=kernel.knowledge_version),
             kernel.executor.execute(build_backward_fault_query_plan(faults),knowledge_version=kernel.knowledge_version)] if definitions and faults else []
    # Restrict BOTH query directions to the masked definition set.
    if queries:
        queries=[q.model_copy(update={'rows':[r for r in q.rows if (r.start_entity_id in definitions or r.target_entity_id in definitions)],
          'row_count':sum(r.start_entity_id in definitions or r.target_entity_id in definitions for r in q.rows)}) for q in queries]
    pkg=RuntimeFusionService().build(bundle=b,data_ranking=data.diagnostic_ranking,config=cfg,
        registered_fault_ids=data.registered_fault_ids,
        forward_associations=associations_from_query_result(queries[0]) if queries else [],
        backward_associations=associations_from_query_result(queries[1]) if queries else [],
        runtime_metadata={'scope':'EVALUATOR_INTERVENTION_ONLY','ablation':arm})
    summary=build_diagnostic_summary(b,pkg,queries,top_k=max(1,len(pkg.birrf_result.candidates)),graph_identity=kernel.identity)
    summary=apply_fusion_decision(data,b,pkg,summary)
    return {'diagnosis':summary['diagnostic_decision'],'fusion':pkg.model_dump(mode='json'),
            'summary':summary,'explanation':render_decision(summary['diagnostic_decision'])}
