#!/usr/bin/env python3
"""Offline source-contract replay of a captured dev8 task; NEVER sends requests.

Re-derives short host citation anchors from the hash-verified saved core. This is
an explicit dev9 projection migration, not a claim to reconstruct old wire bytes.
"""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from astrograph.agents.mission.context import ContextBuilder
from astrograph.agents.mission.sources import diagnostic_sources,visible_ids
from astrograph.agents.mission.models import AgentNarrative,ExplorationProposal,InvestigationPlan,CheckChoice
from astrograph.agents.mission.storage import canonical,sha,file_sha
from astrograph.agents.mission.planner import INSTRUCTION
from astrograph.agents.mission.modes import MODE_OBJECTIVES
from astrograph.llm import OpenAICompatibleConfig,OpenAICompatibleStructuredBackend


def replay(task_dir,output,*,mode,backend,context_length,output_reserve):
    task_dir=Path(task_dir).resolve();output=Path(output)
    if output.exists():raise FileExistsError('Choose a new output directory')
    records=json.loads((task_dir/'context-receipts.json').read_text(encoding='utf-8'))
    core_path=task_dir/'core-result.json';core=json.loads(core_path.read_text(encoding='utf-8'))
    expected=json.loads((task_dir/'core-result-receipt.json').read_text(encoding='utf-8'))['sha256']
    if file_sha(core_path)!=expected:raise ValueError('CORE_HASH_MISMATCH')
    output.mkdir(parents=True)
    def writer(digest,value):
        path=output/('context-'+digest+'.json')
        if not path.exists():path.write_bytes(canonical(value))
        return {'name':path.name,'sha256':digest}
    rows=[]
    models={c.__name__:c for c in (AgentNarrative,ExplorationProposal,InvestigationPlan,CheckChoice)}
    for n,receipt in enumerate(records,1):
        row={'index':n,'schema_name':receipt['schema_name'],'previous_status':receipt['status']}
        try:
            art=receipt['full_payload_artifact'];name=art['name']
            if Path(name).name!=name or not (task_dir/name).resolve().is_relative_to(task_dir):raise ValueError('UNSAFE_CONTEXT_PATH')
            payload=json.loads((task_dir/name).read_text(encoding='utf-8'))
            if sha(payload)!=art['sha256']:raise ValueError('CONTEXT_HASH_MISMATCH')
            old=copy.deepcopy(payload);model=models[receipt['schema_name']]
            if model in (AgentNarrative,ExplorationProposal):
                # Only replace the old host-generated oversized anchor, not RAG sources.
                marker='DIAGNOSIS-' if model is AgentNarrative else 'TASK-'
                source_list=payload.get('sources_untrusted',[])
                external=[s for s in source_list if not str(s.get('citation_id','')).startswith(marker)]
                payload['sources_untrusted']=diagnostic_sources(core,exploration=model is ExplorationProposal)+external
            builder=ContextBuilder(max_tokens=context_length,output_reserve=output_reserve,artifact_writer=writer)
            def envelope(value):return {'instruction':INSTRUCTION,'work_mode':mode,'mode_objective':MODE_OBJECTIVES[mode],**value}
            def measure(value):return backend.measure_structured_context(output_model=model,payload=envelope(value))
            view=builder.build(model,payload,budget_measure=measure)
            wire=backend._body(output_model=model,payload=envelope(view),strategy='json_schema',max_tokens=output_reserve)
            (output/f'{n:02d}-{model.__name__}.request-preview.json').write_bytes(canonical(wire))
            result=builder.records[-1]
            (output/f'{n:02d}-{model.__name__}.receipt.json').write_bytes(canonical(result))
            row.update(status='READY_FOR_LIVE_RETEST',original_bytes=len(canonical(old)),
                upgraded_input_bytes=len(canonical(payload)),projected_bytes=len(canonical(view)),
                projected_sha256=sha(view),visible_citation_ids=visible_ids(view),
                required_sources_visible=all(s['citation_id'] in visible_ids(view) for s in payload.get('sources_untrusted',[]) if s.get('required_for_grounding')),
                included_count=len(result['included_passages']),omitted_count=len(result['omitted_sources']),
                budget_measurement=result['budget_measurement'])
        except Exception as exc:row.update(status='BLOCKED',reason=str(exc),error_type=type(exc).__name__)
        rows.append(row)
    report={'schema_version':'astrograph-dev9-grounding-offline-replay.v1','rows':rows,
        'status':'READY_FOR_LIVE_RETEST' if all(r['status']=='READY_FOR_LIVE_RETEST' for r in rows) else 'BLOCKED',
        'generation_requests_sent':0,'previous_workspace_modified':False,
        'saved_core_sha256':expected,'work_mode':mode,
        'scope':'EXPLICIT_HOST_SOURCE_MIGRATION_NOT_OLD_WIRE_RECONSTRUCTION_NOT_MODEL_QUALITY'}
    (output/'report.json').write_bytes(canonical(report))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--work-mode',choices=['thinking','creative'],required=True)
    p.add_argument('--context-length',type=int,default=16384);p.add_argument('--output-reserve',type=int,default=4096)
    a=p.parse_args()
    backend=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(model='OFFLINE_NOT_GENERATED',
        context_length=a.context_length,max_tokens=a.output_reserve,tokenizer_mode='disabled',
        response_format_strategy='json_schema',repair_attempts=0))
    r=replay(a.task_dir,a.output,mode=a.work_mode,backend=backend,context_length=a.context_length,output_reserve=a.output_reserve)
    print(json.dumps({'status':r['status'],'generation_requests_sent':0,'output':str(a.output)}))
    return 0 if r['status']=='READY_FOR_LIVE_RETEST' else 2
if __name__=='__main__':raise SystemExit(main())
