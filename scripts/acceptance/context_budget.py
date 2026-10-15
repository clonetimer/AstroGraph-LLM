#!/usr/bin/env python3
"""Replay saved context receipts through the current projector. No generation HTTP.

This does NOT make stage 39 pass and does not edit previous acceptance records.
Input artifact hashes must match. Outputs are written only into a new directory.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))

from astrograph.agents.mission.context import ContextBuilder
from astrograph.agents.mission.models import InvestigationPlan, CheckChoice, AgentNarrative, ExplorationProposal, AgentSettings
from astrograph.agents.mission.planner import INSTRUCTION
from astrograph.agents.mission.modes import MODE_OBJECTIVES
from astrograph.agents.mission.storage import canonical, sha
from astrograph.llm import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend


def schema_for(payload):
    if "fixed_task" in payload: return InvestigationPlan
    if "eligible_checks" in payload: return CheckChoice
    if "required_candidate_fault_id" in payload: return AgentNarrative
    if "max_proposals" in payload: return ExplorationProposal
    raise ValueError("Unsupported saved context schema; no guessed task")


def replay(task_dir: Path, output: Path, *, mode: str, backend, context_length: int, output_reserve: int) -> dict:
    if output.exists(): raise FileExistsError("Choose a NEW output directory")
    records=json.loads((task_dir/'context-receipts.json').read_text(encoding='utf-8'))
    if not isinstance(records,list) or not records: raise ValueError("No context receipts")
    output.mkdir(parents=True)
    def save(digest,payload):
        path=output/f'context-{digest}.json'
        raw=canonical(payload)
        if path.exists() and path.read_bytes()!=raw:raise ValueError("Artifact collision")
        if not path.exists():path.write_bytes(raw)
        return {'name':path.name,'sha256':digest}
    rows=[]
    for number,previous in enumerate(records,1):
        item={'index':number,'previous_status':previous.get('status')}
        try:
            artifact=previous['full_payload_artifact'];name=artifact['name']
            path=task_dir/name
            if Path(name).name!=name or not path.resolve().is_relative_to(task_dir.resolve()):
                raise ValueError("Invalid context artifact path")
            if path.stat().st_size>64_000_000:raise ValueError("Artifact too large")
            payload=json.loads(path.read_text(encoding='utf-8'))
            digest=sha(payload)
            if digest!=artifact['sha256'] or digest!=previous['full_payload_sha256']:
                raise ValueError("Saved context hash mismatch")
            model=schema_for(payload)
            def envelope(value):
                return {'instruction':INSTRUCTION,'work_mode':mode,'mode_objective':MODE_OBJECTIVES[mode],**value}
            builder=ContextBuilder(max_tokens=context_length,output_reserve=output_reserve,artifact_writer=save)
            try:
                view=builder.build(model,payload,budget_measure=lambda value:backend.measure_structured_context(output_model=model,payload=envelope(value)))
            finally:
                item['context_receipts']=builder.records
            view_path=output/f'{number:02}-{model.__name__}.view.json'
            view_path.write_bytes(canonical(view))
            item.update(status='READY_FOR_LIVE_RETEST',schema=model.__name__,view=view_path.name,
                        original_bytes=len(canonical(payload)),projected_bytes=len(canonical(view)),
                        view_sha256=sha(view))
        except Exception as exc:
            item.update(status='BLOCKED',error_type=type(exc).__name__,reason=str(exc)[:800])
        rows.append(item)
    result={'schema_version':'astrograph-saved-context-preflight.v1','created_at':datetime.now(timezone.utc).isoformat(),
            'status':'READY_FOR_LIVE_RETEST' if all(r['status']=='READY_FOR_LIVE_RETEST' for r in rows) else 'BLOCKED',
            'scope':'CAPTURED_CONTEXT_CLIENT_PREFLIGHT_NOT_LIVE_LLM_NOT_DIAGNOSIS_BENCHMARK',
            'work_mode':mode,'context_length':context_length,'output_reserve':output_reserve,
            'generation_requests_sent':0,'previous_workspace_modified':False,'rows':rows}
    (output/'report.json').write_bytes(canonical(result))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-dir',type=Path,required=True,help='Existing llm-thinking or llm-creative task directory')
    p.add_argument('--work-mode',choices=['normal','thinking','creative'],required=True)
    p.add_argument('--output',type=Path,required=True,help='NEW output directory')
    p.add_argument('--settings',type=Path,help='Host settings; uses its local tokenizer if enabled')
    p.add_argument('--context-length',type=int,default=16384,help='Used only without --settings')
    p.add_argument('--output-reserve',type=int,default=4096,help='Used only without --settings')
    args=p.parse_args(argv)
    try:
        if args.settings:
            settings=AgentSettings.model_validate_json(args.settings.read_text(encoding='utf-8'))
            if not settings.llm_config:raise ValueError("Host has no LLM config")
            cfg=OpenAICompatibleConfig.from_json(settings.llm_config)
            if not cfg.tokenizer_local_files_only or cfg.tokenizer_trust_remote_code:
                raise ValueError("Offline replay requires local-only tokenizer with remote code disabled")
        else:
            cfg=OpenAICompatibleConfig(model='offline-context-audit',context_length=args.context_length,max_tokens=args.output_reserve,
                tokenizer_mode='disabled',repair_attempts=0,response_format_strategy='json_schema')
        backend=OpenAICompatibleStructuredBackend(cfg)
        result=replay(args.task_dir.resolve(),args.output.resolve(),mode=args.work_mode,backend=backend,
                      context_length=cfg.context_length or args.context_length,output_reserve=cfg.max_tokens)
        print(json.dumps({'status':result['status'],'receipt':str(args.output/'report.json'),
                          'generation_requests_sent':0,'scope':result['scope']},ensure_ascii=False,indent=2))
        return 0 if result['status']=='READY_FOR_LIVE_RETEST' else 2
    except Exception as exc:
        print(json.dumps({'status':'BLOCKED','error_type':type(exc).__name__,
                          'reason':str(exc)[:800]},ensure_ascii=False))
        return 2


if __name__=='__main__':raise SystemExit(main())
