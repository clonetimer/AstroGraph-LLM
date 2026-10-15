#!/usr/bin/env python3
"""Read-only rebind/audit of saved LLM planning requests; never generate or execute.

The old response is NEVER edited into a compliant plan. It is checked against
both its original schema and today's narrowed per-call schema. A ready report
means the client contract is aligned, not that a real model has been re-tested.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from jsonschema import Draft202012Validator
from astrograph.agents.mission.models import InvestigationPlan, CheckChoice
from astrograph.agents.mission.storage import canonical, sha
from astrograph.llm.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleStructuredBackend
from astrograph.llm.task_prompts import optional_tool_descriptions, runtime_tool_contract, task_output_schema
from astrograph.llm.trace_verification import verify_task

MODELS={x.__name__:x for x in (InvestigationPlan,CheckChoice)}


def read_json(path: Path, *, root: Path, max_bytes: int=8_000_000):
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError('TRACE_PATH_ESCAPE_OR_SYMLINK')
    if path.stat().st_size>max_bytes:raise ValueError('TRACE_TOO_LARGE')
    return json.loads(path.read_bytes())


def audit_task(task_dir: Path, *, context_length: int=16384) -> dict:
    task_dir=task_dir.resolve()
    integrity=verify_task(task_dir)
    if integrity['status']!='VERIFIED':raise ValueError('TRACE_INTEGRITY_NOT_VERIFIED')
    rows=[]
    for path in sorted(task_dir.glob('llm-call-*-attempt-*.request.json')):
        wire=read_json(path,root=task_dir)
        user=json.loads(wire['messages'][1]['content'])
        model=MODELS.get(user.get('task'))
        if model is None:continue
        old_schema=user['output_json_schema']
        old_payload=user['payload']; current=copy.deepcopy(old_payload)
        if model is InvestigationPlan:
            current['available_optional_tool_descriptions']=optional_tool_descriptions(current.get('available_optional_tools',[]))
        contract=runtime_tool_contract(model,current)
        new_schema=task_output_schema(model,current)
        Draft202012Validator.check_schema(new_schema)
        b=OpenAICompatibleStructuredBackend(OpenAICompatibleConfig(model=wire['model'],
            tokenizer_mode='disabled',repair_attempts=0,response_format_strategy='json_schema',
            max_tokens=wire['max_tokens'],context_length=context_length))
        new_wire=b._body(output_model=model,payload=current,strategy='json_schema',max_tokens=wire['max_tokens'])
        prompt_schema=json.loads(new_wire['messages'][1]['content'])['output_json_schema']
        if prompt_schema!=new_wire['response_format']['json_schema']['schema']:
            raise ValueError('NEW_PROMPT_WIRE_SCHEMA_MISMATCH')
        measured=b.measure_structured_context(output_model=model,payload=current)
        response_path=path.with_name(path.name.replace('.request.json','.response.bin'))
        response=read_json(response_path,root=task_dir)
        content=response['choices'][0]['message']['content']
        observed=b._json_from_text(content)
        if model is InvestigationPlan:
            old_ids=old_schema.get('$defs',{}).get('PreparatoryAction',{}).get('properties',{}).get('tool_id',{}).get('enum',[])
            old_selected=[x['tool_id'] for x in observed.get('actions',[])]
        else:
            old_ids=old_schema['properties']['tool_id'].get('enum')
            old_selected=[observed.get('tool_id')]
        rows.append({'request_file':path.name,'schema_name':model.__name__,
            'old_request_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'old_response_sha256':hashlib.sha256(response_path.read_bytes()).hexdigest(),
            'old_wire_tool_enum':old_ids,'old_response_selected_tool_ids':old_selected,
            'original_prompt_schema_matches_response_format':old_schema==wire.get('response_format',{}).get('json_schema',{}).get('schema',old_schema),
            'old_response_valid_under_original_schema':Draft202012Validator(old_schema).is_valid(observed),
            'old_response_valid_under_current_bound_schema':Draft202012Validator(new_schema).is_valid(observed),
            'new_tool_contract':contract,'new_tool_contract_sha256':sha(contract),
            'new_prompt_and_wire_schema_match':True,'new_schema_sha256':sha(new_schema),
            'current_budget_preflight':measured,
            'old_response_rewritten':False,
            'status':'CONTRACT_BOUND_READY_FOR_LIVE_RETEST' if measured['fits'] else 'CONTEXT_BUDGET_REJECTED'})
    if not rows:raise ValueError('NO_RECORDED_PLANNING_REQUESTS')
    return {'schema_version':'astrograph-tool-contract-offline-review.v1',
            'status':'READY_FOR_LIVE_RETEST' if all(x['status']=='CONTRACT_BOUND_READY_FOR_LIVE_RETEST' for x in rows) else 'PARTIAL',
            'scope':'CLIENT_SCHEMA_AUTHORITY_ALIGNMENT_NOT_MODEL_QUALITY',
            'trace_integrity':integrity['status'],'requests_checked':len(rows),'requests':rows,
            'generation_requests_sent':0,'tools_executed':0,'old_response_modified':False,
            'previous_workspace_modified':False,'live_model_success_established':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--context-length',type=int,default=16384)
    a=p.parse_args()
    if a.output.resolve().is_relative_to(a.task_dir.resolve()):raise ValueError('Write the report OUTSIDE the old task directory')
    if a.output.exists():raise FileExistsError('Use a new output file; do not overwrite old receipts')
    try:report=audit_task(a.task_dir,context_length=a.context_length)
    except Exception as exc:report={'status':'FAILED','reason':str(exc),'error_type':type(exc).__name__,'generation_requests_sent':0}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_bytes(canonical(report))
    print(json.dumps({'status':report['status'],'output':str(a.output),'generation_requests_sent':0},ensure_ascii=False))
    return 0 if report['status']=='READY_FOR_LIVE_RETEST' else 2


if __name__=='__main__':raise SystemExit(main())
