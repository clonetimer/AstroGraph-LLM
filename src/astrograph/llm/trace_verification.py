"""Offline verification of saved JSON exchanges. NEVER calls a service."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def verify_task(task_dir: Path) -> dict:
    task_dir=Path(task_dir).resolve()
    rows=[]
    def check(artifact):
        name=artifact.get('name')
        if name is None:
            return {'status':'HASHES_ONLY_NO_BYTE_REPLAY'}
        path=task_dir/name
        if Path(name).name!=name or not path.resolve().is_relative_to(task_dir):
            raise ValueError('UNSAFE_TRACE_ARTIFACT')
        if path.stat().st_size>64_000_000:
            raise ValueError('TRACE_ARTIFACT_TOO_LARGE')
        raw=path.read_bytes()
        if len(raw)!=artifact['bytes'] or hashlib.sha256(raw).hexdigest()!=artifact['sha256']:
            raise ValueError('TRACE_ARTIFACT_HASH_MISMATCH:'+name)
        return {'status':'VERIFIED','name':name}
    for index in sorted(task_dir.glob('llm-call-*.index.json')):
        data=json.loads(index.read_text(encoding='utf-8'))
        if data.get('schema_version')!='astrograph-llm-wire-audit.v1':
            raise ValueError('UNKNOWN_TRACE_SCHEMA')
        for item in data['attempts']:
            row={'call_index':index.name,'attempt':item['attempt'],'status':item['status'],
                 'transport_observed':item['transport_observed']}
            for key in ('request','response','parsed_response'):
                if key in item:row[key]=check(item[key])
            name=item['request'].get('name')
            if name:
                body=json.loads((task_dir/name).read_text(encoding='utf-8'))
                envelope=json.loads(body['messages'][1]['content'])
                payload=envelope.get('payload',{})
                row['visible_citation_ids']=payload.get('allowed_citation_ids',[])
                row['schema_name']=(body.get('response_format',{}).get('json_schema',{}).get('name'))
            rows.append(row)
    # A killed process may leave a prepared request without a final attempt receipt.
    pending=[p.name for p in task_dir.glob('llm-call-*-attempt-*.prepared.json')
             if not (task_dir/p.name.replace('.prepared.json','.receipt.json')).exists()]
    return {'schema_version':'astrograph-llm-trace-verification.v1',
            'status':'INCOMPLETE' if pending else ('VERIFIED' if rows else 'NO_HTTP_ATTEMPTS'),
            'rows':rows,'incomplete_requests':sorted(pending), 'generation_requests_sent':0,
            'scope':'ARTIFACT_INTEGRITY_NOT_MODEL_QUALITY_OR_SOURCE_TRUTH'}
