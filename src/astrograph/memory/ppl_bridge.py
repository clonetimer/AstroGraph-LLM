"""Bounded Python/Node adapter; PPL is a policy/ledger provider, never a diagnosis voter."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4
from astrograph.integrations.process_control import run_managed_subprocess
from astrograph.orchestration.models import WorkflowContext, PermissionScope


class PPLBridge:
    def __init__(self, project_root: str | Path, *, profile_sha256: str, timeout_s: float = 5.0,
                 approved_reviewers: frozenset[str] = frozenset()):
        self.root=Path(project_root).resolve()
        self.profile_path=self.root/'configs/memory/diagnostic.profile.json'
        self.script=self.root/'integrations/ppl/bridge.mjs'
        self.profile_sha256=profile_sha256
        self.timeout_s=timeout_s
        self.approved_reviewers=approved_reviewers
        if timeout_s <= 0:raise ValueError('positive PPL timeout is required')
        if not shutil.which('node'):raise RuntimeError('PPL_NODE_UNAVAILABLE')
        self.profile=self._read_profile()

    def _read_profile(self):
        data=self.profile_path.read_bytes()
        if len(data)>2_000_000 or hashlib.sha256(data).hexdigest()!=self.profile_sha256:
            raise ValueError('PPL_PROFILE_HASH_MISMATCH')
        profile=json.loads(data)
        if profile.get('kind')!='research' or profile.get('transactionPolicy',{}).get('commitEndReasons')!=['completed']:
            raise ValueError('PPL_PROFILE_POLICY_REJECTED')
        return profile

    def resolve(self, *, event: dict[str, Any], context: WorkflowContext, state: dict | None = None,
                end_reason: str = 'completed') -> dict[str, Any]:
        self._read_profile()  # fail on mid-task profile replacement
        allowed={'QUESTION_DEFINED','CLAIM_PROPOSED','EVIDENCE_RECORDED','CONCLUSION_REQUESTED','VALIDATION_RESULT'}
        if event.get('type') not in allowed:raise ValueError('PPL_EVENT_NOT_ALLOWED')
        if event.get('type')=='VALIDATION_RESULT':
            if PermissionScope.REVIEW_MEMORY not in context.permissions or context.actor not in self.approved_reviewers:
                raise PermissionError('independent reviewer event required')
        request_id=str(uuid4())
        payload={'request_id':request_id,'profile':self.profile,'state':state,'event':event,
                 'context':{'actor':context.actor},'end_reason':end_reason}
        raw=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8')
        if len(raw)>2_000_000:raise ValueError('PPL_INPUT_TOO_LARGE')
        # Keep the wire format binary and decode explicitly.  `text=True`
        # inherits the Windows ANSI code page and corrupts non-ASCII JSON.
        result=run_managed_subprocess([shutil.which('node'),str(self.script)],input=raw,
                                     capture_output=True,text=False,timeout=self.timeout_s,
                                     cwd=self.script.parent,check=False)
        if result.returncode!=0:raise RuntimeError('PPL_BINDING_FAILED')
        stdout=result.stdout or b''
        if len(stdout)>2_000_000:raise ValueError('PPL_OUTPUT_TOO_LARGE')
        value=json.loads(stdout.decode('utf-8'))
        if value.get('request_id')!=request_id or value.get('schema_version')!='astrograph-ppl-binding-result.v1':
            raise ValueError('PPL_RESPONSE_IDENTITY_MISMATCH')
        if not isinstance(value.get('state'),dict) or type(value.get('committed')) is not bool or value.get('diagnostic_vote_allowed') is not False:
            raise ValueError('PPL_RESPONSE_CONTRACT_INVALID')
        if end_reason!='completed' and value['committed']:
            raise ValueError('PPL_TRUNCATED_TRANSACTION_COMMITTED')
        return value
