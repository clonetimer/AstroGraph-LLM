"""Host-neutral execution policy and durable receipts around the native OAG runtime.

FreeSeek-inspired run snapshots and PPL-inspired canonical-context fidelity are
implemented locally; this module does not claim to embed either reference engine.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from astrograph.runtime import RuntimeRequest
from .contracts import OAGInput, content_hash
from .runtime import OAG_PIPELINE_ID, build_csv_oag_runtime


class AgentProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    role_id: str
    mission: str
    allowed_stages: tuple[str, ...]
    allowed_model_agents: tuple[str, ...] = ()
    may_modify_active_graph: bool = False
    may_execute_control: bool = False


PROFILES = (
    AgentProfile(role_id="evidence", mission="Bind telemetry and model output to graph evidence.",
                 allowed_stages=("validate", "graph", "fuse", "summarize")),
    AgentProfile(role_id="knowledge", mission="Retrieve sources and propose cited ontology candidates.",
                 allowed_stages=("literature", "ontology"), allowed_model_agents=("oag-ontology-extraction",)),
    AgentProfile(role_id="analyst", mission="Review alternatives and report evidence-constrained conclusions.",
                 allowed_stages=("generate",), allowed_model_agents=("oag-diagnosis-explanation",)),
)


class HarnessPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    max_model_calls: int = Field(default=2, ge=0, le=10)
    max_prompt_bytes: int = Field(default=131072, ge=1024, le=1048576)
    profiles: tuple[AgentProfile, ...] = PROFILES

    def role_for(self, stage_id: str) -> AgentProfile:
        name = stage_id.split('.')[1]
        roles = [r for r in self.profiles if name in r.allowed_stages]
        if len(roles) != 1:
            raise ValueError("stage must have exactly one role owner")
        role = roles[0]
        if role.may_modify_active_graph or role.may_execute_control:
            raise ValueError("diagnostic harness roles cannot acquire mutation/control authority")
        return role


class Journal:
    """Hash-chained events; the authoritative result is bound as a separate blob."""
    def __init__(self, root: Path):
        self.root = root
        self.path = root / 'events.jsonl'
        self.seq = 0
        self.previous = '0' * 64

    def append(self, kind: str, payload: dict) -> None:
        body = {'seq': self.seq, 'previous_sha256': self.previous, 'kind': kind, 'payload': payload}
        row = {**body, 'event_sha256': content_hash(body)}
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        self.seq += 1
        self.previous = row['event_sha256']


def verify_run(root: Path) -> dict:
    """Verify event order, terminal-result binding and content integrity."""
    previous = '0' * 64
    events = []
    for index, line in enumerate((root / 'events.jsonl').read_text(encoding='utf-8').splitlines()):
        row = json.loads(line)
        claimed = row.pop('event_sha256')
        if row['seq'] != index or row['previous_sha256'] != previous or claimed != content_hash(row):
            raise ValueError('harness journal integrity failure')
        previous = claimed
        events.append(row)
    if not events or events[0]['kind'] != 'RUN_STARTED' or events[-1]['kind'] != 'RUN_FINISHED':
        raise ValueError('harness run is incomplete; start a new run after investigating interruption')
    result = json.loads((root / 'result.json').read_text(encoding='utf-8'))
    if events[-1]['payload']['result_sha256'] != content_hash(result):
        raise ValueError('harness result integrity failure')
    return {'status': 'PASS', 'event_count': len(events), 'root_sha256': previous,
            'input_sha256': events[0]['payload']['input_sha256'],
            'snapshot_sha256': events[0]['payload']['snapshot_sha256']}


class _Observer:
    def __init__(self, journal: Journal, policy: HarnessPolicy):
        self.journal, self.policy = journal, policy
        self.canonical = None
        self.model_calls = 0
        self.active_role = None

    def before(self, stage: str, payload: dict) -> None:
        role = self.policy.role_for(stage)
        self.active_role = role
        if self.canonical is not None and content_hash(payload.get('summary')) != self.canonical:
            raise ValueError('handoff changed canonical diagnostic summary')
        self.journal.append('STAGE_STARTED', {'stage': stage, 'role': role.role_id,
                                            'input_sha256': content_hash(payload)})

    def after(self, stage: str, payload: dict) -> None:
        if stage == 'oag.summarize.v1':
            self.canonical = content_hash(payload['summary'])
        if self.canonical is not None and content_hash(payload.get('summary')) != self.canonical:
            raise ValueError('stage changed canonical diagnostic summary')
        self.journal.append('STAGE_FINISHED', {'stage': stage, 'output_sha256': content_hash(payload),
                                             'canonical_summary_sha256': self.canonical})

    def failed(self, stage: str, error_type: str) -> None:
        self.journal.append('STAGE_FAILED', {'stage': stage, 'error_type': error_type})


class _BoundedBackend:
    def __init__(self, backend: Any, observer: _Observer):
        self.backend, self.observer = backend, observer

    @property
    def last_call_metadata(self):
        return getattr(self.backend, 'last_call_metadata', {})

    def generate_structured(self, *, agent_id, output_model, payload):
        o = self.observer
        if o.active_role is None or agent_id not in o.active_role.allowed_model_agents:
            raise ValueError('model call is outside the role capability snapshot')
        projected = copy.deepcopy(payload)
        projected['role_contract'] = o.active_role.model_dump(mode='json')
        size = len(json.dumps(projected, ensure_ascii=False, allow_nan=False).encode())
        if size > o.policy.max_prompt_bytes:
            raise ValueError('prompt budget exceeded; canonical evidence cannot be silently truncated')
        if o.model_calls >= o.policy.max_model_calls:
            raise ValueError('model call budget exhausted')
        o.model_calls += 1
        digest = content_hash(projected)
        o.journal.append('MODEL_CALL', {'agent_id': agent_id, 'payload_sha256': digest,
                                        'prompt_bytes': size, 'ordinal': o.model_calls})
        result = self.backend.generate_structured(agent_id=agent_id, output_model=output_model, payload=projected)
        if content_hash(projected) != digest:
            raise ValueError('model backend mutated its received canonical context')
        return result


def execute_harness(data: OAGInput, *, graph_dir: Path, output_root: Path, request_id: str,
                    policy: HarnessPolicy | None = None, paperseek_client=None,
                    llm_backend=None, ontology_backend=None, ontology_context=None) -> dict:
    """Run the canonical pipeline once; replay a verified immutable result by ID."""
    if not request_id.strip():
        raise ValueError('request_id must be nonblank')
    policy = policy or HarnessPolicy()
    # Freeze source graph identity and provider configuration for idempotence.
    providers = {}
    for name, provider in [('paperseek', paperseek_client), ('llm', llm_backend), ('ontology', ontology_backend)]:
        cfg = getattr(provider, 'config', None)
        cfg = cfg.model_dump(mode='json') if hasattr(cfg, 'model_dump') else str(cfg)
        providers[name] = {'type': type(provider).__name__, 'configuration_sha256': content_hash(cfg)}
    snapshot = {'policy': policy.model_dump(mode='json'), 'providers': providers,
                'implementation_sha256': implementation_sha256(),
                'ontology_context_sha256': content_hash(ontology_context),
                'graph': {n: __import__('hashlib').sha256((graph_dir / n).read_bytes()).hexdigest()
                          for n in ('nodes.csv', 'edges.csv')}}
    input_sha, snapshot_sha = content_hash(data.model_dump(mode='json')), content_hash(snapshot)
    output_root.mkdir(parents=True, exist_ok=True)
    run = output_root / ('RUN-' + content_hash(request_id)[:24])
    try:
        run.mkdir()
    except FileExistsError:
        audit = verify_run(run)
        if audit['input_sha256'] != input_sha or audit['snapshot_sha256'] != snapshot_sha:
            raise ValueError('request ID is already bound to different inputs or configuration')
        return json.loads((run / 'result.json').read_text(encoding='utf-8'))
    journal = Journal(run)
    journal.append('RUN_STARTED', {'input_sha256': input_sha, 'snapshot_sha256': snapshot_sha,
                                  'snapshot': snapshot})
    observer = _Observer(journal, policy)
    try:
        runtime = build_csv_oag_runtime(
            graph_dir, observer=observer, paperseek_client=paperseek_client, ontology_context=ontology_context,
            llm_backend=_BoundedBackend(llm_backend, observer) if llm_backend else None,
            ontology_backend=_BoundedBackend(ontology_backend, observer) if ontology_backend else None)
        result = runtime.execute(RuntimeRequest(request_id=request_id, pipeline_id=OAG_PIPELINE_ID,
                                                payload=data.model_dump(mode='json'))).model_dump(mode='json')
    except Exception as exc:
        result = {'status': 'FAILED', 'error': {'type': type(exc).__name__, 'message': str(exc)}}
    result['harness'] = {'run_directory': str(run), 'input_sha256': input_sha,
                         'snapshot_sha256': snapshot_sha, 'model_calls': observer.model_calls}
    (run / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    journal.append('RUN_FINISHED', {'status': result['status'], 'result_sha256': content_hash(result)})
    verify_run(run)
    return result


def implementation_sha256() -> str:
    """A saved run cannot silently replay under changed OAG/core source code."""
    import hashlib
    root=Path(__file__).resolve().parents[1]
    files=[root/'__init__.py']
    for namespace in ('oag','fusion','runtime','knowledge','llm'):
        files.extend((root/namespace).rglob('*.py'))
    return content_hash({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
