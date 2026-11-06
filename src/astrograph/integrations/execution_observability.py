"""Host-owned execution receipts. Never persist prompts, responses, keys or endpoints.

Independent implementation inspired by task-scoped observability, not an Agent
runtime. Every request receipt represents one invocation of the transport. A
received HTTP response is distinct from validated/successful model analysis.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import threading
import time
from .stage_execution import phase_model, check_cancelled, AnalysisCancelled

SCHEMA = 'astrograph-execution.v1'
_ACTIVE = ContextVar('astrograph_execution_journal', default=None)
EVENTS = frozenset({'RUN_STARTED', 'INPUT_VALIDATED', 'MODEL_ANALYSIS', 'FOLLOW_UP',
    'SCHEMA_VALIDATED', 'CITATIONS_VALIDATED', 'CLAIM_REGISTRY_BUILT',
    'INDEPENDENT_VERIFICATION', 'VERIFICATION_FAILED', 'CHUNK_ANALYSIS', 'CHUNK_RESTORED',
    'SYNTHESIS', 'REPORT_WRITE', 'RUN_FINISHED', 'INTERRUPTED'})
STATUSES = frozenset({'RUNNING', 'SUCCEEDED', 'FAILED', 'INTERRUPTED', 'NOT_CONFIGURED',
    'INPUT_TOO_LARGE', 'CONTEXT_TOO_LARGE', 'REPLAYED', 'CANCELLED'})
ARTIFACTS = frozenset({'report.json', 'report.md', 'conversation.json'})
MAX_BYTES = 8 * 1024 * 1024
_ID = re.compile(r'^[a-f0-9]{32}$')
_HASH = re.compile(r'^[a-f0-9]{64}$')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _text(value, maximum=300):
    return ''.join(c for c in str(value) if ord(c) >= 32)[:maximum]


def _count(value):
    return value if type(value) is int and 0 <= value <= 10**12 else None


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def retry_share_violation(operation, *, attempts=8):
    """Retry transient file-sharing denials, not model calls or real access errors.

    Windows reader handles and atomic receipt replacements briefly conflict;
    repeating the same path operation is safe, but invoking an LLM again is not.
    """
    for attempt in range(attempts):
        try:
            return operation()
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(0.005 * (attempt + 1))


def _write(path, document):
    raw = (json.dumps(document, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    if len(raw) > MAX_BYTES:
        raise ValueError('EXECUTION_RECEIPT_TOO_LARGE')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + secrets.token_hex(8) + '.tmp')
    try:
        with temporary.open('xb') as handle:
            handle.write(raw)
            handle.flush()
        retry_share_violation(lambda: temporary.replace(path))
    finally:
        temporary.unlink(missing_ok=True)


def _runs_root(root):
    root = Path(root)
    for path in (root, root / 'execution', root / 'execution' / 'runs'):
        if path.is_symlink():
            raise ValueError('EXECUTION_PATH_INVALID')
    return root / 'execution' / 'runs'


def _run_path(root, run_id):
    if not isinstance(run_id, str) or not _ID.fullmatch(run_id):
        raise ValueError('EXECUTION_RUN_INVALID')
    path = _runs_root(root) / run_id
    if path.is_symlink() or (path / 'run.json').is_symlink():
        raise ValueError('EXECUTION_PATH_INVALID')
    return path


def _load(root, task_id, run_id):
    path = _run_path(root, run_id) / 'run.json'
    def read_receipt():
        if path.stat().st_size > MAX_BYTES:
            raise ValueError('EXECUTION_RECEIPT_TOO_LARGE')
        return json.loads(path.read_text(encoding='utf-8'))
    value = retry_share_violation(read_receipt)
    if (not isinstance(value, dict) or value.get('schema_version') != SCHEMA
            or value.get('task_id') != task_id or value.get('run_id') != run_id
            or value.get('status') not in STATUSES):
        raise ValueError('EXECUTION_BINDING_INVALID')
    for field, maximum in [('events', 4000), ('requests', 1024), ('materials', 1024), ('artifacts', 3)]:
        items = value.get(field)
        if not isinstance(items, list) or len(items) > maximum or any(not isinstance(x, dict) for x in items):
            raise ValueError('EXECUTION_RECORD_INVALID')
    for field in ('task_id', 'run_id', 'kind', 'status', 'started_at', 'ended_at', 'model', 'mode', 'input_sha256', 'reason_code'):
        if value.get(field) is not None and not isinstance(value[field], str):
            raise ValueError('EXECUTION_RECORD_INVALID')
    if value.get('kind') not in {'analysis', 'follow_up'}:
        raise ValueError('EXECUTION_RECORD_INVALID')
    for item in value['requests']:
        if item.get('status') not in {'STARTED', 'FAILED', 'INTERRUPTED', 'RESPONSE_RECEIVED'}:
            raise ValueError('EXECUTION_RECORD_INVALID')
        if item.get('stage') not in EVENTS:
            raise ValueError('EXECUTION_RECORD_INVALID')
        for field in ('request_sha256', 'started_at', 'ended_at'):
            if item.get(field) is not None and not isinstance(item[field], str):
                raise ValueError('EXECUTION_RECORD_INVALID')
    # All public records are constructed below from allowlisted fields, including
    # restored backups. Unknown keys never appear in APIs or downloadable receipts.
    return value


def material_inventory(bundle, receipts=()):
    groups = {}
    original = {x.get('filename'): x.get('original_sha256') for x in receipts if isinstance(x, dict)}
    for source in bundle.get('sources', []):
        key = (source['role'], source['filename'])
        item = groups.setdefault(key, {'role': _text(key[0]), 'filename': _text(key[1], 512),
            'characters': 0, 'segments': 0, 'first_locator': _text(source.get('locator', ''), 600),
            'last_locator': '', 'source_ids': [], 'source_sha256': [],
            'original_sha256': original.get(key[1])})
        item['characters'] += len(source.get('text', ''))
        item['segments'] += 1
        item['last_locator'] = _text(source.get('locator', ''), 600)
        item['source_ids'].append(source['source_id'])
        digest = source.get('sha256')
        if isinstance(digest, str) and _HASH.fullmatch(digest) and digest not in item['source_sha256']:
            item['source_sha256'].append(digest)
    return list(groups.values())


def _usage(document):
    value = document.get('usage') if isinstance(document, dict) else None
    value = value if isinstance(value, dict) else {}
    return {'input_tokens': _count(value.get('prompt_tokens', value.get('input_tokens'))),
            'output_tokens': _count(value.get('completion_tokens', value.get('output_tokens')))}


def summarize_usage(requests):
    out = {'http_requests': len(requests),
           'failed_requests': sum(r.get('status') == 'FAILED' for r in requests),
           'estimated_cost': None, 'cost_status': 'PRICING_NOT_CONFIGURED'}
    for field, prefix in [('input_tokens', 'input'), ('output_tokens', 'output')]:
        values = [_count(r.get('usage', {}).get(field)) for r in requests]
        known = [v for v in values if v is not None]
        out[field] = sum(known) if known and len(known) == len(values) else None
        out['known_' + field] = sum(known)
        out[prefix + '_reported_requests'] = len(known)
    out['usage_complete'] = bool(requests) and all(out[k] is not None for k in ('input_tokens', 'output_tokens'))
    return out


class ExecutionJournalError(RuntimeError):
    """Terminal audit-storage failure; never an invitation to repeat a paid call."""


class RunJournal:
    def __init__(self, root, *, task_id, kind, bundle, model='', mode='', receipts=()):
        if kind not in {'analysis', 'follow_up'} or bundle.get('task_id') != task_id:
            raise ValueError('EXECUTION_BINDING_INVALID')
        self.root = Path(root)
        self.run_id = secrets.token_hex(16)
        self.path = _run_path(root, self.run_id)
        self.path.mkdir(parents=True, exist_ok=False)
        self.lock = threading.RLock()
        self.stage = 'FOLLOW_UP' if kind == 'follow_up' else 'MODEL_ANALYSIS'
        self.data = {'schema_version': SCHEMA, 'task_id': task_id, 'run_id': self.run_id,
            'kind': kind, 'status': 'RUNNING', 'started_at': _now(), 'ended_at': None,
            'model': _text(model), 'mode': mode if mode in {'local', 'online'} else 'configured',
            'input_sha256': bundle.get('input_sha256'), 'materials': material_inventory(bundle, receipts),
            'events': [], 'requests': [], 'artifacts': [], 'reason_code': None}
        self.event('RUN_STARTED')

    def _save(self):
        try:
            _write(self.path / 'run.json', self.data)
        except (OSError, ValueError, TypeError) as exc:
            raise ExecutionJournalError('EXECUTION_RECORD_WRITE_FAILED') from exc

    def event(self, code, **counts):
        if code not in EVENTS or set(counts) - {'completed', 'total', 'index'}:
            raise ValueError('EXECUTION_EVENT_INVALID')
        with self.lock:
            if self.data['status'] != 'RUNNING':
                raise ValueError('EXECUTION_ALREADY_FINISHED')
            if len(self.data['events']) >= 4000:
                raise ValueError('EXECUTION_EVENT_LIMIT')
            self.data['events'].append({'sequence': len(self.data['events']) + 1,
                'at': _now(), 'code': code, 'counts': {k: _count(v) for k, v in counts.items()}})
            if code in {'MODEL_ANALYSIS', 'FOLLOW_UP', 'CHUNK_ANALYSIS', 'SYNTHESIS'}:
                self.stage = code
            self._save()

    def identify_model(self, model):
        """Bind the configured model before the first request, not an endpoint or key."""
        with self.lock:
            if self.data['status'] != 'RUNNING' or self.data['requests']:
                raise ValueError('EXECUTION_MODEL_ALREADY_BOUND')
            self.data['model'] = _text(model)
            self._save()

    def http_call(self, invoke, *, request_sha256):
        check_cancelled()
        if not isinstance(request_sha256, str) or not _HASH.fullmatch(request_sha256):
            raise ValueError('EXECUTION_HASH_INVALID')
        with self.lock:
            if self.data['status'] != 'RUNNING' or len(self.data['requests']) >= 1024:
                raise ValueError('EXECUTION_REQUEST_REJECTED')
            item = {'sequence': len(self.data['requests']) + 1, 'stage': self.stage,
                'request_sha256': request_sha256, 'started_at': _now(), 'ended_at': None,
                'status': 'STARTED', 'model': _text(phase_model() or self.data['model']), 'http_status': None, 'latency_ms': None,
                'usage': {'input_tokens': None, 'output_tokens': None}}
            self.data['requests'].append(item)
            self._save()  # Journal failure blocks sending, rather than an unrecorded paid call.
        start = time.monotonic()
        def response_observer(raw, *, http_status=None):
            # Deliberately ignore raw response bytes; only the transport status is retained.
            item['http_status'] = http_status if type(http_status) is int and 100 <= http_status <= 599 else None
        try:
            result = invoke(response_observer)
            item.update(status='RESPONSE_RECEIVED', usage=_usage(result))
            return result
        except Exception:
            item['status'] = 'FAILED'
            raise
        finally:
            with self.lock:
                item.update(ended_at=_now(), latency_ms=round((time.monotonic() - start)*1000, 3))
                self._save()

    def artifact(self, name, raw):
        with self.lock:
            if name not in ARTIFACTS or self.data['status'] != 'RUNNING' or not isinstance(raw, bytes):
                raise ValueError('EXECUTION_ARTIFACT_INVALID')
            if len(raw) > 16 * 1024 * 1024:
                raise ValueError('EXECUTION_ARTIFACT_TOO_LARGE')
            folder = self.path / 'artifacts'
            if folder.is_symlink():
                raise ValueError('EXECUTION_PATH_INVALID')
            folder.mkdir(exist_ok=True)
            with (folder / name).open('xb') as handle:
                handle.write(raw)
            self.data['artifacts'].append({'name': name, 'bytes': len(raw), 'sha256': _digest(raw)})
            self._save()

    def finish(self, status, reason_code=None):
        with self.lock:
            if status not in STATUSES - {'RUNNING'}:
                raise ValueError('EXECUTION_STATUS_INVALID')
            if self.data['status'] != 'RUNNING':
                return
            # Application codes only, never free-form exception/provider messages.
            reason = reason_code if isinstance(reason_code, str) and re.fullmatch(r'[A-Z][A-Z0-9_]{0,79}', reason_code) else None
            self.event('RUN_FINISHED')
            self.data.update(status=status, ended_at=_now(), reason_code=reason)
            self._save()


def current_run():
    return _ACTIVE.get()


@contextmanager
def run_scope(run):
    token = _ACTIVE.set(run)
    try:
        yield run
    except AnalysisCancelled:
        run.finish('CANCELLED', 'USER_CANCELLED')
        raise
    finally:
        try:
            if run.data['status'] == 'RUNNING':
                run.finish('FAILED', 'EXECUTION_DID_NOT_FINISH')
        finally:
            _ACTIVE.reset(token)


def event(code, **counts):
    run = current_run()
    if run is not None:
        run.event(code, **counts)


def _project(value):
    keys = ('schema_version', 'task_id', 'run_id', 'kind', 'status', 'started_at', 'ended_at',
            'model', 'mode', 'input_sha256', 'reason_code')
    out = {k: value.get(k) for k in keys}
    out['events'] = []
    for x in value['events']:
        if x.get('code') not in EVENTS:
            continue
        counts = x.get('counts') if isinstance(x.get('counts'), dict) else {}
        out['events'].append({'sequence': _count(x.get('sequence')), 'at': _text(x.get('at', ''), 80),
            'code': x['code'], 'counts': {k: _count(counts[k]) for k in ('completed', 'total', 'index') if k in counts}})
    out['requests'] = []
    for r in value.get('requests', []):
        if not isinstance(r, dict):
            continue
        item = {k: r.get(k) for k in ('sequence', 'stage', 'request_sha256', 'started_at', 'ended_at',
                                     'status', 'http_status', 'latency_ms')}
        item['model'] = _text(r.get('model', value.get('model', '')))
        item['sequence'] = _count(r.get('sequence'))
        item['http_status'] = _count(r.get('http_status'))
        latency = r.get('latency_ms')
        item['latency_ms'] = latency if type(latency) in (int, float) and math.isfinite(latency) and latency >= 0 else None
        usage = r.get('usage') if isinstance(r.get('usage'), dict) else {}
        item['usage'] = {k: _count(usage.get(k)) for k in ('input_tokens', 'output_tokens')}
        out['requests'].append(item)
    out['materials'] = []
    for x in value['materials']:
        item = {}
        for key in ('role', 'filename', 'first_locator', 'last_locator', 'original_sha256'):
            if x.get(key) is not None and not isinstance(x[key], str):
                raise ValueError('EXECUTION_RECORD_INVALID')
            item[key] = x.get(key)
        for key in ('characters', 'segments'):
            item[key] = _count(x.get(key))
        for key in ('source_ids', 'source_sha256'):
            if not isinstance(x.get(key), list) or any(not isinstance(y, str) for y in x[key]):
                raise ValueError('EXECUTION_RECORD_INVALID')
            item[key] = x[key]
        out['materials'].append(item)
    out['artifacts'] = []
    for x in value['artifacts']:
        if (x.get('name') not in ARTIFACTS or _count(x.get('bytes')) is None
                or not isinstance(x.get('sha256'), str) or not _HASH.fullmatch(x['sha256'])):
            raise ValueError('EXECUTION_RECORD_INVALID')
        out['artifacts'].append({k: x[k] for k in ('name', 'bytes', 'sha256')})
    out['usage'] = summarize_usage(out['requests'])
    return out


def execution_snapshot(root, task_id):
    runs, unavailable = [], 0
    for path in sorted(_runs_root(root).glob('*/run.json')):
        try:
            runs.append(_project(_load(root, task_id, path.parent.name)))
        except (OSError, ValueError, TypeError, AttributeError):
            unavailable += 1
    runs.sort(key=lambda x: (x.get('started_at') or '', x['run_id']))
    return {'schema_version': SCHEMA, 'task_id': task_id, 'legacy': not runs and not unavailable,
            'unavailable_runs': unavailable, 'runs': runs,
            'usage': summarize_usage([r for run in runs for r in run['requests']]),
            'scope': 'RECORDED_RUNS_ONLY; no historical usage is inferred'}


def recover_interrupted(root, task_id):
    for path in _runs_root(root).glob('*/run.json'):
        try:
            value = _load(root, task_id, path.parent.name)
            if value['status'] != 'RUNNING':
                continue
            value.update(status='INTERRUPTED', ended_at=_now(), reason_code='APPLICATION_RESTARTED')
            if len(value['events']) < 4000:
                value['events'].append({'sequence': len(value['events']) + 1, 'at': _now(),
                                        'code': 'INTERRUPTED', 'counts': {}})
            for request in value.get('requests', []):
                if request.get('status') == 'STARTED':
                    request['status'] = 'INTERRUPTED'  # Server may still have billed it; usage stays unknown.
            _write(path, value)
        except (OSError, ValueError, TypeError, AttributeError):
            continue


def read_artifact(root, task_id, run_id, name):
    if name not in ARTIFACTS:
        raise ValueError('EXECUTION_ARTIFACT_INVALID')
    value = _load(root, task_id, run_id)
    if value['status'] == 'RUNNING':
        raise ValueError('EXECUTION_NOT_FINISHED')
    record = next((x for x in value.get('artifacts', []) if x.get('name') == name), None)
    if record is None:
        raise KeyError('EXECUTION_ARTIFACT_NOT_FOUND')
    folder = _run_path(root, run_id) / 'artifacts'
    path = folder / name
    if folder.is_symlink() or path.is_symlink() or path.stat().st_size > 16*1024*1024:
        raise ValueError('EXECUTION_PATH_INVALID')
    raw = path.read_bytes()
    if len(raw) != record.get('bytes') or _digest(raw) != record.get('sha256'):
        raise ValueError('EXECUTION_ARTIFACT_CHANGED')
    return raw
