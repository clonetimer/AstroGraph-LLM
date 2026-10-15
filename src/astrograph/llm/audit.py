"""Per-task LLM wire audit. No credentials/headers; captured content remains DATA.

Full mode records exact HTTP JSON bytes, hashes and per-attempt response bodies.
Hashes-only mode intentionally cannot be byte-replayed. No automatic replay or
execution of saved model output is provided.
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json

_ACTIVE = ContextVar('astrograph_llm_wire_audit', default=None)


def wire_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


class CallAudit:
    def __init__(self, writer, prefix: str, *, mode: str = 'full'):
        self.writer = writer
        self.prefix = prefix
        self.mode = mode
        self.attempts = 0
        self.current = None
        self.items = []

    def save(self, suffix, raw):
        name = self.prefix + suffix
        digest = hashlib.sha256(raw).hexdigest()
        artifact = {'name':name, 'sha256':digest, 'bytes':len(raw)}
        if self.mode == 'full':
            self.writer(name, raw)
        else:
            artifact['name'] = None
        return artifact

    def event(self, suffix, value):
        # Event metadata itself has no raw prompt/response.
        self.writer(self.prefix + suffix, wire_bytes(value))

    def request(self, body):
        self.attempts += 1
        self.current = {'attempt':self.attempts,
                        'request':self.save(f'-attempt-{self.attempts:03d}.request.json',wire_bytes(body)),
                        'model':body.get('model'), 'status':'REQUEST_PREPARED',
                        'transport_observed':False}
        self.event(f'-attempt-{self.attempts:03d}.prepared.json', self.current)

    def response(self, raw: bytes, *, http_status=200):
        if self.current is None:
            return
        self.current['response'] = self.save(f'-attempt-{self.attempts:03d}.response.bin',raw)
        self.current['http_status'] = http_status
        self.current['transport_observed'] = True

    def complete(self, document=None, error=None):
        if self.current is None:
            return
        if document is not None and 'response' not in self.current:
            # Injected backend tests are not actual wire captures.
            self.current['parsed_response'] = self.save(f'-attempt-{self.attempts:03d}.parsed.json',wire_bytes(document))
        self.current['status'] = 'RESPONSE_RECEIVED' if error is None else 'FAILED'
        if error is not None:
            self.current['error_type'] = type(error).__name__
        self.items.append(self.current)
        self.event(f'-attempt-{self.attempts:03d}.receipt.json',self.current)
        self.current = None

    def index(self):
        self.event('.index.json', {'schema_version':'astrograph-llm-wire-audit.v1',
            'trace_mode':self.mode, 'attempts':self.items,
            'credentials_recorded':False, 'automatic_replay_allowed':False,
            'scope':'HTTP_JSON_BYTES_NOT_SERVER_CHAT_TEMPLATE_OR_TOKEN_IDS'})


@contextmanager
def audit_scope(audit):
    token = _ACTIVE.set(audit)
    try:
        yield audit
    finally:
        try:
            if audit is not None:
                audit.index()
        finally:
            # A full disk or failed writer must not leak this task's recorder
            # into subsequent logical calls in the same process.
            _ACTIVE.reset(token)


def current_audit():
    return _ACTIVE.get()
