"""Admin-governed persistent material knowledge engine settings.

Configuration is scoped to a single *trusted platform deployment*, not to an
end user or an individual spacecraft. Environment activation is process-local;
worker processes explicitly reload the frozen active revision before new runs.
Secret values never enter JSON, logs, history, or GET/probe responses.
"""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import shutil
from threading import RLock
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

_LOCK = RLock()
_FILENAME = 'material_engine_active.v1.json'
_ALLOWED_KEYS = {
    'ASTROGRAPH_MATERIAL_EMBEDDING_MODEL', 'ASTROGRAPH_MATERIAL_QDRANT_URL',
    'ASTROGRAPH_MATERIAL_QDRANT_COLLECTION', 'ASTROGRAPH_MATERIAL_QDRANT_API_KEY',
    'ASTROGRAPH_MATERIAL_VECTOR_CACHE_DB', 'ASTROGRAPH_MATERIAL_COMMUNITY_LLM_ENABLED',
    'ASTROGRAPH_MATERIAL_OCR_ENGINE', 'ASTROGRAPH_MATERIAL_OCR_LANGUAGES',
    'ASTROGRAPH_MATERIAL_RAG_DOC_BUDGET', 'ASTROGRAPH_MATERIAL_RAG_SECTION_BUDGET',
    'ASTROGRAPH_MATERIAL_RAG_HIT_BUDGET', 'ASTROGRAPH_MATERIAL_GRAPH_BUILD_ENABLED',
    'ASTROGRAPH_MATERIAL_GRAPH_LLM_MODEL', 'ASTROGRAPH_MATERIAL_VISION_ENDPOINT',
    'ASTROGRAPH_MATERIAL_VISION_MODEL', 'ASTROGRAPH_MATERIAL_VISION_PAGES',
}


def _qdrant_url_valid(value: str) -> bool:
    parsed = urlsplit(value)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        return False
    if parsed.path not in ('', '/') or parsed.query or parsed.fragment:
        return False
    # Don't allow arbitrary network destinations controlled from the browser.
    # Site operators may allowlist TLS endpoints with a server-side variable.
    trusted = set(filter(None, (s.strip().lower() for s in
        os.environ.get('ASTROGRAPH_ENGINE_QDRANT_ALLOWED_HOSTS', '').split(','))))
    host = parsed.hostname.lower()
    return host in ('localhost', '127.0.0.1', '::1') or (parsed.scheme == 'https' and host in trusted)


class MaterialEngineSettings(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    embedding_model: str = Field(default='', max_length=700)
    vector_backend: Literal['sqlite', 'qdrant'] = 'sqlite'
    qdrant_url: str = Field(default='', max_length=300)
    qdrant_collection: str = Field(default='astrograph_material_embeddings', max_length=90)
    qdrant_api_key: str | None = Field(default=None, max_length=4096, exclude=True)
    community_llm_enabled: bool = False
    graph_build_enabled: bool = False
    graph_llm_model: str = Field(default='', max_length=180)
    ocr_engine: Literal['disabled', 'tesseract', 'docling'] = 'disabled'
    ocr_languages: str = Field(default='chi_sim+eng', max_length=50)
    vision_enabled: bool = False
    vision_endpoint: str = Field(default='', max_length=300)
    vision_model: str = Field(default='', max_length=120)
    vision_pages: int = Field(default=0, ge=0, le=5)
    rag_documents: int = Field(default=8, ge=1, le=40)
    rag_sections: int = Field(default=16, ge=1, le=80)
    rag_hits: int = Field(default=5, ge=1, le=20)

    @model_validator(mode='after')
    def validate_services(self) -> 'MaterialEngineSettings':
        if self.embedding_model:
            folder = Path(self.embedding_model).expanduser()
            if not folder.is_absolute():
                raise ValueError('embedding_model must be an absolute local directory')
            allowed = os.environ.get('ASTROGRAPH_ENGINE_MODEL_ROOT', '').strip()
            if allowed and not folder.resolve().is_relative_to(Path(allowed).expanduser().resolve()):
                raise ValueError('embedding_model must stay within ASTROGRAPH_ENGINE_MODEL_ROOT')
        if not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_-]{1,89}', self.qdrant_collection):
            raise ValueError('Qdrant collection has invalid characters')
        # Validate dormant endpoints as well: a saved SQLite profile may later
        # be switched to Qdrant, and GET/probe must never echo embedded URL
        # credentials or accept untrusted remote hosts in the meantime.
        if self.qdrant_url and not _qdrant_url_valid(self.qdrant_url):
            raise ValueError('Qdrant must be loopback or an allowlisted HTTPS host')
        if self.qdrant_api_key and self.vector_backend != 'qdrant':
            raise ValueError('A Qdrant API key requires vector_backend=qdrant')
        if self.vector_backend == 'qdrant':
            if not self.embedding_model:
                raise ValueError('Qdrant needs a local embedding_model')
            if not self.qdrant_url:
                raise ValueError('Qdrant must be loopback or an allowlisted HTTPS host')
        if not re.fullmatch(r'[a-zA-Z0-9_]+(?:\+[a-zA-Z0-9_]+)*', self.ocr_languages):
            raise ValueError('Invalid OCR language pack identifiers')
        if self.vision_enabled:
            from urllib.parse import urlsplit
            from ipaddress import ip_address
            url = urlsplit(self.vision_endpoint)
            try:
                localhost = ip_address(url.hostname or '').is_loopback
            except ValueError:
                localhost = url.hostname == 'localhost'
            if (url.scheme != 'http' or not localhost or url.username or url.password or
                url.query or url.fragment or not self.vision_endpoint.endswith('/chat/completions')
                or not self.vision_model or not 1 <= self.vision_pages <= 5):
                raise ValueError('Visual model requires loopback HTTP /chat/completions, model and 1-5 pages')
        return self


class EngineUpdateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expected_revision: str | None = Field(default=None, max_length=100)
    settings: MaterialEngineSettings


class EngineRollbackRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expected_revision: str = Field(min_length=10, max_length=100)
    target_revision: str = Field(min_length=10, max_length=100)


class MaterialEngineRegistry:
    def __init__(self, platform_root: Path, secret_store: Any = None):
        self.root = Path(platform_root).expanduser().resolve() / '.engine_settings'
        self.secrets = secret_store

    @property
    def active_file(self) -> Path:
        return self.root / _FILENAME

    @contextmanager
    def _exclusive_revision_write(self):
        """Protect compare-and-swap writes between independent API processes on POSIX."""
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        lockfile = self.root / '.update.lock'
        with lockfile.open('a+b') as handle:
            if os.name == 'posix':
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == 'posix':
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read(self, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('schema_version') != 'astrograph-material-engine-settings.v1':
            raise ValueError('Engine settings schema mismatch')
        payload['settings'] = MaterialEngineSettings.model_validate(payload['settings']).model_dump(exclude={'qdrant_api_key'})
        return payload

    def _write(self, path: Path, payload: dict[str, Any]) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name == 'posix':
            os.chmod(self.root, 0o700)
        staged = self.root / ('.pending_' + secrets.token_hex(10))
        try:
            with staged.open('x', encoding='utf-8') as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            if os.name == 'posix':
                staged.chmod(0o600)
            os.replace(staged, path)
        finally:
            staged.unlink(missing_ok=True)

    def current(self) -> dict[str, Any]:
        with _LOCK:
            record = self._read(self.active_file)
            if record:
                return {k:v for k,v in record.items() if k != 'qdrant_secret_ref'} | {
                    'qdrant_api_key_configured':bool(record.get('qdrant_secret_ref') and self.secrets
                        and self.secrets.get(record.get('qdrant_secret_ref')))}
            # Present effective pre-existing environment settings to operators,
            # without silently persisting them or ever exposing API keys.
            defaults = MaterialEngineSettings().model_dump(exclude={'qdrant_api_key'})
            names = {
                'embedding_model':'ASTROGRAPH_MATERIAL_EMBEDDING_MODEL',
                'qdrant_url':'ASTROGRAPH_MATERIAL_QDRANT_URL',
                'qdrant_collection':'ASTROGRAPH_MATERIAL_QDRANT_COLLECTION',
                'ocr_languages':'ASTROGRAPH_MATERIAL_OCR_LANGUAGES',
                'graph_llm_model':'ASTROGRAPH_MATERIAL_GRAPH_LLM_MODEL',
                'vision_endpoint':'ASTROGRAPH_MATERIAL_VISION_ENDPOINT',
                'vision_model':'ASTROGRAPH_MATERIAL_VISION_MODEL',
            }
            for field, env_name in names.items():
                if env_name in os.environ:
                    defaults[field] = os.environ[env_name]
            if defaults['qdrant_url']:
                defaults['vector_backend'] = 'qdrant'
            defaults['community_llm_enabled'] = os.environ.get('ASTROGRAPH_MATERIAL_COMMUNITY_LLM_ENABLED','0').lower() in {'1','true','yes'}
            defaults['graph_build_enabled'] = os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_BUILD_ENABLED','0').lower() in {'1','true','yes'}
            defaults['vision_enabled'] = bool(defaults['vision_endpoint'] and defaults['vision_model'])
            defaults['ocr_engine'] = os.environ.get('ASTROGRAPH_MATERIAL_OCR_ENGINE','') or 'disabled'
            for field, env_name in [('rag_documents','ASTROGRAPH_MATERIAL_RAG_DOC_BUDGET'),
                                    ('rag_sections','ASTROGRAPH_MATERIAL_RAG_SECTION_BUDGET'),
                                    ('rag_hits','ASTROGRAPH_MATERIAL_RAG_HIT_BUDGET'),
                                    ('vision_pages','ASTROGRAPH_MATERIAL_VISION_PAGES')]:
                try:
                    defaults[field] = int(os.environ.get(env_name,defaults[field]))
                except ValueError:
                    pass
            return {'schema_version':'astrograph-material-engine-settings.v1',
                    'revision':None,'updated_at':None,'settings':defaults,
                    'qdrant_api_key_configured':bool(os.environ.get('ASTROGRAPH_MATERIAL_QDRANT_API_KEY')),
                    'source':'ENVIRONMENT_OR_DEFAULT_NOT_SAVED'}

    def revisions(self) -> list[dict[str, Any]]:
        with _LOCK:
            entries = []
            for path in sorted(self.root.glob('revision-*.json'), reverse=True)[:30]:
                try:
                    row = self._read(path)
                    if row:
                        entries.append({'revision':row['revision'], 'updated_at':row['updated_at'],
                                        'restored_from':row.get('restored_from')})
                except (ValueError, OSError, KeyError):
                    continue
            return entries

    def save(self, settings: MaterialEngineSettings, *, expected_revision: str | None,
             restored_from: str | None = None, secret_ref: str | None = None) -> dict[str, Any]:
        with _LOCK, self._exclusive_revision_write():
            current = self._read(self.active_file)
            if (current or {}).get('revision') != expected_revision:
                raise RuntimeError('ENGINE_REVISION_CONFLICT')
            config = settings.model_dump(exclude={'qdrant_api_key'})
            revision = 'revision-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '-' + secrets.token_hex(4)
            # Use revision-bound keyring references; no credentials in file or responses.
            if settings.qdrant_api_key:
                if self.secrets is None:
                    raise ValueError('An OS keyring is needed to persist a Qdrant key')
                secret_ref = self.secrets.make_ref('engine-qdrant',revision)
                if not self.secrets.set(secret_ref,settings.qdrant_api_key):
                    raise ValueError('OS keyring refused to store Qdrant key; settings not saved')
            elif secret_ref is None and current and config['vector_backend'] == 'qdrant':
                # A credential belongs to its destination.  Changing Qdrant's
                # endpoint, or disabling Qdrant and re-enabling it later, must
                # require an explicit new key, not silently reuse an old one.
                old_settings = current.get('settings') or {}
                if (old_settings.get('vector_backend') == 'qdrant' and
                    old_settings.get('qdrant_url') == config['qdrant_url']):
                    secret_ref = current.get('qdrant_secret_ref')
            if config['vector_backend'] != 'qdrant':
                secret_ref = None
            record = {'schema_version':'astrograph-material-engine-settings.v1',
                      'revision':revision, 'updated_at':datetime.now(timezone.utc).isoformat(),
                      'settings':config, 'qdrant_secret_ref':secret_ref}
            if restored_from:
                record['restored_from'] = restored_from
            try:
                self._write(self.root / (revision + '.json'),record)
                self._write(self.active_file,record)
            except Exception:
                (self.root / (revision + '.json')).unlink(missing_ok=True)
                raise
            self.activate(record)
            return self.current()

    def active_qdrant_key(self) -> str | None:
        with _LOCK:
            current = self._read(self.active_file)
            return (self.secrets.get(current.get('qdrant_secret_ref')) if current and self.secrets else None)

    def rollback(self, *, expected_revision: str, target_revision: str) -> dict[str, Any]:
        with _LOCK:
            if not re.fullmatch(r'revision-\d{8}T\d{12}-[a-f0-9]{8}', target_revision):
                raise ValueError('Invalid target revision')
            target = self._read(self.root / (target_revision + '.json'))
            if not target or target['revision'] != target_revision:
                raise ValueError('Target revision not found')
            settings = MaterialEngineSettings.model_validate(target['settings'])
            return self.save(settings, expected_revision=expected_revision,
                             restored_from=target_revision, secret_ref=target.get('qdrant_secret_ref'))

    def activate(self, record: dict[str, Any] | None = None) -> None:
        with _LOCK:
            row = record if record is not None else self._read(self.active_file)
            if row is None:
                return
            data = MaterialEngineSettings.model_validate(row['settings'])
            env = {
                'ASTROGRAPH_MATERIAL_EMBEDDING_MODEL':data.embedding_model,
                'ASTROGRAPH_MATERIAL_QDRANT_URL':data.qdrant_url if data.vector_backend=='qdrant' else '',
                'ASTROGRAPH_MATERIAL_QDRANT_COLLECTION':data.qdrant_collection,
                'ASTROGRAPH_MATERIAL_VECTOR_CACHE_DB':str(self.root / 'vectors.sqlite3'),
                'ASTROGRAPH_MATERIAL_COMMUNITY_LLM_ENABLED':'1' if data.community_llm_enabled else '0',
                'ASTROGRAPH_MATERIAL_GRAPH_BUILD_ENABLED':'1' if data.graph_build_enabled else '0',
                'ASTROGRAPH_MATERIAL_GRAPH_LLM_MODEL':data.graph_llm_model,
                'ASTROGRAPH_MATERIAL_OCR_ENGINE':'' if data.ocr_engine=='disabled' else data.ocr_engine,
                'ASTROGRAPH_MATERIAL_OCR_LANGUAGES':data.ocr_languages,
                'ASTROGRAPH_MATERIAL_RAG_DOC_BUDGET':str(data.rag_documents),
                'ASTROGRAPH_MATERIAL_RAG_SECTION_BUDGET':str(data.rag_sections),
                'ASTROGRAPH_MATERIAL_RAG_HIT_BUDGET':str(data.rag_hits),
                'ASTROGRAPH_MATERIAL_VISION_ENDPOINT':data.vision_endpoint if data.vision_enabled else '',
                'ASTROGRAPH_MATERIAL_VISION_MODEL':data.vision_model if data.vision_enabled else '',
                'ASTROGRAPH_MATERIAL_VISION_PAGES':str(data.vision_pages) if data.vision_enabled else '0',
            }
            api_key = (self.secrets.get(row.get('qdrant_secret_ref')) if self.secrets else None)
            env['ASTROGRAPH_MATERIAL_QDRANT_API_KEY'] = api_key or ''
            for name in _ALLOWED_KEYS:
                value = env.get(name,'')
                if value:
                    os.environ[name] = value
                else:
                    os.environ.pop(name,None)
            os.environ['ASTROGRAPH_MATERIAL_ENGINE_REVISION'] = row['revision']


def engine_probe(settings: MaterialEngineSettings, *, network: bool = False,
                 api_key_configured: bool = False, api_key: str | None = None) -> dict[str, Any]:
    """Bounded capability checks, no model download or untrusted document processing."""
    model_dir = Path(settings.embedding_model).expanduser() if settings.embedding_model else None
    checks: dict[str, Any] = {}
    checks['embedding'] = {
        'status':'DISABLED' if not model_dir else ('LOCAL_FILES_PRESENT_NOT_MODEL_LOADED' if model_dir.is_dir() and
           importlib.util.find_spec('sentence_transformers') else 'UNAVAILABLE'),
        'model_directory_exists':bool(model_dir and model_dir.is_dir()),
        'dependency_installed':bool(importlib.util.find_spec('sentence_transformers')),
    }
    checks['vector'] = {'backend':settings.vector_backend,
                        'status':'LOCAL_SQLITE_AVAILABLE' if settings.vector_backend=='sqlite' else 'NOT_PROBED',
                        'key_configured':api_key_configured}
    if settings.vector_backend=='qdrant':
        checks['vector']['status'] = 'QDRANT_NOT_REACHABILITY_TESTED'
        checks['vector']['dependency_installed'] = bool(importlib.util.find_spec('qdrant_client'))
        if network:
            from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler
            class NoRedirect(HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    raise ValueError('Redirect forbidden')
            try:
                # Outbound destination is constrained by schema and server allowlist.
                headers = {'api-key':api_key} if api_key else {}
                req = Request(settings.qdrant_url.rstrip('/') + '/collections', method='GET', headers=headers)
                with build_opener(ProxyHandler({}),NoRedirect()).open(req, timeout=3) as response:
                    checks['vector']['status'] = ('REACHABLE' if response.status == 200 else 'HTTP_ERROR')
            except Exception as exc:
                checks['vector']['status'] = 'UNREACHABLE'
                checks['vector']['error_type'] = type(exc).__name__
    checks['ocr'] = {
        'engine':settings.ocr_engine, 'status':'DISABLED',
    }
    if settings.ocr_engine=='tesseract':
        checks['ocr']['status'] = ('LOCAL_BINARY_PRESENT' if shutil.which('tesseract') and shutil.which('pdftoppm') and shutil.which('pdfinfo') else 'MISSING_BINARY')
    elif settings.ocr_engine=='docling':
        checks['ocr']['status'] = ('DEPENDENCY_INSTALLED_NOT_MODEL_TESTED' if importlib.util.find_spec('docling') else 'MISSING_DEPENDENCY')
    checks['vision'] = {'status':('CONFIGURED_UNVERIFIED_VISUAL_EXTRACTION' if settings.vision_enabled else 'DISABLED'),
                        'automatic_diagnosis_evidence_allowed':False}
    checks['community'] = {'status':'ENABLED_NEEDS_APPROVED_GRAPH_AND_LLM' if settings.community_llm_enabled else 'DISABLED',
                           'automatic_knowledge_publication':False}
    checks['graph_builder'] = {'status':'ENABLED_NEEDS_EXPERT_REVIEW' if settings.graph_build_enabled else 'DISABLED'}
    return {'schema_version':'astrograph-engine-capability-probe.v1','checks':checks,
            'probe_kind':'LOCAL_DEPENDENCY_AND_OPTIONAL_BOUNDED_ENDPOINT_ONLY',
            'production_qualified':False}


def load_active_engine_environment(platform_root: Path) -> None:
    """Load the active revision before a separately spawned material-review worker."""
    from astrograph.integrations.platform_secrets import PlatformSecretStore
    MaterialEngineRegistry(platform_root,PlatformSecretStore()).activate()
