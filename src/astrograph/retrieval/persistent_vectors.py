"""Scope- and revision-bound disk-backed embedding cache.

This deliberately implements exact search over verified float32 vectors, not a
clustered ANN service. Payload contains neither raw source text nor credentials.
Only reviewed, still-applicable excerpts may be placed in the cache. A source
or model revision change yields a new key, never silently reuses stale vectors.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
from typing import Iterator, Sequence

import numpy as np

SCHEMA = "astrograph-scope-pinned-vector-cache.v1"
MAX_DIM = 8192
MAX_ROWS = 100000


def model_revision_sha256(root: str | Path, *, max_bytes: int = 12 * 1024**3) -> str:
    """Hash model files, not merely its directory path or modification time."""
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("embedding model must be a local directory")
    paths = sorted(p for p in root.rglob('*') if p.is_file())
    if not paths or sum(p.stat().st_size for p in paths) > max_bytes:
        raise ValueError("invalid or oversized embedding model revision")
    digest = hashlib.sha256()
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("model directory cannot include symlinks")
        digest.update(path.relative_to(root).as_posix().encode() + b'\0')
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                digest.update(block)
    return digest.hexdigest()


def identity(chunk: object) -> str:
    return hashlib.sha256(json.dumps({
        'source_sha256': chunk.source_sha256, 'locators': list(chunk.source_locators),
        'text_sha256': hashlib.sha256(chunk.text.encode()).hexdigest(),
        'safety': chunk.content_safety_status, 'review': chunk.review_status,
        'modalities': list(chunk.modalities),
    }, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def _encode(vector: np.ndarray) -> tuple[bytes, int, str]:
    v = np.asarray(vector, dtype='<f4').ravel()
    if not 1 <= len(v) <= MAX_DIM or not np.isfinite(v).all():
        raise ValueError("non-finite or oversized embedding vector")
    norm = float(np.linalg.norm(v))
    if not 0.98 <= norm <= 1.02:
        raise ValueError("embedding vector must be unit-normalized")
    raw = v.tobytes()
    return raw, len(v), hashlib.sha256(raw).hexdigest()


class PersistentVectorCache:
    """Single-machine SQLite/WAL cache, exact ranking outside SQLite.

    Multiple processes can read and write independently. Offline callers provide
    a fully hashed model revision and exact equipment scope.
    """
    def __init__(self, path: str | Path):
        requested = Path(path).expanduser()
        if requested.is_symlink():
            raise ValueError("vector cache DB must not be a symlink")
        self.path = requested.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS vectors (
                equipment_scope TEXT NOT NULL, model_sha256 TEXT NOT NULL,
                source_identity TEXT NOT NULL, dimension INTEGER NOT NULL,
                vector BLOB NOT NULL, vector_sha256 TEXT NOT NULL,
                PRIMARY KEY(equipment_scope, model_sha256, source_identity))''')
            db.execute('CREATE INDEX IF NOT EXISTS idx_vectors_scope ON vectors(equipment_scope, model_sha256)')
        if os.name == 'posix':
            try: os.chmod(self.path, 0o600)
            except OSError: pass

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=15)
        db.execute('PRAGMA busy_timeout=15000')
        db.execute('PRAGMA journal_mode=WAL')
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _validate_scope(scope: str, revision: str) -> None:
        if not scope or scope.strip().upper() in {'UNKNOWN', 'UNSPECIFIED'} or len(scope) > 160:
            raise ValueError("exact equipment scope required for persisted vectors")
        if len(revision) != 64 or any(ch not in '0123456789abcdef' for ch in revision):
            raise ValueError("fully hashed model revision required")

    def get_many(self, *, scope: str, revision: str, identities: Sequence[str]) -> dict[str, np.ndarray]:
        self._validate_scope(scope, revision)
        if len(identities) > MAX_ROWS:
            raise ValueError('oversized vector cache query')
        if not identities:
            return {}
        selected = set(identities)
        vectors: dict[str, np.ndarray] = {}
        with self._db() as db:
            # SQLite bind-variable budget varies between distributions.
            for i in range(0, len(identities), 300):
                batch = list(identities[i:i+300])
                marks = ','.join('?' for _ in batch)
                rows = db.execute('SELECT source_identity,dimension,vector,vector_sha256 FROM vectors '
                    f'WHERE equipment_scope=? AND model_sha256=? AND source_identity IN ({marks})',
                    (scope, revision, *batch)).fetchall()
                for key, dimension, raw, sha in rows:
                    if (key not in selected or dimension < 1 or dimension > MAX_DIM
                            or len(raw) != dimension * 4 or hashlib.sha256(raw).hexdigest() != sha):
                        raise ValueError('persisted embedding integrity check failed')
                    array = np.frombuffer(raw, dtype='<f4').copy()
                    if not np.isfinite(array).all() or not 0.98 <= float(np.linalg.norm(array)) <= 1.02:
                        raise ValueError('persisted embedding vector invalid')
                    vectors[key] = array
        if len({v.shape[0] for v in vectors.values()}) > 1:
            raise ValueError('persisted vector dimensions disagree')
        return vectors

    def put_many(self, *, scope: str, revision: str, vectors: dict[str, np.ndarray]) -> int:
        self._validate_scope(scope, revision)
        if len(vectors) > MAX_ROWS:
            raise ValueError('oversized vector cache write')
        rows = []
        for key, vector in vectors.items():
            if len(key) != 64 or any(ch not in '0123456789abcdef' for ch in key):
                raise ValueError('invalid source identity')
            raw, dim, digest = _encode(vector)
            rows.append((scope, revision, key, dim, raw, digest))
        if len({r[3] for r in rows}) > 1:
            raise ValueError('inconsistent embedding dimensions')
        with self._db() as db:
            count = db.execute('SELECT COUNT(*) FROM vectors').fetchone()[0]
            if count + len(rows) > MAX_ROWS:
                raise ValueError('vector cache capacity requires deliberate retention policy')
            db.executemany('''INSERT INTO vectors VALUES (?,?,?,?,?,?)
                ON CONFLICT(equipment_scope,model_sha256,source_identity)
                DO UPDATE SET dimension=excluded.dimension, vector=excluded.vector,
                    vector_sha256=excluded.vector_sha256''', rows)
        return len(rows)

    def delete_scope(self, scope: str) -> int:
        if not scope or scope.upper() == 'UNKNOWN':
            raise ValueError('explicit scope required to delete cached vectors')
        with self._db() as db:
            return db.execute('DELETE FROM vectors WHERE equipment_scope=?', (scope,)).rowcount


def semantic_rank_with_cache(queries: Sequence[str], chunks: Sequence[object], *,
                             model: object, model_path: str | Path, scope: str | None,
                             cache_path: str | Path | None = None) -> tuple[list[list[str]], dict]:
    """Rank task-safe texts locally; persist ONLY independently approved references.

    A parser's HUMAN_REVIEWED status means its *representation* was accepted
    for this upload, not that an expert approved the *knowledge*. Uploaded
    materials remain ephemeral, even if the same bytes recur in another job.
    """
    eligible = [c for c in chunks if c.review_status in {'HUMAN_REVIEWED', 'EXPERT_APPROVED_REFERENCE'}
                and c.content_safety_status in {'PASS', 'REVIEWED'}]
    persistable = [c for c in eligible if c.review_status == 'EXPERT_APPROVED_REFERENCE'
                   and c.content_safety_status == 'REVIEWED']
    store = None
    revision = None
    if cache_path and scope and scope.upper() not in {'UNKNOWN', 'UNSPECIFIED'}:
        revision = model_revision_sha256(model_path)
        store = PersistentVectorCache(cache_path)
    keys = [identity(c) for c in eligible]
    persistent_keys = {identity(c) for c in persistable}
    existing = store.get_many(scope=scope, revision=revision, identities=list(persistent_keys)) if store else {}
    missing = {key:c for key,c in zip(keys,eligible) if key not in existing}
    new: dict[str,np.ndarray] = {}
    if missing:
        values = np.asarray(model.encode([c.text[:2500] for c in missing.values()],
                                         normalize_embeddings=True), dtype='<f4')
        if len(values)!=len(missing):
            raise ValueError('model returned unexpected number of embeddings')
        new=dict(zip(missing,values))
        for vector in new.values(): _encode(vector)
    to_persist = {key:vector for key,vector in new.items() if key in persistent_keys}
    if store and to_persist:
        store.put_many(scope=scope,revision=revision,vectors=to_persist)
    vecs = [existing.get(key, new.get(key)) for key in keys]
    if len({len(v) for v in vecs}) > 1:
        raise ValueError('embedding dimension mismatch')
    query_values = np.asarray(model.encode([q[:900] for q in queries],
                                           normalize_embeddings=True), dtype='<f4') if queries else np.empty((0,0))
    ranks=[]
    for vec in query_values:
        _encode(vec)
        if vecs and len(vec)!=len(vecs[0]):
            raise ValueError('query/document embedding dimension mismatch')
        scored=[(float(vec @ doc), c.chunk_id) for c,doc in zip(eligible,vecs)]
        ranks.append([name for score,name in sorted(scored,key=lambda v:(-v[0],v[1]))
                      if math.isfinite(score) and score >= .25][:16])
    return ranks, {'status':'SUCCEEDED', 'persistence':'SQLITE_EXACT' if store else 'MEMORY_ONLY',
                   'source_vectors_reused':len(existing), 'source_vectors_created':len(new),
                   'source_vectors_persisted':len(to_persist) if store else 0,
                   'ephemeral_uploaded_vectors':len(eligible)-len(persistable),
                   'indexed_reviewed_fragments':len(persistable),
                   'model_sha256':revision,'network_download_allowed':False,
                   'claim_boundary':'UPLOAD_EMBEDDINGS_ARE_EPHEMERAL; SIMILARITY_NOT_CAUSAL_PROOF'}
