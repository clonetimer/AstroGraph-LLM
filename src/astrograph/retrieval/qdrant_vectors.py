"""Optional Qdrant ANN backend with exact model/scope/session eligibility gates.

Requires a separately deployed Qdrant server and locally installed qdrant-client.
The Qdrant collection contains no source text. Only caller-reviewed chunk IDs
are admissible for retrieval; old/revoked sources cannot contaminate new tasks.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import uuid
from urllib.parse import urlsplit
from ipaddress import ip_address
from typing import Any, Sequence

import numpy as np
from astrograph.retrieval.persistent_vectors import identity, model_revision_sha256, _encode


def _endpoint(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {'http','https'} or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError('invalid Qdrant service URL')
    try: local = ip_address(parsed.hostname).is_loopback
    except ValueError: local = parsed.hostname == 'localhost'
    if parsed.scheme == 'http' and not local:
        raise ValueError('non-local Qdrant requires HTTPS')
    return url


def _pid(scope: str, revision: str, content_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f'astrograph-embeddings:{scope}:{revision}:{content_id}'))


def qdrant_rank(*, queries: Sequence[str], chunks: Sequence[Any], model: Any,
                model_path: str | Path, scope: str, url: str, collection: str,
                api_key: str | None = None, client: Any = None) -> tuple[list[list[str]], dict]:
    if not scope or scope.upper() in {'UNKNOWN','UNSPECIFIED'}:
        raise ValueError('exact equipment scope required for vector server')
    if not collection.startswith('astrograph_') or len(collection) > 80 or not collection.replace('_','').isalnum():
        raise ValueError('vector collection must be administrator-namespaced')
    # Validate even if a test injects a fake client.
    endpoint = _endpoint(url)
    from qdrant_client import QdrantClient, models
    own = client is None
    client = client or QdrantClient(url=endpoint, api_key=api_key, timeout=30)
    try:
        revision = model_revision_sha256(model_path)
        # Only expert-approved references may be persisted at the remote ANN
        # service. Session-upload native text may be searched semantically,
        # but is kept locally in memory for the lifetime of the request.
        eligible = [c for c in chunks if c.review_status == 'EXPERT_APPROVED_REFERENCE'
                    and c.content_safety_status == 'REVIEWED']
        ephemeral = [c for c in chunks if c.review_status == 'HUMAN_REVIEWED'
                     and c.content_safety_status in {'PASS','REVIEWED'}]
        keys = [identity(c) for c in eligible]
        ids = [_pid(scope,revision,key) for key in keys]
        id_positions = {point_id: i for i, point_id in enumerate(ids)}
        # Check the persisted collection first. Previous behaviour re-encoded
        # every reference on every task, defeating most of the benefits of
        # persistence even though vectors were written to Qdrant.
        collection_exists = client.collection_exists(collection)
        existing_dimensions = None
        if collection_exists:
            current = client.get_collection(collection)
            config = current.config.params.vectors
            existing_dimensions = getattr(config,'size',None)
            if not isinstance(existing_dimensions,int) or existing_dimensions < 1:
                raise ValueError('existing Qdrant collection has unsupported vector configuration')
        reused: set[str] = set()
        # Client.retrieve is available on supported modern Qdrant versions.
        # A fake/legacy client without it falls back to safe recomputation.
        if collection_exists and hasattr(client, 'retrieve'):
            for start in range(0,len(ids),128):
                stored = client.retrieve(collection_name=collection, ids=ids[start:start+128],
                    with_payload=True, with_vectors=False)
                for entry in stored:
                    stored_id = str(entry.id)
                    if stored_id not in id_positions:
                        raise ValueError('Qdrant returned a non-session source identity')
                    pos = id_positions[stored_id]
                    chunk = eligible[pos]
                    payload = getattr(entry,'payload',None) or {}
                    if any((payload.get(key) != value) for key,value in (
                        ('equipment_scope',scope), ('model_revision',revision),
                        ('source_identity',keys[pos]),('source_sha256',chunk.source_sha256),
                        ('review_status',chunk.review_status),
                        ('content_safety_status',chunk.content_safety_status))):
                        raise ValueError('Qdrant persisted source provenance mismatch')
                    reused.add(stored_id)
        pending = set()
        missing_positions = []
        for i, point_id in enumerate(ids):
            if point_id not in reused and point_id not in pending:
                missing_positions.append(i)
                pending.add(point_id)
        encoded = (np.asarray(model.encode([eligible[i].text[:2500] for i in missing_positions],
            normalize_embeddings=True), dtype='<f4') if missing_positions else np.empty((0, 0), dtype='<f4'))
        dimension = len(encoded[0]) if len(encoded) else existing_dimensions or 0
        for vector in encoded:
            _encode(vector)
            if len(vector) != dimension:
                raise ValueError('embedding dimension mismatch')
        if collection_exists and dimension and existing_dimensions != dimension:
            # Never silently replace a collection when a model/config changes.
            raise ValueError('existing Qdrant collection vector dimensions mismatch')
        if not collection_exists and dimension:
            client.create_collection(collection_name=collection,
                vectors_config=models.VectorParams(size=dimension,distance=models.Distance.COSINE),
                on_disk_payload=True)
        batch=[]
        for pos,vec in zip(missing_positions,encoded):
            c=eligible[pos]
            batch.append(models.PointStruct(id=ids[pos],vector=vec.tolist(),payload={
                'equipment_scope':scope,'model_revision':revision,
                'source_identity':keys[pos],'source_sha256':c.source_sha256,
                'review_status':c.review_status,'content_safety_status':c.content_safety_status}))
        for pos in range(0,len(batch),128):
            client.upsert(collection_name=collection,points=batch[pos:pos+128],wait=True)
        id_to_chunk=dict(zip(ids,[c.chunk_id for c in eligible]))
        session_vectors=(np.asarray(model.encode([c.text[:2500] for c in ephemeral],
            normalize_embeddings=True),dtype='<f4') if ephemeral else np.empty((0,0),dtype='<f4'))
        for item in session_vectors:
            _encode(item)
            if dimension and len(item)!=dimension:
                raise ValueError('session and indexed embedding dimensions disagree')
        output=[]
        for q in queries:
            v=np.asarray(model.encode([q[:900]],normalize_embeddings=True)[0],dtype='<f4')
            _encode(v)
            if dimension and len(v)!=dimension: raise ValueError('query embedding dimension mismatch')
            if len(session_vectors) and len(v)!=len(session_vectors[0]):
                raise ValueError('query and session embedding dimension mismatch')
            server_rank=[]
            if ids:
                # HasIdCondition prevents retrieving *any* vector outside the
                # current job's scope/revision/approved source allowlist.
                restriction=models.Filter(must=[
                    models.FieldCondition(key='equipment_scope',match=models.MatchValue(value=scope)),
                    models.FieldCondition(key='model_revision',match=models.MatchValue(value=revision)),
                    models.HasIdCondition(has_id=ids)])
                resp=client.query_points(collection_name=collection,query=v.tolist(),
                    query_filter=restriction,limit=16,score_threshold=.25,with_payload=False)
                server_rank=[id_to_chunk[str(p.id)] for p in resp.points if str(p.id) in id_to_chunk]
            ephemeral_scores=[(float(v @ vec),c.chunk_id) for c,vec in zip(ephemeral,session_vectors)]
            local_rank=[key for score,key in sorted(ephemeral_scores,key=lambda z:(-z[0],z[1]))
                        if np.isfinite(score) and score>=.25][:16]
            # RRF avoids inventing cross-backend similarity calibration.
            fusion={}
            for ordered in (server_rank,local_rank):
                for pos,key in enumerate(ordered,1):
                    fusion[key]=fusion.get(key,0.) + 1./(60+pos)
            output.append(sorted(fusion,key=lambda key:(-fusion[key],key))[:16])
        return output, {'status':'SUCCEEDED','persistence':'QDRANT_ANN_WITH_EPHEMERAL_SESSION_SEARCH',
            'model_sha256':revision,'indexed_reviewed_fragments':len(eligible),
            'source_vectors_reused':len(reused),'source_vectors_created':len(batch),
            'ephemeral_uploaded_vectors':len(ephemeral),
            'retrieval_filter':'SCOPE_REVISION_AND_CURRENT_TASK_EXPERT_APPROVED_SOURCE_IDS',
            'network_download_allowed':False,'claim_boundary':'UPLOAD_EMBEDDINGS_NEVER_SENT_TO_QDRANT; SIMILARITY_NOT_CAUSAL_PROOF'}
    finally:
        if own and hasattr(client,'close'):
            client.close()
