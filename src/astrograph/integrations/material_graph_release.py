"""Validate/export reviewed material graphs and optionally import into isolated Neo4j labels.

Publication is a host-admin action, not an agent tool and not a material-review
job side effect. The importer never uses or mutates legacy AstroGraph Active KG
labels, and it cannot author telemetry evidence facts or execute controls.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

from astrograph.integrations.material_graph_builder import ALLOWED_TRIPLES, GRAPH_PROPOSAL_SCHEMA, causal_quotation_defects
from astrograph.integrations.material_applicability import assess_contexts
from astrograph.integrations.material_temporal import normalize_reviewed_temporal, assess_temporal_causal_path


class MaterialGraphReleaseError(ValueError):
    pass


def verify_graph_release(path: str | Path, *, require_signature: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    root=Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise MaterialGraphReleaseError('graph release must be a directory')
    manifest=json.loads((root/'release.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version')!='astrograph-material-graph-release.v1':
        raise MaterialGraphReleaseError('not a material graph release')
    signature=manifest.get('signature_hmac_sha256')
    if signature is not None:
        key=os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_RELEASE_SIGNING_KEY','')
        if not key or len(key)<32 or manifest.get('signature_algorithm')!='HMAC-SHA256':
            raise MaterialGraphReleaseError('material graph signature key or algorithm unavailable')
        unsigned={k:v for k,v in manifest.items() if k!='signature_hmac_sha256'}
        encoded=json.dumps(unsigned,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
        expected=hmac.new(key.encode('utf-8'),encoded.encode('utf-8'),hashlib.sha256).hexdigest()
        if not isinstance(signature,str) or not hmac.compare_digest(signature,expected):
            raise MaterialGraphReleaseError('material graph release signature mismatch')
    elif require_signature:
        raise MaterialGraphReleaseError('Neo4j publication requires a signed material graph release')
    expected_files={'nodes.csv','edges.csv','graph.json'}
    if set(manifest.get('files_sha256',{}))!=expected_files:
        raise MaterialGraphReleaseError('unexpected graph release file manifest')
    for name,checksum in manifest['files_sha256'].items():
        if not isinstance(checksum,str) or len(checksum)!=64:
            raise MaterialGraphReleaseError('invalid graph release hash')
        real=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if real!=checksum:
            raise MaterialGraphReleaseError('graph release checksum mismatch: '+name)
    graph=json.loads((root/'graph.json').read_text(encoding='utf-8'))
    if (graph.get('schema_version')!='astrograph-material-reviewed-graph.v1' or
        graph.get('snapshot')!=manifest['snapshot'] or
        graph.get('equipment_scope')!=manifest['equipment_scope'] or
        len(graph.get('nodes',[]))!=manifest['node_count'] or
        len(graph.get('relations',[]))!=manifest['edge_count'] or
        len(graph.get('retractions',[]))!=manifest.get('retracted_claim_count',0)):
        raise MaterialGraphReleaseError('graph release manifest/content mismatch')
    indexed={n['entity_id']:n for n in graph['nodes']}
    if len(indexed)!=len(graph['nodes']):
        raise MaterialGraphReleaseError('duplicate node IDs')
    active_proposal_ids={edge['proposal_id'] for edge in graph['relations']}
    tombstones=graph.get('retractions',[])
    if (len(active_proposal_ids)!=len(graph['relations']) or
        len({entry['proposal_id'] for entry in tombstones})!=len(tombstones) or
        any(entry['proposal_id'] in active_proposal_ids or
            not 1<=entry['version']<=manifest['snapshot'] or
            not entry.get('reviewer') or not entry.get('truth_source_ref') or
            not entry.get('reason') for entry in tombstones)):
        raise MaterialGraphReleaseError('graph release has duplicate or invalid revocation records')
    for relation in graph['relations']:
        if relation.get('reviewed_temporal') is not None:
            try:
                if (relation.get('relation_type') != 'CAUSES' or relation.get('reviewed_applicability') is None):
                    raise ValueError('invalid temporal causality')
                normalize_reviewed_temporal(relation['reviewed_temporal'], equipment_scope=manifest['equipment_scope'])
                if assess_temporal_causal_path([relation], equipment_scope=manifest['equipment_scope'])['conflict_reasons']:
                    raise ValueError('temporal context exceeds approved scope')
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                raise MaterialGraphReleaseError('invalid approved causal temporal metadata') from exc
        if (relation.get('schema_version')!=GRAPH_PROPOSAL_SCHEMA or
            relation.get('review_status')!='INDEPENDENTLY_REVIEWED_REFERENCE_RELATION' or
            relation.get('equipment_scope')!=manifest['equipment_scope'] or
            not 1<=relation.get('snapshot',0)<=manifest['snapshot'] or
            not relation.get('truth_source_ref') or not relation.get('reviewer') or
            not relation.get('provenance',{}).get('source_locators') or
            (relation['source']['entity_type'],relation['relation_type'],
             relation['target']['entity_type']) not in ALLOWED_TRIPLES or
            relation.get('polarity') not in {'AFFIRMATIVE','NEGATED'} or
            causal_quotation_defects(relation) or
            assess_contexts([relation], equipment_scope=manifest['equipment_scope'])['conflict_reasons'] or
            not relation.get('evidence_quote') or
            indexed.get(relation['source']['entity_id']) is None or
            indexed.get(relation['target']['entity_id']) is None):
            raise MaterialGraphReleaseError('graph release contains an unapproved or malformed relation')
    return manifest,graph


# Parameterized Cypher is static. We represent relationships as claim nodes with
# typed properties: no dynamic Cypher labels/types and no legacy graph labels.
_CLAIM_CYPHER = '''
MERGE (s:AstroMaterialEntity {entity_id: $source_id, equipment_scope: $scope})
ON CREATE SET s.entity_type=$source_type, s.name=$source_name
MERGE (t:AstroMaterialEntity {entity_id: $target_id, equipment_scope: $scope})
ON CREATE SET t.entity_type=$target_type, t.name=$target_name
MERGE (c:AstroMaterialClaim {proposal_id: $proposal_id})
ON CREATE SET c.relation_type=$relation_type, c.polarity=$polarity,
              c.evidence_quote=$quote, c.source_sha256=$source_sha256,
              c.locators=$locators, c.review_status=$review_status,
              c.equipment_scope=$scope, c.snapshot=$snapshot,
              c.truth_source_ref=$truth_source_ref, c.release_id=$release_id,
              c.claim_modality=$claim_modality, c.condition_quote=$condition_quote,
              c.reviewer=$reviewer, c.reviewed_applicability_json=$reviewed_applicability_json,
              c.reviewed_temporal_json=$reviewed_temporal_json
SET c.current_release_active=true, c.release_id=$release_id
MERGE (s)-[:ASTRO_MATERIAL_SUBJECT]->(c)
MERGE (c)-[:ASTRO_MATERIAL_OBJECT]->(t)
'''

# The release head is locked first, before any claim mutations. Publishing an
# older signed snapshot must not re-enable withdrawn or superseded knowledge.
_LOCK_HEAD_CYPHER = '''
MERGE (h:AstroMaterialReleaseHead {equipment_scope:$scope})
ON CREATE SET h.snapshot=-1, h.release_id=''
SET h.lock_nonce=$release_id
RETURN h.snapshot AS snapshot, h.release_id AS release_id
'''
_RETIRE_STALE_CYPHER = '''
MATCH (c:AstroMaterialClaim {equipment_scope:$scope})
WHERE c.release_id <> $release_id OR c.release_id IS NULL
SET c.current_release_active=false, c.last_release_checked=$release_id
'''
_COMMIT_HEAD_CYPHER = '''
MATCH (h:AstroMaterialReleaseHead {equipment_scope:$scope})
SET h.snapshot=$snapshot, h.release_id=$release_id
'''


def _publish_batch(tx: Any, release: dict[str, Any], graph: dict[str, Any]):
    head=tx.run(_LOCK_HEAD_CYPHER,scope=release['equipment_scope'],
                release_id=release['release_id']).single()
    if head is None:
        raise MaterialGraphReleaseError('Neo4j material graph head lock unavailable')
    previous=int(head['snapshot'])
    if (previous>release['snapshot'] or
        (previous==release['snapshot'] and head['release_id']!=release['release_id'])):
        raise MaterialGraphReleaseError('cannot publish older or conflicting material graph snapshot')
    count=0
    for item in graph['relations']:
        tx.run(_CLAIM_CYPHER,
            source_id=item['source']['entity_id'],source_type=item['source']['entity_type'],
            source_name=item['source']['name'],target_id=item['target']['entity_id'],
            target_type=item['target']['entity_type'],target_name=item['target']['name'],
            proposal_id=item['proposal_id'],relation_type=item['relation_type'],
            polarity=item['polarity'],quote=item['evidence_quote'],
            source_sha256=item['provenance']['source_sha256'],
            locators=item['provenance']['source_locators'],
            review_status=item['review_status'],
            claim_modality=item.get('claim_modality',''),
            condition_quote=item.get('condition_quote') or '',reviewer=item['reviewer'],
            reviewed_applicability_json=json.dumps(item.get('reviewed_applicability'),
                                               sort_keys=True,ensure_ascii=False),
            reviewed_temporal_json=json.dumps(item.get('reviewed_temporal'),
                                               sort_keys=True,ensure_ascii=False),
            scope=release['equipment_scope'],
            snapshot=item['snapshot'],truth_source_ref=item['truth_source_ref'],
            release_id=release['release_id'])
        count+=1
    # Claims absent from the *new* snapshot are no longer active in this
    # material-only namespace. They remain present for historical audit.
    tx.run(_RETIRE_STALE_CYPHER,scope=release['equipment_scope'],
           release_id=release['release_id'])
    tx.run(_COMMIT_HEAD_CYPHER,scope=release['equipment_scope'],
           snapshot=release['snapshot'],release_id=release['release_id'])
    return count


def publish_release_neo4j(path: str | Path, *, confirm_write: bool,
                          driver: Any | None = None) -> dict[str, Any]:
    """Privileged explicit import into a separate material graph namespace.

    This is NOT executed by any normal task. Production deployments should use
    a dedicated Neo4j database and a role restricted to AstroMaterial* labels.
    """
    release,graph=verify_graph_release(path,require_signature=confirm_write)
    if not confirm_write:
        return {**release,'status':'VERIFIED_DRY_RUN','neo4j_write_performed':False}
    if os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_NEO4J_WRITE_ENABLED','')!='1':
        raise PermissionError('explicit Neo4j material graph writer gate is disabled')
    database=os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_NEO4J_DATABASE','').strip()
    if not database or database.lower() in {'system','neo4j'}:
        raise PermissionError('configure a dedicated non-default material graph Neo4j database')
    close=False
    if driver is None:
        uri=os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_NEO4J_URI','')
        user=os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_NEO4J_USER','')
        password=os.environ.get('ASTROGRAPH_MATERIAL_GRAPH_NEO4J_PASSWORD','')
        if not uri.startswith(('bolt://','bolt+s://','neo4j+s://','neo4j://')) or not user or not password:
            raise PermissionError('dedicated Neo4j writer URI and credentials required')
        from neo4j import GraphDatabase
        driver=GraphDatabase.driver(uri,auth=(user,password),connection_timeout=5.0)
        close=True
    try:
        with driver.session(database=database) as session:
            count=session.execute_write(_publish_batch,release,graph)
    finally:
        if close:
            driver.close()
    return {'status':'IMPORTED_TO_MATERIAL_NAMESPACE','release_id':release['release_id'],
            'claims_imported':count,'neo4j_database':database,
            'inactive_claims_synchronized':True,
            'legacy_active_kg_modified':False}


__all__=['MaterialGraphReleaseError','verify_graph_release','publish_release_neo4j']
