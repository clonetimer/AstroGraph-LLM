"""Browser-workbench API integration, permissions, provenance and lifecycle gates."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from astrograph.api import create_app

SCOPE='LAB_TEST_ADCS_V2'
TEXT='ADCS_RW_FRICTION_INCREASE 关联 反作用飞轮轴承磨损。\n反作用飞轮轴承磨损会导致飞轮电流升高。\n'
EXTRACT={'entities':[
    {'local_id':'f','entity_type':'FAULT_MODE','name':'ADCS_RW_FRICTION_INCREASE'},
    {'local_id':'m','entity_type':'FAILURE_MECHANISM','name':'反作用飞轮轴承磨损'},
    {'local_id':'s','entity_type':'SYMPTOM','name':'飞轮电流升高'},
],'relations':[
    {'source_local_id':'f','relation_type':'ASSOCIATED_WITH','target_local_id':'m',
     'evidence_quote':'ADCS_RW_FRICTION_INCREASE 关联 反作用飞轮轴承磨损。'},
    {'source_local_id':'m','relation_type':'MANIFESTS_AS','target_local_id':'s',
     'evidence_quote':'反作用飞轮轴承磨损会导致飞轮电流升高。'},
]}


class MockBackend:
    model_id='mock-workbench-graph-model'
    def generate_structured(self, *, agent_id, output_model, payload):
        assert agent_id=='material-reference-graph-extractor'
        assert payload['ontology_allowlist']
        return output_model.model_validate(EXTRACT)


@pytest.fixture
def system(tmp_path,monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT',str(tmp_path/'platform'))
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEW_TOKEN','test-only-material-review-token-012345')
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEWER','expert')
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEWERS','expert')
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_RELEASE_SIGNING_KEY','test-only-signing-material-graph-key-012345')
    monkeypatch.delenv('ASTROGRAPH_MATERIAL_GRAPH_PUBLISHERS',raising=False)
    monkeypatch.delenv('ASTROGRAPH_MATERIAL_GRAPH_NEO4J_WRITE_ENABLED',raising=False)
    monkeypatch.setattr('astrograph.integrations.material_graph_builder._backend_from_environment',lambda:MockBackend())
    return TestClient(create_app(),base_url='http://localhost'),tmp_path,monkeypatch


def login(client):
    result=client.post('/v1/platform/material-knowledge/review-session',json={
        'credential':'test-only-material-review-token-012345'})
    assert result.status_code==200,result.text
    assert 'httponly' in result.headers['set-cookie'].lower()
    assert 'samesite=strict' in result.headers['set-cookie'].lower()
    assert 'test-only-material-review-token' not in result.headers['set-cookie']
    return {'x-material-csrf':result.json()['csrf']}


def build(client,headers):
    return client.post('/v1/platform/material-knowledge/graph/build',headers=headers,
        data={'equipment_scope':SCOPE,'max_chunks':'4'},
        files=[('reference_files',('manual.md',TEXT.encode('utf-8'),'text/markdown'))])


def test_workbench_uses_ephemeral_cookie_and_csrf_not_client_review_token(system):
    client,_root,_monkeypatch=system
    endpoint='/v1/platform/material-knowledge'
    assert client.get(endpoint+'/review-session').json()['authenticated'] is False
    assert client.get(endpoint+'/pending').status_code==403
    assert client.get(endpoint+'/graph/status').status_code==403
    assert build(client,{}).status_code==403
    assert client.post(endpoint+'/review-session',json={'credential':'bad'}).status_code==403
    csrf=login(client)
    assert client.get(endpoint+'/review-session').json()['authenticated'] is True
    assert not client.get(endpoint+'/review-session').json()['publisher']
    assert build(client,{}).status_code==403   # Authenticated cookie is insufficient for writes.
    assert client.delete(endpoint+'/review-session').status_code==403
    assert client.delete(endpoint+'/review-session',headers=csrf).status_code==200
    assert client.get(endpoint+'/pending').status_code==403


def test_graph_build_review_export_and_provenance_download(system):
    client,root,monkeypatch=system
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_PUBLISHERS','expert')
    csrf=login(client)
    result=build(client,csrf)
    assert result.status_code==200,result.text
    body=result.json()
    assert body['proposal_count']==2 and body['staging_status']=='PENDING_EXPERT_REVIEW'
    assert body['source_group_id'].startswith('JOB-')
    assert body['active_kg_modified'] is False
    assert body['source_receipts'][0]['sha256']
    location=f"/v1/platform/material-knowledge/graph/sources/{body['source_group_id']}/manual.md"
    source=client.get(location)
    assert source.status_code==200 and source.content==TEXT.encode('utf-8')
    assert 'no-store' in source.headers['cache-control']
    pending=client.get('/v1/platform/material-knowledge/pending').json()
    assert len(pending['graph_proposals'])==2
    for i,item in enumerate(pending['graph_proposals']):
        assert item['evidence_quote'] in TEXT
        assert item['provenance']['source_sha256']==body['source_receipts'][0]['sha256']
        response=client.post(f"/v1/platform/material-knowledge/graph/{item['proposal_id']}/decision",headers=csrf,
            json={'decision':'CONFIRM','truth_source_ref':'Independent manual revision 5',
                  'equipment_scope':SCOPE,'expected_snapshot':i})
        assert response.status_code==200,response.text
        assert response.json()['snapshot']==i+1
    reviewed=client.get('/v1/platform/material-knowledge/graph/reviewed',params={'equipment_scope':SCOPE}).json()
    assert reviewed['snapshot']==2 and reviewed['total']==2
    status=client.get('/v1/platform/material-knowledge/graph/status',params={'equipment_scope':SCOPE}).json()
    assert status['active_approved_relations']==2
    assert not (root/'platform'/'material_knowledge'/'graph_releases').exists()  # no auto publish
    release=client.post('/v1/platform/material-knowledge/graph/export',headers=csrf,
        json={'equipment_scope':SCOPE,'snapshot':2})
    assert release.status_code==200,release.text
    assert release.json()['signature_hmac_sha256']
    from astrograph.integrations.material_graph_release import verify_graph_release
    directory=root/'platform'/'material_knowledge'/'graph_releases'/release.json()['release_id']
    assert verify_graph_release(directory,require_signature=True)[0]['edge_count']==2
    assert client.post('/v1/platform/material-knowledge/graph/publish',headers=csrf,json={
        'equipment_scope':SCOPE,'snapshot':2,'confirm_release_id':release.json()['release_id'],
        'confirm_write':True}).status_code==422  # host Neo4j writer is intentionally disabled
    assert client.get(location).status_code==200
    assert client.get(location.replace('manual.md','../manual.md')).status_code!=200


def test_publication_role_gate_and_snapshot_cas_withdrawal(system):
    client,_root,monkeypatch=system
    csrf=login(client)
    assert client.post('/v1/platform/material-knowledge/graph/export',headers=csrf,
        json={'equipment_scope':SCOPE}).status_code==403
    body=build(client,csrf).json()
    proposal=client.get('/v1/platform/material-knowledge/pending').json()['graph_proposals'][0]
    route='/v1/platform/material-knowledge/graph/'+proposal['proposal_id']
    wrong=client.post(route+'/decision',headers=csrf,json={
        'decision':'CONFIRM','truth_source_ref':'Quality reference',
        'equipment_scope':SCOPE,'expected_snapshot':100})
    assert wrong.status_code==422
    good=client.post(route+'/decision',headers=csrf,json={
        'decision':'CONFIRM','truth_source_ref':'Quality reference',
        'equipment_scope':SCOPE,'expected_snapshot':0})
    assert good.status_code==200
    assert client.post(route+'/retract',json={'reason':'Independent correction 2026',
        'truth_source_ref':'Quality reference','expected_snapshot':1}).status_code==403
    revoke=client.post(route+'/retract',headers=csrf,json={
        'reason':'Independent correction 2026','truth_source_ref':'Quality reference','expected_snapshot':1})
    assert revoke.status_code==200,revoke.text
    assert revoke.json()['snapshot']==2
    latest=client.get('/v1/platform/material-knowledge/graph/reviewed',params={'equipment_scope':SCOPE}).json()
    assert latest['total']==0 and len(latest['retractions'])==1


def test_build_fails_closed_for_unsafe_and_non_reference_uploads(system):
    client,_root,_monkeypatch=system
    csrf=login(client)
    url='/v1/platform/material-knowledge/graph/build'
    def send(files,scope=SCOPE):
        return client.post(url,headers=csrf,data={'equipment_scope':scope,'max_chunks':'2'},files=files)
    assert send([('reference_files',('../bad.md',b'a','text/plain'))]).status_code==422
    assert send([('reference_files',('raw.csv',b'1,2','text/csv'))]).status_code==422
    assert send([('reference_files',('empty.md',b'','text/markdown'))]).status_code==422
    assert send([('reference_files',('same.md',b'x','text/plain')),
                 ('reference_files',('SAME.MD',b'y','text/plain'))]).status_code==422
    assert send([('reference_files',('manual.md',TEXT.encode(),'text/plain'))],scope='UNKNOWN').status_code==422
    assert send([('reference_files',('too_large.md',b'z'*(16*1024*1024+1),'text/plain'))]).status_code==422


def test_workbench_disabled_default_and_remote_http_login_rejected(tmp_path,monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT',str(tmp_path/'jobs'))
    for name in ('ASTROGRAPH_MATERIAL_REVIEW_TOKEN','ASTROGRAPH_MATERIAL_REVIEWERS','ASTROGRAPH_MATERIAL_REVIEWER'):
        monkeypatch.delenv(name,raising=False)
    client=TestClient(create_app())
    assert client.get('/v1/platform/material-knowledge/review-session').json()['configured'] is False
    assert client.get('/v1/platform/material-knowledge/pending').status_code==503
    assert client.post('/v1/platform/material-knowledge/review-session',json={'credential':'xxx'}).status_code==503
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEW_TOKEN','s'*36)
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEWER','expert')
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEWERS','expert')
    remote=TestClient(create_app(),base_url='http://astrograph.example.net')
    assert remote.post('/v1/platform/material-knowledge/review-session',json={'credential':'s'*36}).status_code==403


def test_graph_write_gates_run_before_body_parse_and_stale_publication(system):
    client,_root,monkeypatch=system
    endpoint='/v1/platform/material-knowledge'
    # A malformed/non-multipart body from an unauthenticated sender is rejected
    # by the middleware, before FastAPI tries to parse graph references.
    unauth=client.post(endpoint+'/graph/build',content=b'invalid-multipart-content',
                       headers={'content-type':'application/octet-stream'})
    assert unauth.status_code==403
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_PUBLISHERS','expert')
    csrf=login(client)
    assert build(client,csrf).status_code==200
    claim=client.get(endpoint+'/pending').json()['graph_proposals'][0]
    approved=client.post(endpoint+'/graph/'+claim['proposal_id']+'/decision',headers=csrf,json={
        'decision':'CONFIRM','truth_source_ref':'Independent verified service note',
        'equipment_scope':SCOPE,'expected_snapshot':0})
    assert approved.status_code==200
    stale=client.post(endpoint+'/graph/publish',headers=csrf,json={
        'equipment_scope':SCOPE,'snapshot':0,
        'confirm_release_id':f'{SCOPE}-SNAPSHOT-00000000','confirm_write':True})
    assert stale.status_code==422 and 'stale' in stale.json()['detail']
