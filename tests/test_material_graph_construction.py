"""Acceptance: LLM-sourced typed propositions, independent review, graph publication."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from astrograph.api import create_app
from astrograph.integrations.material_case_memory import MaterialCaseMemory, MaterialMemoryConflict
from astrograph.integrations.material_graph_builder import (
    GraphChunkExtraction, _balanced_chunks, build_material_graph, validate_fragment,
)
from astrograph.integrations.material_graph_release import (
    MaterialGraphReleaseError, publish_release_neo4j, verify_graph_release,
)
from astrograph.integrations.upload_report import UploadReportJobStore


JOB1='JOB-'+'A'*24
JOB2='JOB-'+'B'*24
SCOPE='SAT-A_ADCS_V2'
TEXT=('ADCS_RW_FRICTION_INCREASE 关联 反作用飞轮轴承磨损。\n'
      '反作用飞轮轴承磨损会导致飞轮电流升高。\n'
      'ADCS_RW_FRICTION_INCREASE 应检查飞轮电流升高。\n')


@dataclass
class Chunk:
    chunk_id: str='DOC-1'
    source_sha256: str='e'*64
    filename: str='reference.md'
    source_locators: tuple[str,...]=('line:1-3',)
    section_path: tuple[str,...]=('异常分析',)
    text: str=TEXT


def payload(*, bad_quote: bool = False, polarity: str = 'AFFIRMATIVE'):
    return {
        'entities': [
            {'local_id':'f','entity_type':'FAULT_MODE','name':'ADCS_RW_FRICTION_INCREASE','aliases':[]},
            {'local_id':'m','entity_type':'FAILURE_MECHANISM','name':'反作用飞轮轴承磨损','aliases':[]},
            {'local_id':'s','entity_type':'SYMPTOM','name':'飞轮电流升高','aliases':[]},
        ],
        'relations': [
            {'source_local_id':'f','relation_type':'ASSOCIATED_WITH','target_local_id':'m',
             'evidence_quote':('这是不存在的模型虚构原文引句' if bad_quote else 'ADCS_RW_FRICTION_INCREASE 关联 反作用飞轮轴承磨损。'),
             'polarity':polarity},
            {'source_local_id':'m','relation_type':'MANIFESTS_AS','target_local_id':'s',
             'evidence_quote':'反作用飞轮轴承磨损会导致飞轮电流升高。','polarity':'AFFIRMATIVE'},
        ],
    }


class FakeExtractor:
    model_id='mock-structured-extractor'
    def __init__(self, graph=None):
        self.graph=graph or payload()
        self.calls=0
    def generate_structured(self, *, agent_id, output_model, payload):
        assert agent_id=='material-reference-graph-extractor'
        assert 'ontology_allowlist' in payload
        self.calls+=1
        return output_model.model_validate(self.graph)


def memory(tmp_path):
    return MaterialCaseMemory(tmp_path/'material_knowledge'/'cases.sqlite3',reviewers=('engineer',))


def build(tmp_path, *, polarity='AFFIRMATIVE'):
    mem=memory(tmp_path)
    graph=payload(polarity=polarity)
    result=build_material_graph(chunks=[Chunk()],source_group_id=JOB1,memory=mem,
                                enabled=True,backend=FakeExtractor(graph))
    assert result['status']=='SUCCEEDED'
    assert result['proposal_count']==2
    assert result['staging_status']=='PENDING_EXPERT_REVIEW'
    return mem,result


def test_typed_source_anchoring_rejects_hallucinations_and_mismatched_schema():
    good,errors=validate_fragment(GraphChunkExtraction.model_validate(payload()),Chunk(),source_group_id=JOB1)
    assert not errors and len(good)==2
    assert all(x['provenance']['source_sha256']=='e'*64 for x in good)
    assert all(x['evidence_status']=='MODEL_EXTRACTED_REFERENCE_CLAIM_UNVERIFIED' for x in good)
    graph=payload(bad_quote=True)
    # Direct model-created properties do not control the source filename/hash.
    graph['relations'].append({'source_local_id':'f','relation_type':'MITIGATED_BY',
        'target_local_id':'m','evidence_quote':'ADCS_RW_FRICTION_INCREASE 关联 反作用飞轮轴承磨损。',
        'polarity':'AFFIRMATIVE'})
    valid,bad=validate_fragment(GraphChunkExtraction.model_validate(graph),Chunk(),source_group_id=JOB1)
    assert len(valid)==1
    assert {x['reason'] for x in bad}=={'QUOTE_NOT_FOUND_IN_SOURCE','ONTOLOGY_TRIPLE_NOT_ALLOWED'}
    assert all(x['source']['entity_type']!='COMPONENT' for x in valid)


def test_invalid_endpoint_and_duplicate_local_ids_are_not_accepted():
    graph=payload()
    graph['relations'][0]['source_local_id']='not-present'
    valid,bad=validate_fragment(GraphChunkExtraction.model_validate(graph),Chunk(),source_group_id=JOB1)
    assert len(valid)==1 and bad[0]['reason']=='UNKNOWN_ENDPOINT'
    graph=payload()
    graph['entities'][1]['local_id']='f'
    valid,bad=validate_fragment(GraphChunkExtraction.model_validate(graph),Chunk(),source_group_id=JOB1)
    assert not valid and bad[0]['reason']=='DUPLICATE_LOCAL_ID'


def test_unconfigured_model_never_makes_up_graph(tmp_path,monkeypatch):
    for var in ('ASTROGRAPH_MATERIAL_GRAPH_LLM_BASE_URL','ASTROGRAPH_MATERIAL_GRAPH_LLM_MODEL',
                'ASTROGRAPH_PLATFORM_LLM_BASE_URL','ASTROGRAPH_PLATFORM_LLM_MODEL'):
        monkeypatch.delenv(var,raising=False)
    mem=memory(tmp_path)
    result=build_material_graph(chunks=[Chunk()],source_group_id=JOB1,memory=mem,enabled=True)
    assert result['status']=='MODEL_NOT_CONFIGURED' and result['proposals']==[]
    assert not mem.pending_graph()
    disabled=build_material_graph(chunks=[Chunk()],source_group_id=JOB1,memory=mem,enabled=False)
    assert disabled['status']=='DISABLED'


def test_graph_chunk_selection_gives_multiple_references_fair_coverage(tmp_path):
    from dataclasses import replace
    docs=[replace(Chunk(),source_sha256='a'*64,chunk_id=f'A-{n}') for n in range(9)]
    docs += [replace(Chunk(),source_sha256='b'*64,chunk_id='B-0')]
    assert [x.chunk_id for x in _balanced_chunks(docs[:])[:3]]==['A-0','B-0','A-1']
    result=build_material_graph(chunks=docs,source_group_id=JOB1,memory=None,
                                enabled=True,backend=FakeExtractor(),max_chunks=2)
    assert result['source_documents_total']==2
    assert result['source_documents_considered']==2
    assert result['chunks_processed']==2
    assert result['chunks_truncated']==8


def test_scoped_confirmed_two_hop_graph_and_snapshot_isolation(tmp_path):
    mem,result=build(tmp_path)
    proposals=mem.pending_graph()
    assert len(proposals)==2
    for item in proposals:
        assert item['provenance']['source_locators']
        assert item['evidence_quote'] in Chunk().text
    assert mem.retrieve_graph(fault_id='ADCS_RW_FRICTION_INCREASE',fault_name='N/A',
        equipment_scope=SCOPE,exclude_source_group_id=JOB2,snapshot=0)['paths']==[]
    with pytest.raises(PermissionError):
        mem.decide_graph(proposal_id=proposals[0]['proposal_id'],reviewer='unknown',
            decision='CONFIRM',truth_source_ref='Manual rev2',equipment_scope=SCOPE)
    with pytest.raises(ValueError):
        mem.decide_graph(proposal_id=proposals[0]['proposal_id'],reviewer='engineer',
            decision='CONFIRM',truth_source_ref='Manual rev2',equipment_scope='UNKNOWN')
    with pytest.raises(MaterialMemoryConflict):
        mem.decide_graph(proposal_id=proposals[0]['proposal_id'],reviewer='engineer',
            decision='CONFIRM',truth_source_ref='Manual rev2',equipment_scope=SCOPE,
            expected_snapshot=999)
    for i,item in enumerate(proposals):
        receipt=mem.decide_graph(proposal_id=item['proposal_id'],reviewer='engineer',
            decision='CONFIRM',truth_source_ref='Independent engineer assessment #99',
            equipment_scope=SCOPE,expected_snapshot=i)
        assert receipt['snapshot']==i+1
    hits=mem.retrieve_graph(fault_id='ADCS_RW_FRICTION_INCREASE',fault_name='N/A',
        equipment_scope=SCOPE,exclude_source_group_id=JOB2,snapshot=2)
    assert hits['status']=='SUCCEEDED'
    assert {x['relation_type'] for x in hits['paths']}=={'ASSOCIATED_WITH','MANIFESTS_AS'}
    assert not any(x.get('current_root_cause_proven') for x in hits['paths'])
    assert not mem.retrieve_graph(fault_id='ADCS_RW_FRICTION_INCREASE',fault_name='N/A',
        equipment_scope='SAT-B_ADCS_V1',exclude_source_group_id=JOB2,snapshot=2)['paths']
    assert not mem.retrieve_graph(fault_id='ADCS_RW_FRICTION_INCREASE',fault_name='N/A',
        equipment_scope=SCOPE,exclude_source_group_id=JOB1,snapshot=2)['paths']
    assert not mem.retrieve_graph(fault_id='ADCS_RW_FRICTION_INCREASE',fault_name='N/A',
        equipment_scope=SCOPE,exclude_source_group_id=JOB2,snapshot=0)['paths']
    with pytest.raises(MaterialMemoryConflict):
        mem.decide_graph(proposal_id=proposals[0]['proposal_id'],reviewer='engineer',
            decision='CONFIRM',truth_source_ref='New note',equipment_scope=SCOPE)


def test_rejected_and_negated_relations_do_not_turn_into_positive_support(tmp_path):
    mem,result=build(tmp_path,polarity='NEGATED')
    proposals=mem.pending_graph()
    neg=next(p for p in proposals if p['polarity']=='NEGATED')
    pos=next(p for p in proposals if p['polarity']=='AFFIRMATIVE')
    assert mem.decide_graph(proposal_id=pos['proposal_id'],reviewer='engineer',
         decision='REJECT',truth_source_ref='not supported')['status']=='REJECTED'
    assert mem.snapshot()==0
    mem.decide_graph(proposal_id=neg['proposal_id'],reviewer='engineer',
         decision='CONFIRM',truth_source_ref='Reviewed statement negation',equipment_scope=SCOPE)
    retrieved=mem.retrieve_graph(fault_id='ADCS_RW_FRICTION_INCREASE',fault_name='N/A',
        equipment_scope=SCOPE,exclude_source_group_id=JOB2,snapshot=1)
    assert len(retrieved['paths'])==1
    assert retrieved['paths'][0]['polarity']=='NEGATED'


def test_immutable_graph_release_checksum_and_no_automatic_neo4j_write(tmp_path,monkeypatch):
    mem,_=build(tmp_path)
    for idx,entry in enumerate(mem.pending_graph()):
        mem.decide_graph(proposal_id=entry['proposal_id'],reviewer='engineer',
                         decision='CONFIRM',truth_source_ref='Independent lab sheet 3',
                         equipment_scope=SCOPE,expected_snapshot=idx)
    result=mem.export_graph_release(equipment_scope=SCOPE,snapshot=2,output_root=tmp_path/'published')
    assert result['edge_count']==2 and result['node_count']==3
    path=Path(result['directory'])
    manifest,graph=verify_graph_release(path)
    assert graph['equipment_scope']==SCOPE and manifest['snapshot']==2
    assert publish_release_neo4j(path,confirm_write=False)['status']=='VERIFIED_DRY_RUN'
    with pytest.raises(MaterialGraphReleaseError,match='signed'):
        publish_release_neo4j(path,confirm_write=True)
    with pytest.raises(FileExistsError):
        mem.export_graph_release(equipment_scope=SCOPE,snapshot=2,output_root=tmp_path/'published')
    (path/'graph.json').write_text('{"tampered":true}',encoding='utf-8')
    with pytest.raises(MaterialGraphReleaseError):
        verify_graph_release(path)


def test_staged_claim_idempotence_and_content_integrity(tmp_path):
    mem,result=build(tmp_path)
    same=mem.stage_graph_proposals(source_group_id=JOB1,proposals=result['candidate_relations'])
    assert len(same)==2
    with pytest.raises(MaterialMemoryConflict):
        modified={**result['candidate_relations'][0],'evidence_quote':'altered'}
        mem.stage_graph_proposals(source_group_id=JOB1,proposals=[modified])


def test_signed_release_requires_same_key_and_isolated_neo4j_writer(tmp_path,monkeypatch):
    mem,_=build(tmp_path)
    for item in mem.pending_graph():
        mem.decide_graph(proposal_id=item['proposal_id'],reviewer='engineer',
                         decision='CONFIRM',truth_source_ref='Review 2026',equipment_scope=SCOPE)
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_RELEASE_SIGNING_KEY','A'*48)
    release=mem.export_graph_release(equipment_scope=SCOPE,snapshot=2,output_root=tmp_path/'signed')
    path=Path(release['directory'])
    manifest,graph=verify_graph_release(path,require_signature=True)
    assert manifest['signature_algorithm']=='HMAC-SHA256'
    assert len(manifest['signature_hmac_sha256'])==64
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_RELEASE_SIGNING_KEY','B'*48)
    with pytest.raises(MaterialGraphReleaseError,match='signature'):
        verify_graph_release(path)
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_RELEASE_SIGNING_KEY','A'*48)

    class FakeTransaction:
        def __init__(self): self.queries=[]
        def run(self,statement,**params):
            self.queries.append((statement,params))
            if 'RETURN h.snapshot AS snapshot' in statement:
                class Result:
                    def single(self): return {'snapshot':-1,'release_id':''}
                return Result()
    class FakeSession:
        def __init__(self): self.transaction=FakeTransaction()
        def __enter__(self): return self
        def __exit__(self,*_): pass
        def execute_write(self,callback,*args): return callback(self.transaction,*args)
    class FakeDriver:
        def __init__(self): self.database=None;self.connection=FakeSession()
        def session(self,*,database): self.database=database;return self.connection
    driver=FakeDriver()
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_NEO4J_WRITE_ENABLED','1')
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_NEO4J_DATABASE','astrograph_material_approved')
    result=publish_release_neo4j(path,confirm_write=True,driver=driver)
    assert result['status']=='IMPORTED_TO_MATERIAL_NAMESPACE'
    assert result['claims_imported']==2
    assert driver.database=='astrograph_material_approved'
    claims=[(sql,params) for sql,params in driver.connection.transaction.queries
            if 'MERGE (c:AstroMaterialClaim' in sql]
    assert len(claims)==2
    assert all('AstroMaterialEntity' in sql and 'EvidenceFact' not in sql
               and params['review_status']=='INDEPENDENTLY_REVIEWED_REFERENCE_RELATION'
               for sql,params in claims)
    assert any('SET c.current_release_active=false' in sql
               for sql,_ in driver.connection.transaction.queries)
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_NEO4J_DATABASE','neo4j')
    with pytest.raises(PermissionError):
        publish_release_neo4j(path,confirm_write=True,driver=driver)
    manifest['files_sha256']['graph.json']='0'*64
    (path/'release.json').write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf-8')
    with pytest.raises(MaterialGraphReleaseError,match='signature'):
        verify_graph_release(path)


def test_unsigned_release_may_be_inspected_but_never_published(tmp_path,monkeypatch):
    monkeypatch.delenv('ASTROGRAPH_MATERIAL_GRAPH_RELEASE_SIGNING_KEY',raising=False)
    mem,_=build(tmp_path)
    for item in mem.pending_graph():
        mem.decide_graph(proposal_id=item['proposal_id'],reviewer='engineer',
                         decision='CONFIRM',truth_source_ref='Review 2026',equipment_scope=SCOPE)
    release=mem.export_graph_release(equipment_scope=SCOPE,snapshot=2,output_root=tmp_path/'unsigned')
    path=Path(release['directory'])
    assert verify_graph_release(path)[0]['edge_count']==2
    assert publish_release_neo4j(path,confirm_write=False)['status']=='VERIFIED_DRY_RUN'
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_NEO4J_WRITE_ENABLED','1')
    with pytest.raises(MaterialGraphReleaseError,match='signed'):
        publish_release_neo4j(path,confirm_write=True)


def test_graph_review_api_is_privileged_and_pending_routes_include_graph(tmp_path,monkeypatch):
    mem,_=build(tmp_path)
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT',str(tmp_path))
    app=create_app()
    with TestClient(app) as client:
        disabled=client.get('/v1/platform/material-knowledge/pending')
        assert disabled.status_code==503
        monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEW_TOKEN','X'*40)
        monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEWERS','engineer')
        monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEWER','engineer')
        assert client.get('/v1/platform/material-knowledge/pending',
                          headers={'x-material-review-token':'wrong'}).status_code==403
        headers={'x-material-review-token':'X'*40}
        assert len(client.get('/v1/platform/material-knowledge/pending',headers=headers).json()['graph_proposals'])==2
        proposal=mem.pending_graph()[0]['proposal_id']
        answer=client.post(f'/v1/platform/material-knowledge/graph/{proposal}/decision',
            headers=headers,json={'decision':'CONFIRM','truth_source_ref':'Verified engineering manual',
                                  'equipment_scope':SCOPE,'expected_snapshot':0})
        assert answer.status_code==200,answer.text
        assert answer.json()['snapshot']==1
        endpoint=f'/v1/platform/material-knowledge/graph/{proposal}/retract'
        correction={'reason':'Independent correction of an outdated relation',
                    'truth_source_ref':'Validation record #27','expected_snapshot':1}
        assert client.post(endpoint,json=correction).status_code==403
        assert client.post(endpoint,headers=headers,json={**correction,'reason':'too'}).status_code==422
        withdrawn=client.post(endpoint,headers=headers,json=correction)
        assert withdrawn.status_code==200,withdrawn.text
        assert withdrawn.json()['snapshot']==2
        assert withdrawn.json()['status']=='RETRACTED'


def test_full_material_job_builds_graph_and_next_job_reuses_only_reviewed(tmp_path,monkeypatch):
    from astrograph.integrations import material_graph_builder as builder
    model=FakeExtractor()
    monkeypatch.setattr(builder,'_backend_from_environment',lambda: model)
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_BUILD_ENABLED','1')
    store=UploadReportJobStore(tmp_path)
    fixture=Path(__file__).resolve().parents[1]/'examples'/'result_log_review'

    def runjob():
        job=store.create('diagnostic result with document graph')['job_id']
        blobs={p.name:p.read_bytes() for p in fixture.iterdir() if p.is_file()}
        manifest=json.loads(blobs['analysis_input.json'])
        manifest['equipment_scope']=SCOPE
        blobs['analysis_input.json']=json.dumps(manifest).encode()
        blobs['reference.md']=('# 文献\n\n'+TEXT).encode()
        for name,raw in blobs.items():
            store.put_file(job,name,raw)
        assert store.preflight(job)['can_start']
        return job,store.run(job)

    first,report=runjob()
    graph=report['analysis']['material_reasoning']['graph_construction']
    assert graph['status']=='SUCCEEDED' and graph['proposal_count']>=1
    assert (tmp_path/first/'output'/'material_graph_construction.json').exists()
    assert not report['analysis']['material_reasoning']['candidates'][0]['approved_document_graph']['paths']
    mem=MaterialCaseMemory(tmp_path/'material_knowledge'/'cases.sqlite3',reviewers=('engineer',))
    for proposal in mem.pending_graph():
        mem.decide_graph(proposal_id=proposal['proposal_id'],reviewer='engineer',
                         decision='CONFIRM',truth_source_ref='Manual review 1',
                         equipment_scope=SCOPE)
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_BUILD_ENABLED','0')
    second,report2=runjob()
    found=report2['analysis']['material_reasoning']['candidates'][0]['approved_document_graph']
    assert found['status']=='SUCCEEDED'
    assert any(p['relation_type']=='ASSOCIATED_WITH' for p in found['paths'])
    assert not report2['analysis']['effect_guard']['active_kg_modified']
    assert model.calls>0


def test_host_cli_standalone_build_review_and_signed_release(tmp_path,monkeypatch,capsys):
    from astrograph.integrations.platform_cli import main
    from astrograph.integrations import material_graph_builder as builder
    monkeypatch.setattr(builder,'_backend_from_environment',FakeExtractor)
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEWERS','engineer')
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEWER','engineer')
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_RELEASE_SIGNING_KEY','X'*48)
    source=tmp_path/'standalone-reference.md'
    source.write_text('# 飞轮相关问题\n\n'+TEXT,encoding='utf-8')
    args=['--root',str(tmp_path/'job_store')]
    assert main(args+['graph-build','--reference',str(source),
                      '--equipment-scope',SCOPE,'--max-chunks','2'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['proposal_count']>=2 and result['model_called']
    assert result['source_group_id'].startswith('JOB-')
    assert main(args+['graph-pending'])==0
    staged=json.loads(capsys.readouterr().out)['graph_proposals']
    assert len(staged)>=2
    for item in staged:
        assert main(args+['graph-decide',item['proposal_id'],'--decision','CONFIRM',
                          '--truth-source-ref','Independent lab assessment',
                          '--equipment-scope',SCOPE])==0
        assert json.loads(capsys.readouterr().out)['status']=='CONFIRMED'
    assert main(args+['graph-export','--equipment-scope',SCOPE,
                      '--output',str(tmp_path/'releases')])==0
    release=json.loads(capsys.readouterr().out)['directory']
    assert main(args+['graph-verify',release])==0
    assert json.loads(capsys.readouterr().out)['verification_status']=='VERIFIED'
    assert main(args+['graph-publish-neo4j',release])==0
    assert json.loads(capsys.readouterr().out)['status']=='VERIFIED_DRY_RUN'
    assert main(args+['graph-retract',staged[0]['proposal_id'],
                      '--reason','Corrected engineering applicability scope',
                      '--truth-source-ref','Engineering change notice #7',
                      '--expected-snapshot','2'])==0
    assert json.loads(capsys.readouterr().out)['status']=='RETRACTED'
    assert main(args+['graph-export','--equipment-scope',SCOPE,
                      '--output',str(tmp_path/'releases')])==0
    newer=json.loads(capsys.readouterr().out)
    assert newer['edge_count']==1 and newer['retracted_claim_count']==1


def test_critical_negation_is_not_extracted_as_positive_graph_fact():
    from dataclasses import replace
    text='ADCS_RW_FRICTION_INCREASE 不会导致 反作用飞轮轴承磨损。'
    graph=payload()
    graph['relations']=[dict(graph['relations'][0],evidence_quote=text)]
    chunk=replace(Chunk(),text=text)
    valid,rejected=validate_fragment(GraphChunkExtraction.model_validate(graph),chunk,source_group_id=JOB1)
    assert valid==[]
    assert rejected==[{'reason':'EXPLICIT_NEGATION_CANNOT_BE_AFFIRMATIVE'}]
    graph['relations'][0]['polarity']='NEGATED'
    valid,rejected=validate_fragment(GraphChunkExtraction.model_validate(graph),chunk,source_group_id=JOB1)
    assert len(valid)==1 and not rejected
    assert valid[0]['polarity_check']=='EXPLICIT_NEGATION_PRESENT'


def test_conditional_claim_is_labeled_before_expert_review():
    from dataclasses import replace
    text='如果出现过载，ADCS_RW_FRICTION_INCREASE 可能关联 反作用飞轮轴承磨损。'
    graph=payload()
    graph['relations']=[dict(graph['relations'][0],evidence_quote=text)]
    proposals,rejected=validate_fragment(GraphChunkExtraction.model_validate(graph),
                        replace(Chunk(),text=text),source_group_id=JOB1)
    assert not rejected and len(proposals)==1
    assert proposals[0]['claim_modality']=='HYPOTHETICAL_OR_CONDITIONAL'
    assert proposals[0]['evidence_status']=='MODEL_EXTRACTED_REFERENCE_CLAIM_UNVERIFIED'


def test_retracted_approval_excluded_from_future_snapshots_and_export(tmp_path):
    mem,_=build(tmp_path)
    proposals=mem.pending_graph()
    for n,p in enumerate(proposals):
        mem.decide_graph(proposal_id=p['proposal_id'],reviewer='engineer',decision='CONFIRM',
                         truth_source_ref='Independent test record',equipment_scope=SCOPE,
                         expected_snapshot=n)
    original=mem.retrieve_graph(fault_id='ADCS_RW_FRICTION_INCREASE',fault_name='?',
        equipment_scope=SCOPE,exclude_source_group_id=JOB2,snapshot=2)
    assert len(original['paths'])==2
    target=proposals[0]['proposal_id']
    with pytest.raises(PermissionError):
        mem.retract_graph(proposal_id=target,reviewer='unapproved',
                          reason='Technical correction',truth_source_ref='Correction ticket')
    with pytest.raises(MaterialMemoryConflict):
        mem.retract_graph(proposal_id=target,reviewer='engineer',
                          reason='Technical correction',truth_source_ref='Correction ticket',
                          expected_snapshot=1)
    decision=mem.retract_graph(proposal_id=target,reviewer='engineer',
        reason='Independent evidence disproves previous applicability',
        truth_source_ref='Verification record R-2026',expected_snapshot=2)
    assert decision['status']=='RETRACTED' and decision['snapshot']==3
    now=mem.retrieve_graph(fault_id='ADCS_RW_FRICTION_INCREASE',fault_name='?',
        equipment_scope=SCOPE,exclude_source_group_id=JOB2,snapshot=3)
    assert all(r['proposal_id']!=target for r in now['paths'])
    assert len(mem.retrieve_graph(fault_id='ADCS_RW_FRICTION_INCREASE',fault_name='?',
        equipment_scope=SCOPE,exclude_source_group_id=JOB2,snapshot=2)['paths'])==2
    trail=mem.graph_retraction_audit(equipment_scope=SCOPE,snapshot=3)
    assert len(trail)==1 and trail[0]['proposal_id']==target
    newer=mem.export_graph_release(equipment_scope=SCOPE,snapshot=3,output_root=tmp_path/'new')
    m,g=verify_graph_release(newer['directory'])
    assert m['edge_count']==1 and m['retracted_claim_count']==1
    assert [r['proposal_id'] for r in g['retractions']]==[target]
    assert not {e['proposal_id'] for e in g['relations']} & {target}
    older=mem.export_graph_release(equipment_scope=SCOPE,snapshot=2,output_root=tmp_path/'old')
    assert verify_graph_release(older['directory'])[0]['edge_count']==2
    with pytest.raises(MaterialMemoryConflict):
        mem.retract_graph(proposal_id=target,reviewer='engineer',
                          reason='Duplicate retraction',truth_source_ref='Record duplicate')


def test_graph_conflicts_are_flagged_as_unresolved_reference_disputes(tmp_path):
    from dataclasses import replace
    mem,_=build(tmp_path)
    first=next(p for p in mem.pending_graph() if p['relation_type']=='ASSOCIATED_WITH')
    mem.decide_graph(proposal_id=first['proposal_id'],reviewer='engineer',
        decision='CONFIRM',truth_source_ref='Engineering source A',equipment_scope=SCOPE)
    text='ADCS_RW_FRICTION_INCREASE 不会导致 反作用飞轮轴承磨损。'
    data=payload()
    data['relations']=[dict(data['relations'][0],evidence_quote=text,polarity='NEGATED')]
    chunk=replace(Chunk(),text=text,chunk_id='DOC-N',source_sha256='f'*64)
    result=build_material_graph(chunks=[chunk],source_group_id=JOB2,memory=mem,
                                enabled=True,backend=FakeExtractor(data))
    assert result['proposal_count']==1
    second=next(p for p in mem.pending_graph() if p['source_group_id']==JOB2)
    mem.decide_graph(proposal_id=second['proposal_id'],reviewer='engineer',
        decision='CONFIRM',truth_source_ref='Engineering source B',equipment_scope=SCOPE)
    found=mem.retrieve_graph(fault_id='ADCS_RW_FRICTION_INCREASE',fault_name='?',
        equipment_scope=SCOPE,exclude_source_group_id='',snapshot=2)
    assert found['contradicted_relation_groups']==1
    assert len(found['paths'])==2
    assert all(p['conflict_status']=='CONTESTED_BY_APPROVED_REFERENCE' for p in found['paths'])
    assert not found['diagnostic_vote_allowed']


def test_neo4j_incremental_publish_deactivates_withdrawn_claims_and_blocks_rollback(tmp_path,monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_RELEASE_SIGNING_KEY','S'*48)
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_NEO4J_WRITE_ENABLED','1')
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_NEO4J_DATABASE','material_graph_private')
    mem,_=build(tmp_path)
    proposals=mem.pending_graph()
    for p in proposals:
        mem.decide_graph(proposal_id=p['proposal_id'],reviewer='engineer',
            decision='CONFIRM',truth_source_ref='Independent fault manual v3',equipment_scope=SCOPE)
    prior=mem.export_graph_release(equipment_scope=SCOPE,snapshot=2,output_root=tmp_path/'exports')
    mem.retract_graph(proposal_id=proposals[0]['proposal_id'],reviewer='engineer',
        reason='Independent lab observation invalidates this assertion',
        truth_source_ref='Lab evidence R17')
    current=mem.export_graph_release(equipment_scope=SCOPE,snapshot=3,output_root=tmp_path/'exports')

    class Result:
        def __init__(self, value): self.value=value
        def single(self): return self.value

    class Transaction:
        def __init__(self):
            self.snapshot=-1
            self.release_id=''
            self.claims={}
            self.queries=[]
        def run(self,cypher,**params):
            self.queries.append((cypher,params))
            if 'RETURN h.snapshot AS snapshot' in cypher:
                return Result({'snapshot':self.snapshot,'release_id':self.release_id})
            if 'MERGE (c:AstroMaterialClaim' in cypher:
                self.claims[params['proposal_id']]={'release_id':params['release_id'],'active':True}
            if 'SET c.current_release_active=false' in cypher:
                for item in self.claims.values():
                    if item['release_id']!=params['release_id']:
                        item['active']=False
            if 'SET h.snapshot=$snapshot' in cypher:
                self.snapshot=params['snapshot']
                self.release_id=params['release_id']
            return Result(None)

    class Session:
        def __init__(self,tx): self.tx=tx
        def __enter__(self): return self
        def __exit__(self,*_): pass
        def execute_write(self,callback,*args): return callback(self.tx,*args)

    class Driver:
        def __init__(self,tx): self.tx=tx
        def session(self,*,database):
            assert database=='material_graph_private'
            return Session(self.tx)

    tx=Transaction()
    driver=Driver(tx)
    assert publish_release_neo4j(prior['directory'],confirm_write=True,driver=driver)['claims_imported']==2
    assert sum(item['active'] for item in tx.claims.values())==2
    published=publish_release_neo4j(current['directory'],confirm_write=True,driver=driver)
    assert published['claims_imported']==1
    assert published['inactive_claims_synchronized']
    assert tx.claims[proposals[0]['proposal_id']]['active'] is False
    assert tx.claims[proposals[1]['proposal_id']]['active'] is True
    with pytest.raises(MaterialGraphReleaseError,match='older'):
        publish_release_neo4j(prior['directory'],confirm_write=True,driver=driver)
    assert tx.snapshot==3


def test_graph_status_cli_and_privileged_api_track_active_vs_retracted(tmp_path,monkeypatch,capsys):
    from astrograph.integrations.platform_cli import main
    mem,_=build(tmp_path)
    ids=[p['proposal_id'] for p in mem.pending_graph()]
    for p in ids:
        mem.decide_graph(proposal_id=p,reviewer='engineer',decision='CONFIRM',
                         truth_source_ref='Independent engineer review',equipment_scope=SCOPE)
    mem.retract_graph(proposal_id=ids[0],reviewer='engineer',
                      reason='Evidence contradicts approved statement',
                      truth_source_ref='Corrective test data')
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT',str(tmp_path))
    assert main(['--root',str(tmp_path),'graph-status','--equipment-scope',SCOPE])==0
    status=json.loads(capsys.readouterr().out)
    assert status['snapshot']==3
    assert status['active_approved_relations']==1
    assert status['approved_relations_ever']==2
    assert status['retracted_relations']==1
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEW_TOKEN','Z'*40)
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEWERS','engineer')
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEWER','engineer')
    with TestClient(create_app()) as client:
        uri='/v1/platform/material-knowledge/graph/status'
        assert client.get(uri).status_code==403
        r=client.get(uri,headers={'x-material-review-token':'Z'*40},
                     params={'equipment_scope':SCOPE})
        assert r.status_code==200
        assert r.json()['active_approved_relations']==1
