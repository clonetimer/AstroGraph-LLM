"""Acceptance tests: hybrid-RAG, bound KG, independent memory and library review."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from astrograph.api import create_app
from astrograph.integrations.material_case_memory import MaterialCaseMemory, MaterialMemoryConflict
from astrograph.integrations.upload_report import UploadReportJobStore
from astrograph.integrations.platform_knowledge import (
    PlatformKnowledgeClaim, PlatformKnowledgeExplanation, _validate_explanation,
)

EXAMPLE=Path(__file__).resolve().parents[1]/'examples'/'result_log_review'


def build(store, *, equipment_scope=None, log_override=None, ref_override=None):
    job=store.create('根据诊断结果、运行日志和文献，分析根因及处置')['job_id']
    files={p.name:p.read_bytes() for p in EXAMPLE.iterdir() if p.is_file()}
    manifest=json.loads(files['analysis_input.json'].decode())
    if equipment_scope:
        manifest['equipment_scope']=equipment_scope
    if log_override is not None:
        files['operations.log']=log_override.encode()
    if ref_override is not None:
        files['reference.md']=ref_override.encode()
    files['analysis_input.json']=json.dumps(manifest,ensure_ascii=False).encode()
    for name,blob in files.items():
        store.put_file(job,name,blob)
    assert store.preflight(job)['can_start']
    return job


def test_hybrid_fusion_is_visible_and_never_promotes_ground_truth(tmp_path):
    store=UploadReportJobStore(tmp_path)
    job=build(store)
    report=store.run(job)
    assert report['status']=='SUCCEEDED'
    review=report['analysis']['result_log_review']
    fused=report['analysis']['material_reasoning']
    assert fused['schema_version']=='astrograph-material-knowledge-fusion.v1'
    assert review['top_fault_id']=='ADCS_RW_FRICTION_INCREASE'
    assert fused['graph_query']['status']=='NOT_CONFIGURED'
    assert fused['semantic_backend']['status']=='NOT_CONFIGURED'
    c=fused['candidates'][0]
    assert c['fault_id']==review['top_fault_id']
    assert c['reference_retrieval']['hits']
    assert c['reference_retrieval']['hits'][0]['source_locators']
    assert c['reference_retrieval']['hits'][0]['source_sha256']
    assert 'BM25' in c['reference_retrieval']['methods']
    assert 'CHAR_NGRAM' in c['reference_retrieval']['methods']
    assert c['log_evidence_matrix']['supporting_ids']
    assert c['evidence_assessment']['verified_current_root_cause'] is False
    assert all(not action['control_authorized'] for action in review['handling_measures'])
    assert fused['case_learning']['status']=='PENDING_EXPERT_REVIEW'
    assert fused['document_learning']['status']=='PENDING_EXPERT_REVIEW'
    assert all(x['status']=='PENDING' for x in fused['case_learning']['proposals'])
    assert all(x['status']=='PENDING' for x in fused['document_learning']['proposals'])
    md=store.report_markdown_path(job).read_text(encoding='utf-8')
    assert 'RAG × 知识图谱' in md
    assert 'MLOG-' in md
    assert '知识持续积累' in md
    assert not report['analysis']['effect_guard']['active_kg_modified']


def test_log_contradiction_is_cue_not_confirmed_cause(tmp_path):
    store=UploadReportJobStore(tmp_path)
    job=build(store,log_override='2026-10-08T10:00:00Z INFO ADCS motor_current_a normal\n'
                           '2026-10-08T10:00:45Z INFO wheel_0 winding_temp recovered\n')
    report=store.run(job)
    c=report['analysis']['material_reasoning']['candidates'][0]
    log=c['log_evidence_matrix']
    assert len(log['contradicting_ids'])==2
    assert not log['supporting_ids']
    assert c['evidence_assessment']['assessment']=='CONTRADICTORY_CUES_REQUIRE_REVIEW'
    assert not log['temporal_causality_verified']
    assert not c['evidence_assessment']['verified_current_root_cause']


def test_only_independently_confirmed_equipment_scoped_cases_are_retrievable(tmp_path):
    root=tmp_path/'jobs'
    store=UploadReportJobStore(root)
    first=build(store,equipment_scope='SAT-A_ADCS_V2')
    a=store.run(first)['analysis']['material_reasoning']
    first_c=a['case_learning']['proposals'][0]['case_id']
    first_d=a['document_learning']['proposals'][0]['document_id']
    mem=MaterialCaseMemory(root/'material_knowledge'/'cases.sqlite3',reviewers=('chief-reviewer',))
    assert mem.snapshot()==0
    assert mem.retrieve_confirmed(fault_id='ADCS_RW_FRICTION_INCREASE',equipment_scope='SAT-A_ADCS_V2',
                                  exclude_source_group_id='ANOTHER')['hits']==[]
    with pytest.raises(PermissionError):
        mem.decide(case_id=first_c,reviewer='untrusted',decision='CONFIRM',truth_source_ref='TICKET-100',
                   verified_fault_id='ADCS_RW_FRICTION_INCREASE',equipment_scope='SAT-A_ADCS_V2')
    with pytest.raises(ValueError):
        mem.decide(case_id=first_c,reviewer='chief-reviewer',decision='CONFIRM',truth_source_ref='TICKET-100',
                   verified_fault_id='ADCS_RW_FRICTION_INCREASE',equipment_scope='UNKNOWN')
    assert mem.snapshot()==0
    decided=mem.decide(case_id=first_c,reviewer='chief-reviewer',decision='CONFIRM',
                      truth_source_ref='Independent lab ticket 100',verified_fault_id='ADCS_RW_FRICTION_INCREASE',
                      equipment_scope='SAT-A_ADCS_V2',approved_actions=['由值班专家复核电机驱动机构，按已批准规程处理'],
                      expected_snapshot=0)
    assert decided['snapshot']==1
    with pytest.raises(MaterialMemoryConflict):
        mem.decide(case_id=first_c,reviewer='chief-reviewer',decision='CONFIRM',
                   truth_source_ref='Other source',verified_fault_id='ADCS_RW_FRICTION_INCREASE',
                   equipment_scope='SAT-A_ADCS_V2')
    docs=mem.decide_document(document_id=first_d,reviewer='chief-reviewer',decision='CONFIRM',
                             truth_source_ref='Reviewed applicable manual v2',equipment_scope='SAT-A_ADCS_V2',
                             expected_snapshot=1)
    assert docs['snapshot']==2
    assert not mem.retrieve_confirmed(fault_id='ADCS_RW_FRICTION_INCREASE',equipment_scope='SAT-B_ADCS_V1',
                                      exclude_source_group_id='OTHER')['hits']
    assert not mem.retrieve_confirmed(fault_id='ADCS_RW_FRICTION_INCREASE',equipment_scope='SAT-A_ADCS_V2',
                                      exclude_source_group_id=first)['hits']
    # A new job with the same bound equipment can reuse only approved history.
    second=build(store,equipment_scope='SAT-A_ADCS_V2',ref_override='# unrelated \nNo relevant reference data.')
    b=store.run(second)['analysis']['material_reasoning']
    assert b['memory_snapshot']==2
    assert b['approved_library_chunks_used']>=1
    c=b['candidates'][0]
    assert any(hit['knowledge_validation_status']=='EXPERT_APPROVED_HISTORICAL_REFERENCE'
               for hit in c['reference_retrieval']['hits'])
    assert c['historical_confirmed_cases']['hits'][0]['case_id']==first_c
    assert c['review_action_options'][0]['control_authorized'] is False
    assert c['evidence_assessment']['verified_current_root_cause'] is False
    unknown=build(store)
    empty=store.run(unknown)['analysis']['material_reasoning']
    assert empty['approved_library_chunks_used']==0
    assert empty['candidates'][0]['historical_confirmed_cases']['status']=='EQUIPMENT_BINDING_REQUIRED'


def test_rejected_document_never_in_searchable_library(tmp_path):
    store=UploadReportJobStore(tmp_path)
    job=build(store,equipment_scope='SAT-A_ADCS_V2')
    result=store.run(job)['analysis']['material_reasoning']
    doc=result['document_learning']['proposals'][0]['document_id']
    mem=MaterialCaseMemory(tmp_path/'material_knowledge'/'cases.sqlite3',reviewers=('r1',))
    assert mem.decide_document(document_id=doc,reviewer='r1',decision='REJECT',truth_source_ref='wrong edition')['status']=='REJECTED'
    assert mem.retrieve_reference_chunks(equipment_scope='SAT-A_ADCS_V2',exclude_source_group_id='OTHER',snapshot=0)==[]
    assert not mem.pending_documents()


def test_historical_misclassification_is_counterexample_not_action_advice(tmp_path):
    root=tmp_path/'jobs'
    store=UploadReportJobStore(root)
    first=build(store,equipment_scope='SAT-A_ADCS_V2')
    first_fusion=store.run(first)['analysis']['material_reasoning']
    case_id=first_fusion['case_learning']['proposals'][0]['case_id']
    memory=MaterialCaseMemory(root/'material_knowledge'/'cases.sqlite3',reviewers=('independent-expert',))
    approved=memory.decide(case_id=case_id,reviewer='independent-expert',decision='CONFIRM',
                          truth_source_ref='Lab-failure-analysis-777',
                          verified_fault_id='ADCS_RW_JAM',equipment_scope='SAT-A_ADCS_V2',
                          approved_actions=['只针对经确认的卡滞故障使用的独立批准规程'])
    assert approved['status']=='CONFIRMED'
    second=build(store,equipment_scope='SAT-A_ADCS_V2')
    new_fusion=store.run(second)['analysis']['material_reasoning']
    current=new_fusion['candidates'][0]
    history=current['historical_confirmed_cases']['hits']
    assert len(history)==1
    assert history[0]['verified_fault_id']=='ADCS_RW_JAM'
    assert history[0]['relation_status']=='PRIOR_UPSTREAM_MISCLASSIFICATION'
    assert current['evidence_assessment']['historical_correction_ids']==[case_id]
    assert current['review_action_options']==[]
    assert not current['evidence_assessment']['verified_current_root_cause']


def test_llm_material_citations_must_resolve_to_frozen_context():
    snapshot={'candidate_fault_id':'ADCS_RW_FRICTION_INCREASE','evidence':[]}
    context={'candidates':[{'log_evidence_matrix':{'events':[{'evidence_id':'MLOG-00001'}]},
                            'reference_retrieval':{'hits':[{'knowledge_id':'DOC-1'}]},
                            'graph_evidence_paths':[{'graph_path_id':'KG-ABC'}],
                            'historical_confirmed_cases':{'hits':[{'case_id':'MRC-123'}]}}]}
    valid=PlatformKnowledgeExplanation(candidate_fault_id=snapshot['candidate_fault_id'],
         abstained=False,summary='仅描述上下文可定位的线索',
         claims=[PlatformKnowledgeClaim(kind='MATERIAL_LOG_CUE',text='日志线索',citation_ids=['MLOG-00001']),
                 PlatformKnowledgeClaim(kind='GRAPH_ASSOCIATION',text='图谱关联',citation_ids=['KG-ABC']),
                 PlatformKnowledgeClaim(kind='MATERIAL_REFERENCE',text='文献片段',citation_ids=['DOC-1']),
                 PlatformKnowledgeClaim(kind='CONFIRMED_CASE',text='历史已审案例',citation_ids=['MRC-123'])])
    _validate_explanation(valid,snapshot,[],[],[],context)
    bad=valid.model_copy(update={'claims':[
        PlatformKnowledgeClaim(kind='GRAPH_ASSOCIATION',text='伪造的图谱证据',citation_ids=['KG-FAKE'])]})
    with pytest.raises(ValueError,match='unknown or unapproved'):
        _validate_explanation(bad,snapshot,[],[],[],context)


def _graph_files(folder: Path):
    nodes=folder/'nodes.csv'
    edges=folder/'edges.csv'
    with nodes.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['node_id','node_type','name','properties_json'])
        w.writeheader()
        w.writerow({'node_id':'ADCS_RW_FRICTION_INCREASE','node_type':'FaultMode','name':'friction','properties_json':'{}'})
        w.writerow({'node_id':'EV_RW_MOTOR_CURRENT','node_type':'EvidenceDefinition','name':'motor_current','properties_json':'{}'})
    with edges.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['source_id','target_id','relation','properties_json'])
        w.writeheader()
        w.writerow({'source_id':'ADCS_RW_FRICTION_INCREASE','target_id':'EV_RW_MOTOR_CURRENT',
                    'relation':'REQUIRES_EVIDENCE','properties_json':'{"review_status":"REVIEWED"}'})
    binding=folder/'kg_binding.json'
    binding.write_text(json.dumps({'read_user':'readonly','release_id':'KG-RELEASE-1',
          'knowledge_version':'KG-V1','nodes_sha256':hashlib.sha256(nodes.read_bytes()).hexdigest(),
          'edges_sha256':hashlib.sha256(edges.read_bytes()).hexdigest(),
          'fallback_mode':'csv_on_unavailable','fallback_nodes_csv':'nodes.csv','fallback_edges_csv':'edges.csv'}))
    return binding,edges


def test_bound_csv_graph_is_readonly_and_detects_hash_drift(tmp_path,monkeypatch):
    binding,edges=_graph_files(tmp_path)
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_KG_BINDING_JSON',str(binding))
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_KG_MODE','csv')
    store=UploadReportJobStore(tmp_path/'jobs')
    log='2026-10-08T12:00:00Z WARN ADCS EV_RW_MOTOR_CURRENT motor_current_a elevated'
    first=build(store,log_override=log)
    r=store.run(first)['analysis']['material_reasoning']
    assert r['graph_query']['status']=='SUCCEEDED'
    assert r['graph_query']['backend']=='csv_reference_read_only'
    path=r['candidates'][0]['graph_evidence_paths'][0]
    assert path['evidence_definition_id']=='EV_RW_MOTOR_CURRENT'
    assert path['matched_log_ids']
    assert not r['active_kg_modified']
    assert not path['relation_provenance'].get('auto_confirmed',False)
    edges.write_text(edges.read_text()+'\nTAMPERED',encoding='utf-8')
    second=build(store)
    b=store.run(second)['analysis']['material_reasoning']
    assert b['status']=='PARTIAL'
    assert b['graph_query']['status']=='FAILED'
    assert b['graph_query']['error_type']=='ValueError'
    assert b['candidates'][0]['graph_evidence_paths']==[]


def test_review_api_fails_closed_by_default_and_checks_admin_token(tmp_path,monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT',str(tmp_path/'api-jobs'))
    monkeypatch.delenv('ASTROGRAPH_MATERIAL_REVIEW_TOKEN',raising=False)
    client=TestClient(create_app())
    assert client.get('/v1/platform/material-knowledge/pending').status_code==503
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEW_TOKEN','A_VALID_32_PLUS_CHARACTER_LOCAL_TOKEN_1234')
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEWER','expert1')
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEWERS','expert1')
    assert client.get('/v1/platform/material-knowledge/pending').status_code==403
    headers={'x-material-review-token':'A_VALID_32_PLUS_CHARACTER_LOCAL_TOKEN_1234'}
    listed=client.get('/v1/platform/material-knowledge/pending',headers=headers)
    assert listed.status_code==200
    assert listed.json()['snapshot']==0
    assert client.get('/v1/platform/capabilities').json()['material_knowledge_fusion']['manual_approval_required'] is True
    store=UploadReportJobStore(tmp_path/'api-jobs')
    job=build(store,equipment_scope='SAT-A_ADCS_V2')
    r=store.run(job)['analysis']['material_reasoning']
    case_id=r['case_learning']['proposals'][0]['case_id']
    req={'decision':'CONFIRM','truth_source_ref':'independent expert record 2026',
         'equipment_scope':'SAT-A_ADCS_V2','verified_fault_id':'ADCS_RW_FRICTION_INCREASE',
         'approved_actions':['review only'],'expected_snapshot':0}
    blocked=client.post(f'/v1/platform/material-knowledge/cases/{case_id}/decision',json=req)
    assert blocked.status_code==403
    approved=client.post(f'/v1/platform/material-knowledge/cases/{case_id}/decision',json=req,headers=headers)
    assert approved.status_code==200
    assert approved.json()['status']=='CONFIRMED'
    assert client.post(f'/v1/platform/material-knowledge/cases/{case_id}/decision',json=req,headers=headers).status_code==422
