"""R6: independent symptom coverage; withheld outcome may not enter retrieval."""
import hashlib
import json
from pathlib import Path

from astrograph.retrieval.observation_balanced import retrieve_observation_balanced
from astrograph.retrieval.reported_observations import review_numbered_observations
from astrograph.integrations.platform_knowledge import (
    UploadedDocumentKnowledgeIndex, PlatformKnowledgeEnrichmentService, PlatformKnowledgeConfig
)

CASE = (
    '故障现象：\n'
    '1. 输出总电流和主误差电压3次跌落\n'
    '2. 轴温最高升至181°C，壳温最高升至150°C\n'
    '3. 调整南太阳翼角度时出现堵转\n'
)


def make_docs(tmp_path):
    docs = {
        'electrical.md': '# Power\n输出总电流和主误差电压出现跌落，需对照采样时刻和供电负载。',
        'thermal.md': '# Thermal\n轴温升高与壳温变化需要测点校准及热设计阈值。',
        'actuation.md': '# Mechanism\n南太阳翼角度调整出现堵转告警，应核对指令和实际角度反馈。',
    }
    paths=[]
    for name, text in docs.items():
        target=tmp_path/name
        target.write_text(text, encoding='utf-8')
        paths.append(target)
    return paths


def test_multisymptom_retrieval_balances_electrical_thermal_actuation(tmp_path):
    index=UploadedDocumentKnowledgeIndex(top_k=3)
    chunks, _=index.build(make_docs(tmp_path))
    ledger=review_numbered_observations(CASE)
    hits, receipt=retrieve_observation_balanced(index, chunks,
        controlled_query='', task=CASE, observation_ledger=ledger)
    assert receipt['strategy']=='OBSERVATION_BALANCED_RRF_NON_VOTING'
    assert receipt['observation_queries']==3
    assert len(hits)==3
    assert {x['filename'] for x in hits}=={'electrical.md','thermal.md','actuation.md'}
    assert receipt['observation_coverage_count']==3
    assert not receipt['uncovered_observation_ids']
    assert all(not x['diagnostic_vote_allowed'] for x in hits)
    assert all(x['knowledge_validation_status']=='UNREVIEWED_SESSION_UPLOAD' for x in hits)


def test_no_labeled_truth_or_mechanism_in_queries(tmp_path):
    index=UploadedDocumentKnowledgeIndex(top_k=3)
    chunks,_=index.build(make_docs(tmp_path))
    ledger=review_numbered_observations(CASE)
    hits, receipt=retrieve_observation_balanced(index, chunks,
        controlled_query='', task=CASE, observation_ledger=ledger)
    assert receipt['query_count']==4
    assert receipt['query_sha256']==[
        hashlib.sha256(text.encode()).hexdigest()
        for text in [CASE.strip(), *(x['reported_text'] for x in ledger['observations'])]
    ]
    assert not any(x.get('verified_root_cause', False) for x in hits)
    assert receipt['retrieval_coverage_is_not_mechanism_validation']


def test_no_documents_and_missing_observation_are_distinguished():
    ledger=review_numbered_observations(CASE)
    hits,receipt=retrieve_observation_balanced(
        UploadedDocumentKnowledgeIndex(), [], controlled_query='',
        task=CASE, observation_ledger=ledger)
    assert hits==[]
    assert receipt['observation_coverage_count']==0
    assert len(receipt['uncovered_observation_ids'])==3
    no_ledger=review_numbered_observations('No numbered lines')
    hits,receipt=retrieve_observation_balanced(
        UploadedDocumentKnowledgeIndex(), [], controlled_query='',
        task='No numbered lines', observation_ledger=no_ledger)
    assert receipt['strategy']=='LEGACY_SINGLE_QUERY'


def test_live_enrichment_contains_coverage_receipt_without_cause(tmp_path):
    service=PlatformKnowledgeEnrichmentService(PlatformKnowledgeConfig(local_top_k=3))
    result=service.enrich(task=CASE,
        diagnosis_snapshot={'candidate_fault_id':None,'evidence':[]},
        document_paths=make_docs(tmp_path), output_root=tmp_path/'out')
    assert result['diagnosis_frozen']
    assert result['paperseek']['status']=='NOT_CONFIGURED'
    assert result['local_retrieval']['strategy_receipt']['observation_coverage_count']==3
    assert result['user_reported_observation_ledger']['no_frozen_fault_candidate_generated']
    assert result['generation']['status']=='NOT_CONFIGURED'
    assert all(x['diagnostic_vote_allowed'] is False for x in result['local_retrieval']['hits'])
    assert (tmp_path/'out'/'knowledge_enrichment.json').is_file()


def _reviewed_relation(eid, sid, sname, stype, tid, tname, ttype, relation, *,
                       scope='SAT_DEMO_A', polarity='AFFIRMATIVE'):
    return {
        'proposal_id':eid, 'equipment_scope':scope, 'snapshot':1,
        'review_status':'INDEPENDENTLY_REVIEWED_REFERENCE_RELATION',
        'source':{'entity_id':sid,'name':sname,'entity_type':stype},
        'target':{'entity_id':tid,'name':tname,'entity_type':ttype},
        'relation_type':relation, 'polarity':polarity,
        'claim_modality':'REFERENCE_ASSERTION_REQUIRES_REVIEW',
        'provenance':{'source_sha256':'a'*64,'source_locators':['page:5']},
        'evidence_quote':f'{sname} 与 {tname} 关系由文献描述，仍需专家核验。',
        'truth_source_ref':'independent-review-1','reviewer':'expert',
    }


def _synthetic_multistage_edges():
    # Deliberately unrelated to the provided solar-wing gold case: this is a
    # generic ontology/depth regression, not a stored target diagnosis.
    return [
        _reviewed_relation('a','f','FAULT_PUMP','FAULT_MODE',
                           'm1','seal wear','FAILURE_MECHANISM','ASSOCIATED_WITH'),
        _reviewed_relation('b','m1','seal wear','FAILURE_MECHANISM',
                           'm2','lubricant contamination','FAILURE_MECHANISM','ASSOCIATED_WITH'),
        _reviewed_relation('c','m2','lubricant contamination','FAILURE_MECHANISM',
                           'm3','cooling efficiency loss','FAILURE_MECHANISM','ASSOCIATED_WITH'),
        _reviewed_relation('d','m3','cooling efficiency loss','FAILURE_MECHANISM',
                           's','bearing temperature rises','SYMPTOM','MANIFESTS_AS'),
    ]


def test_four_hop_reviewed_mechanism_association_opt_in_without_causal_proof():
    from astrograph.integrations.material_multihop import evaluate_bidirectional_hypotheses
    kwargs = dict(
        candidates=[{'fault_id':'PUMP','fault_name':'FAULT_PUMP'}],
        equipment_scope='SAT_DEMO_A', log_lines=[],
        reviewed_edges=_synthetic_multistage_edges(), snapshot=4,
    )
    old=evaluate_bidirectional_hypotheses(**kwargs)
    assert not old['candidates'][0]['expected_symptoms']
    extended=evaluate_bidirectional_hypotheses(**kwargs, max_hops=4)
    row=extended['candidates'][0]
    assert {s['symptom_id'] for s in row['expected_symptoms']}=={'s'}
    assert row['forward_path_count'] >= 1
    assert row['expected_symptoms'][0]['sensor_observation_verified'] is False
    assert not extended['current_root_cause_proven']
    assert not extended['diagnostic_vote_allowed']
    excluded=[*(_synthetic_multistage_edges()[:2]),
              {**_synthetic_multistage_edges()[2], 'equipment_scope':'SAT_OTHER'},
              _synthetic_multistage_edges()[3]]
    blocked=evaluate_bidirectional_hypotheses(**{**kwargs,'reviewed_edges':excluded}, max_hops=4)
    assert not blocked['candidates'][0]['expected_symptoms']


def test_document_ontology_accepts_reviewer_gated_mechanism_to_mechanism():
    from astrograph.integrations.material_graph_builder import (
        GraphChunkExtraction,validate_fragment,ALLOWED_TRIPLES
    )
    assert ('FAILURE_MECHANISM','ASSOCIATED_WITH','FAILURE_MECHANISM') in ALLOWED_TRIPLES
    from dataclasses import dataclass
    @dataclass
    class Chunk:
        text: str = 'seal wear 与 lubricant contamination 相关。'
        chunk_id: str = 'chunk-1'
        filename: str = 'manual.md'
        source_locators: tuple = ('line:1',)
        section_path: tuple = ('Equipment',)
        source_sha256: str = 'b'*64
    proposed=GraphChunkExtraction.model_validate({'entities':[
        {'local_id':'a','entity_type':'FAILURE_MECHANISM','name':'seal wear'},
        {'local_id':'b','entity_type':'FAILURE_MECHANISM','name':'lubricant contamination'}],
        'relations':[{'source_local_id':'a','target_local_id':'b',
                      'relation_type':'ASSOCIATED_WITH',
                      'evidence_quote':'seal wear 与 lubricant contamination 相关。'}]})
    rows, rejected=validate_fragment(proposed,Chunk(),source_group_id='TEST-GROUP')
    assert not rejected
    assert len(rows)==1
    assert rows[0]['evidence_status'] == 'MODEL_EXTRACTED_REFERENCE_CLAIM_UNVERIFIED'
    assert rows[0]['active_kg_modified'] is False


def test_hop_extension_requires_admin_opt_in(monkeypatch):
    import pytest
    from astrograph.integrations.material_reasoning import _graph_review_hops
    monkeypatch.delenv('ASTROGRAPH_MATERIAL_GRAPH_MAX_HOPS', raising=False)
    assert _graph_review_hops()==3
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_MAX_HOPS','4')
    assert _graph_review_hops()==4
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_GRAPH_MAX_HOPS','999')
    with pytest.raises(ValueError, match='must be 3 or 4'):
        _graph_review_hops()
