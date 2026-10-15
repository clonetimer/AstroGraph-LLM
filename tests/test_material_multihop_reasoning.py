"""Acceptance tests: bounded actual paths, abstention, source identity and review queue."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from astrograph.api import create_app
from astrograph.integrations.material_graph_builder import GraphChunkExtraction, validate_fragment
from astrograph.integrations.material_case_memory import MaterialCaseMemory
from astrograph.integrations.material_multihop import evaluate_multihop, default_public_graph_path
from astrograph.integrations.result_log_review import SourceLine

JOB1='JOB-'+'B'*24
JOB2='JOB-'+'C'*24
SCOPE='SAT_X_ADCS_V1'
FAULT='ADCS_RW_FRICTION_INCREASE'
MECH='反作用飞轮轴承磨损'
SYMPTOM='飞轮电流升高'
TEXT=(f'{FAULT} 关联 {MECH}。\n'
      f'{MECH} 引起 {SYMPTOM}。\n'
      f'反作用飞轮 容易发生 {MECH}。\n'
      '反作用飞轮 容易出现 轴承间隙变化。\n')


@dataclass
class Chunk:
    text: str = TEXT
    chunk_id: str = 'TEST-SOURCE-CHUNK'
    filename: str = 'reviewed-manual.md'
    source_locators: tuple[str,...] = ('line:1-4',)
    section_path: tuple[str,...] = ('ADCS','机理核验')
    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


def graph_json(*, conditional=False, negative=False):
    return {'entities':[
      {'local_id':'f','entity_type':'FAULT_MODE','name':FAULT},
      {'local_id':'m','entity_type':'FAILURE_MECHANISM','name':MECH},
      {'local_id':'s','entity_type':'SYMPTOM','name':SYMPTOM},
      {'local_id':'c','entity_type':'COMPONENT','name':'反作用飞轮'},
      {'local_id':'m2','entity_type':'FAILURE_MECHANISM','name':'轴承间隙变化'}],
      'relations':[
        {'source_local_id':'f','target_local_id':'m','relation_type':'ASSOCIATED_WITH',
         'evidence_quote':f'{FAULT} 关联 {MECH}。',
         'polarity':'NEGATED' if negative else 'AFFIRMATIVE'},
        {'source_local_id':'m','target_local_id':'s','relation_type':'MANIFESTS_AS',
         'evidence_quote':f'{MECH} 引起 {SYMPTOM}。'},
        {'source_local_id':'c','target_local_id':'m','relation_type':'SUSCEPTIBLE_TO',
         'evidence_quote':f'反作用飞轮 容易发生 {MECH}。'},
        {'source_local_id':'c','target_local_id':'m2','relation_type':'SUSCEPTIBLE_TO',
         'evidence_quote':'反作用飞轮 容易出现 轴承间隙变化。'},
      ]}


def reviewed(tmp_path, *, scope=SCOPE, negative=False, all_edges=True):
    mem=MaterialCaseMemory(tmp_path/'material_knowledge'/'cases.sqlite3',reviewers=('expert',))
    records,errors=validate_fragment(GraphChunkExtraction.model_validate(graph_json(negative=negative)),Chunk(),source_group_id=JOB1)
    assert not errors and len(records)==4
    if not all_edges:
        records=records[:1]  # missing the crucial mechanism->symptom hop
    mem.stage_graph_proposals(source_group_id=JOB1,proposals=records)
    for i,rec in enumerate(records):
        mem.decide_graph(proposal_id=rec['proposal_id'],decision='CONFIRM',reviewer='expert',
                         truth_source_ref='manual-independent-QA-004',equipment_scope=scope,
                         expected_snapshot=i)
    return mem,records


def log(message: str) -> list[SourceLine]:
    return [SourceLine('log_reports','operations.log','line:7',message,'1'*64)]


def run(mem, log_lines, *, scope=SCOPE, snapshot=None):
    version=mem.snapshot() if snapshot is None else snapshot
    return evaluate_multihop(fault_id=FAULT,fault_name='反作用飞轮摩擦增大',
        equipment_scope=scope,log_lines=log_lines,
        reviewed_edges=mem.approved_graph_for_reasoning(
            equipment_scope=scope,snapshot=version,exclude_source_group_id=JOB2),
        snapshot=version,max_hops=3,public_graph_path=default_public_graph_path())


def test_two_three_hop_paths_and_no_reverse_causality(tmp_path):
    mem,_=reviewed(tmp_path)
    r=run(mem,log('2026-10-08T06:10:00 飞轮电流升高 告警'))
    assert r['decision']=='PROVISIONAL_EVIDENCE_BACKED_HYPOTHESIS'
    assert r['response_mode']=='EVIDENCE_BOUND_PROVISIONAL_ONLY'
    assert any(p['hop_count']==2 and p['terminal_entity']['entity_type']=='SYMPTOM'
               and p['eligible_as_hypothesis_path'] for p in r['approved_paths'])
    assert any(p['hop_count']==3 and p['contains_reverse_association']
               and not p['eligible_as_hypothesis_path'] for p in r['approved_paths'])
    assert all(h['edge_id'].startswith('MGR-') for p in r['approved_paths'] for h in p['hops'])
    assert all(h['provenance']['source_sha256']==Chunk().source_sha256
               for p in r['approved_paths'] for h in p['hops'])
    assert r['independent_log_support_cues'][0]['evidence_id']=='MLOG-00001'
    assert not r['verified_current_root_cause'] and not r['automatic_control_authorized']


def test_absence_missing_hop_and_negative_log_all_abstain(tmp_path):
    mem,_=reviewed(tmp_path/'good')
    none=run(mem,log('2026-10-08 normal subsystem checks'))
    assert none['decision']=='ABSTAIN_NO_SYMPTOM_LOG_ANCHOR'
    assert 'MISSING_INDEPENDENT_SYMPTOM_OBSERVATION' in none['knowledge_gap_codes']
    opposition=run(mem,log('2026-10-08 飞轮电流升高 已排除、恢复正常'))
    assert opposition['decision']=='ABSTAIN_CONTRADICTORY_LOG_EVIDENCE'
    incomplete,_=reviewed(tmp_path/'missing',all_edges=False)
    r=run(incomplete,log('2026-10-08 飞轮电流升高 告警'))
    assert r['decision']=='ABSTAIN_NO_SYMPTOM_LOG_ANCHOR'
    assert not r['independent_log_support_cues']


def test_wrong_or_unknown_scope_fails_closed(tmp_path):
    mem,_=reviewed(tmp_path)
    o=run(mem,log('飞轮电流升高 告警'),scope='SAT_Y_ADCS_V1')
    assert o['decision']=='ABSTAIN_NO_APPROVED_PATH'
    assert not o['approved_paths']
    x=evaluate_multihop(fault_id=FAULT,fault_name=FAULT,equipment_scope=None,
        log_lines=log('飞轮电流升高'),reviewed_edges=[],public_graph_path=default_public_graph_path())
    assert x['decision']=='ABSTAIN_EQUIPMENT_SCOPE_REQUIRED'


def test_negated_or_contested_review_claim_never_supports_hypothesis(tmp_path):
    mem,records=reviewed(tmp_path,negative=True)
    bad=run(mem,log('飞轮电流升高 告警'))
    assert bad['decision']!='PROVISIONAL_EVIDENCE_BACKED_HYPOTHESIS'
    assert any(p['negated_edge_ids'] for p in bad['approved_paths'])
    # A second independent approved opposite polarity should explicitly contest the same triple.
    positive,_=validate_fragment(GraphChunkExtraction.model_validate(graph_json()),Chunk(),source_group_id=JOB2)
    opposite=positive[0]
    mem.stage_graph_proposals(source_group_id=JOB2,proposals=[opposite])
    mem.decide_graph(proposal_id=opposite['proposal_id'],reviewer='expert',decision='CONFIRM',
        truth_source_ref='independent-journal-QA-009',equipment_scope=SCOPE)
    # exclude a nonexistent source group to expose both independently approved batches
    mixed=evaluate_multihop(fault_id=FAULT,fault_name=FAULT,equipment_scope=SCOPE,
        log_lines=log('飞轮电流升高 告警'),
        reviewed_edges=mem.approved_graph_for_reasoning(equipment_scope=SCOPE,snapshot=mem.snapshot(),
                         exclude_source_group_id='NONEXISTENT'),snapshot=mem.snapshot(),
        public_graph_path=default_public_graph_path())
    assert mixed['knowledge_conflict_groups']==1
    assert mixed['decision']=='ABSTAIN_GRAPH_PATH_BLOCKED' or mixed['decision']=='ABSTAIN_NO_SYMPTOM_LOG_ANCHOR'


def test_retraction_applies_only_to_new_snapshot(tmp_path):
    mem,records=reviewed(tmp_path)
    before=run(mem,log('飞轮电流升高'))
    assert before['decision']=='PROVISIONAL_EVIDENCE_BACKED_HYPOTHESIS'
    mem.retract_graph(proposal_id=records[1]['proposal_id'],reviewer='expert',
                      reason='Independent engineering verification revokes this mechanism to symptom mapping',
                      truth_source_ref='correction-memo-2026-QA-02')
    after=run(mem,log('飞轮电流升高'))
    assert after['decision']=='ABSTAIN_NO_SYMPTOM_LOG_ANCHOR'
    assert run(mem,log('飞轮电流升高'),snapshot=4)['decision']=='PROVISIONAL_EVIDENCE_BACKED_HYPOTHESIS'


def test_public_reference_multi_hop_is_never_approved_root_cause():
    r=evaluate_multihop(fault_id='ADCS-WHEEL-SAT',fault_name='反作用飞轮动量饱和',
        equipment_scope='SAT_X_ADCS_V1',log_lines=log('轮速增大'),reviewed_edges=[],
        public_graph_path=default_public_graph_path(),max_hops=3)
    assert r['public_seed_lookup']['status']=='MATCHED_UNREVIEWED_ONLY'
    assert r['public_reference_paths']
    assert all(not p['eligible_as_hypothesis_path'] for p in r['public_reference_paths'])
    assert r['decision']=='ABSTAIN_NO_APPROVED_PATH'
    unknown=evaluate_multihop(fault_id='UNKNOWN_999',fault_name='火星引擎意念控制故障',
        equipment_scope='SAT_X_ADCS_V1',log_lines=log('故障'),reviewed_edges=[],
        public_graph_path=default_public_graph_path())
    assert unknown['decision']=='ABSTAIN_NO_APPROVED_PATH'
    assert not unknown['public_reference_paths']


def test_invalid_source_provenance_rejected_without_missing_edge_hallucination(tmp_path):
    mem,_=reviewed(tmp_path)
    edges=mem.approved_graph_for_reasoning(equipment_scope=SCOPE,snapshot=mem.snapshot(),exclude_source_group_id=JOB2)
    for item in edges:
        if item['relation_type']=='MANIFESTS_AS':
            item['provenance']['source_sha256']='not-a-sha'
    r=evaluate_multihop(fault_id=FAULT,fault_name=FAULT,equipment_scope=SCOPE,
        log_lines=log('飞轮电流升高'),reviewed_edges=edges)
    assert 'INVALID_APPROVED_RELATION_PROVENANCE' in r['knowledge_gap_codes']
    assert r['decision']!='PROVISIONAL_EVIDENCE_BACKED_HYPOTHESIS'


def test_gap_queue_idempotent_protected_and_never_publishes(tmp_path):
    mem,_=reviewed(tmp_path)
    values=[{'fault_id':FAULT,'knowledge_gap_codes':['MISSING_INDEPENDENT_SYMPTOM_OBSERVATION']}]
    first=mem.stage_reasoning_gaps(source_group_id=JOB2,equipment_scope=SCOPE,candidates=values,
        source_digests=['2'*64],snapshot=4)
    again=mem.stage_reasoning_gaps(source_group_id=JOB2,equipment_scope=SCOPE,candidates=values,
        source_digests=['2'*64],snapshot=4)
    assert first==again
    assert len(mem.pending_reasoning_gaps())==1
    with pytest.raises(PermissionError):
        mem.resolve_reasoning_gap(gap_id=first[0]['gap_id'],reviewer='unauthorized',resolution_ref='manual-QA-034')
    cleared=mem.resolve_reasoning_gap(gap_id=first[0]['gap_id'],reviewer='expert',resolution_ref='manual-QA-034')
    assert not cleared['graph_promoted'] and not mem.pending_reasoning_gaps()
    with pytest.raises(ValueError):
        mem.resolve_reasoning_gap(gap_id=first[0]['gap_id'],reviewer='expert',resolution_ref='manual-QA-034')


def test_gap_api_auth_and_csrf(tmp_path,monkeypatch):
    monkeypatch.setenv('ASTROGRAPH_PLATFORM_ROOT',str(tmp_path/'platform'))
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEW_TOKEN','test-only-long-credential-for-review-123456')
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEWER','expert')
    monkeypatch.setenv('ASTROGRAPH_MATERIAL_REVIEWERS','expert')
    client=TestClient(create_app(),base_url='http://localhost')
    store=MaterialCaseMemory(tmp_path/'platform'/'material_knowledge'/'cases.sqlite3',reviewers=('expert',))
    gap_id=store.stage_reasoning_gaps(source_group_id=JOB1,equipment_scope=SCOPE,
        candidates=[{'fault_id':'UNKNOWN_FAULT','knowledge_gap_codes':['MISSING_APPROVED_FAULT_RELATIONS']}],
        source_digests=[],snapshot=0)[0]['gap_id']
    endpoint='/v1/platform/material-knowledge/reasoning/gaps'
    assert client.get(endpoint).status_code==403
    assert client.post(endpoint+'/'+gap_id+'/resolve',json={'resolution_ref':'review-QA-002'}).status_code==403
    login=client.post('/v1/platform/material-knowledge/review-session',json={'credential':'test-only-long-credential-for-review-123456'})
    assert login.status_code==200
    csrf=login.json()['csrf']
    assert len(client.get(endpoint).json()['knowledge_gaps'])==1
    assert len(client.get('/v1/platform/material-knowledge/pending').json()['reasoning_gaps'])==1
    assert client.post(endpoint+'/'+gap_id+'/resolve',json={'resolution_ref':'review-QA-002'}).status_code==403
    response=client.post(endpoint+'/'+gap_id+'/resolve',headers={'x-material-csrf':csrf},json={'resolution_ref':'review-QA-002'})
    assert response.status_code==200,response.text
    assert response.json()['graph_promoted'] is False
    assert client.get(endpoint).json()['knowledge_gaps']==[]
    assert client.post(endpoint+'/'+gap_id+'/resolve',headers={'x-material-csrf':csrf},json={'resolution_ref':'review-QA-002'}).status_code==422


def test_ui_and_report_have_explicit_abstention_contract():
    root=Path(__file__).resolve().parents[1]
    js=(root/'src/astrograph/resources/platform_report.js').read_text(encoding='utf-8')
    expert=(root/'src/astrograph/resources/platform_graph_workbench.js').read_text(encoding='utf-8')
    html=(root/'src/astrograph/resources/platform_ui.html').read_text(encoding='utf-8')
    assert '2～3 跳证据约束推理' in js and 'public_reference_paths' in js
    assert 'reasoning_gaps' in expert and 'resolveReasoningGap' in expert
    assert 'data-review-kind="reasoning_gaps"' in html


def test_complete_platform_run_uses_reviewed_multihop_and_withholds_llm_narrative(tmp_path):
    from astrograph.integrations.upload_report import UploadReportJobStore
    repo=tmp_path/'run-root'
    mem,_=reviewed(repo)
    store=UploadReportJobStore(repo)
    files=Path(__file__).resolve().parents[1]/'examples/result_log_review'
    manifest=json.loads((files/'analysis_input.json').read_text(encoding='utf-8'))
    manifest['equipment_scope']=SCOPE
    blobs={'analysis_input.json':json.dumps(manifest).encode(),
           'diagnosis_result.json':(files/'diagnosis_result.json').read_bytes(),
           'reference.md':(files/'reference.md').read_bytes(),
           'operations.log':'2026-10-08T06:10:00Z WARN 飞轮电流升高，请核查。'.encode('utf-8')}
    job=store.create('故障原因及辅助处置')['job_id']
    for name,data in blobs.items():
        store.put_file(job,name,data)
    assert store.preflight(job)['can_start']
    report=store.run(job)
    assert report['status']=='SUCCEEDED'
    reasoning=report['analysis']['material_reasoning']
    assert reasoning['candidates'][0]['multi_hop']['decision']=='PROVISIONAL_EVIDENCE_BACKED_HYPOTHESIS'
    assert reasoning['answer_policy']['provisional_hypotheses_allowed']
    knowledge=report['analysis']['knowledge_enrichment']
    assert knowledge['final_explanation']['source']=='DETERMINISTIC_MATERIAL_MULTIHOP_POLICY'
    assert knowledge['material_answer_guard']['unverified_generated_narrative_exposed'] is False
    assert '证据约束多跳推理' in store.report_markdown_path(job).read_text(encoding='utf-8')
    assert 'reasoning_gap_learning' in reasoning


def test_llm_hallucination_draft_is_not_user_facing():
    from astrograph.integrations.material_multihop import apply_material_answer_guard
    draft={'generation':{'status':'SUCCEEDED','explanation':{'summary':'根因已经确诊，可以下发指令。'}},
           'final_explanation':{'explanation':{'summary':'根因已经确诊，可以下发指令。'}}}
    result=apply_material_answer_guard(draft,{'answer_policy':{'all_candidates_abstained':True},'candidates':[]})
    assert '无法确定根因' in result['final_explanation']['explanation']['summary']
    assert '根因已经确诊' not in result['final_explanation']['explanation']['summary']
    assert result['generation']['explanation'] is None  # no unchecked claims in public API JSON
    assert result['generation']['unverified_explanation_withheld_from_report']
    assert len(result['generation']['unverified_explanation_sha256']) == 64
    assert '根因已经确诊' not in json.dumps(result,ensure_ascii=False)
    assert result['material_answer_guard']['status']=='ABSTAIN'
    assert result['material_answer_guard']['automatic_control_authorized'] is False


def test_validated_cited_llm_result_is_visible_for_expert_review():
    from astrograph.integrations.material_multihop import apply_material_answer_guard
    draft={'generation':{
        'status':'SUCCEEDED',
        'citation_reference_validation':'PASS',
        'explanation':{
            'candidate_fault_id':'RW_FRICTION',
            'abstained':False,
            'summary':'模型原始摘要不直接作为确诊结论。',
            'claims':[{'kind':'HYPOTHESIS','text':'摩擦增大值得优先排查。',
                       'citation_ids':['MLOG-001','KREF-001']}],
            'recommended_review_actions':['核对轮速、指令与电流变化。'],
            'limitations':['仍需领域专家结合实物状态确认。'],
        },
    },'final_explanation':{'explanation':{'summary':'legacy'}}}
    result=apply_material_answer_guard(
        draft,{'answer_policy':{'all_candidates_abstained':False},'candidates':[]})
    visible=result['model_explanation']
    assert visible['status']=='AVAILABLE_REQUIRES_EXPERT_REVIEW'
    assert visible['claims'][0]['citation_ids']==['MLOG-001','KREF-001']
    assert visible['recommended_review_actions']==['核对轮速、指令与电流变化。']
    assert visible['automatic_control_authorized'] is False
    assert result['material_answer_guard']['validated_structured_model_output_exposed'] is True
    assert result['final_explanation']['source']=='DETERMINISTIC_MATERIAL_MULTIHOP_POLICY'


def test_llm_raw_provider_receipts_and_errors_are_withheld_from_material_report():
    from astrograph.integrations.material_multihop import apply_material_answer_guard
    suspicious='未经核实：执行虚假的航天器控制命令'
    draft={'generation':{
        'status':'FAILED','explanation':None,
        'error':{'type':'ParsingError','message':suspicious},
        'backend_receipt':{'provider_response':suspicious},
        'arbitrary_provider_field':{'suggestion':suspicious},
        'semantic_hallucination_assessment':'NOT_EVALUATED',
    },'deterministic_synthesis':{'explanation':{'summary':suspicious}},
      'final_explanation':{'explanation':{'summary':suspicious}}}
    result=apply_material_answer_guard(draft,{'answer_policy':{'all_candidates_abstained':True},'candidates':[]})
    assert suspicious not in json.dumps(result,ensure_ascii=False)
    assert 'error' not in result['generation']
    assert 'backend_receipt' not in result['generation']
    assert result['generation']['error_details_withheld_from_report']
    assert result['generation']['backend_receipt_withheld_from_report']
    assert result['generation']['status']=='FAILED'
    assert result['final_explanation']['source']=='DETERMINISTIC_MATERIAL_MULTIHOP_POLICY'
