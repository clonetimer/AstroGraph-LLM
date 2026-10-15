"""R4 safety regression: disagreement, abstention, and literal-source gating."""
from astrograph.retrieval.material_claim_guard import audit_material_claims
from astrograph.integrations.material_reasoning import build_competing_hypothesis_review
from astrograph.integrations.material_multihop import apply_material_answer_guard


def reasoning():
    return {
        'answer_policy': {'all_candidates_abstained': False},
        'candidates': [{
            'fault_id': 'RW_FRICTION', 'fault_name': 'wheel friction',
            'log_evidence_matrix': {
                'events': [{'evidence_id': 'MLOG-00001', 'excerpt': 'motor current high', 'assessment':'POSSIBLE_SUPPORTING_CUE'}],
                'supporting_ids': ['MLOG-00001'], 'contradicting_ids': ['MLOG-00002']},
            'reference_retrieval': {
                'hits': [{'knowledge_id': 'REF-001', 'quote': 'In test mode, friction may increase motor current.',
                          'knowledge_validation_status': 'EXPERT_APPROVED_HISTORICAL_REFERENCE'}],
                'reviewed_counterevidence': {'items': [{'edge_id': 'EDGE-NEG', 'polarity': 'NEGATED', 'claim_modality':'REFERENCE_ASSERTION_REQUIRES_REVIEW', 'quote':'not an expected consequence', 'status':'REVIEWED_NEGATIVE_RELATION'}]}},
            'bidirectional_graph_review': {'missing_symptom_cue_ids': ['SYMPTOM-A']},
            'contrastive_reference_review': {'competitor_fault_id': 'RW_POWER', 'primary_only_chunk_ids':['REF-001']},
        }, {
            'fault_id': 'RW_POWER', 'fault_name': 'wheel power issue',
            'log_evidence_matrix': {'supporting_ids':['MLOG-00001'], 'contradicting_ids':[]},
            'reference_retrieval': {'reviewed_counterevidence': {'items':[]}, 'hits':[]},
            'bidirectional_graph_review': {'missing_symptom_cue_ids':[]},
        }],
    }


def knowledge(claims):
    return {'generation': {
        'status': 'SUCCEEDED', 'citation_reference_validation':'PASS',
        'explanation': {'candidate_fault_id':'RW_FRICTION','abstained':False,
                        'summary':'Root cause proved — unsupported',
                        'claims':claims, 'recommended_review_actions':['execute shutdown now']}}}


def test_competing_review_keeps_shared_cues_and_absences_separate():
    review=build_competing_hypothesis_review(reasoning()['candidates'], [
        {'symptom_id':'SYMPTOM-B','candidate_fault_ids':['RW_FRICTION','RW_POWER']}])
    assert review['status']=='READY_FOR_EXPERT_COMPARISON'
    assert review['candidates'][0]['approved_negative_relation_ids']==['EDGE-NEG']
    assert review['candidates'][0]['missing_expected_symptom_ids']==['SYMPTOM-A']
    assert review['candidates'][1]['shared_symptom_ids']==['SYMPTOM-B']
    assert review['missing_cue_is_not_negative_evidence']
    assert not review['diagnostic_vote_allowed']


def test_valid_id_does_not_validate_causal_paraphrase():
    claims=[
        {'kind':'MATERIAL_LOG_CUE','text':'motor current high proves wheel friction',
         'citation_ids':['MLOG-00001']},
        {'kind':'MATERIAL_REFERENCE','text':'In test mode, friction may increase motor current.',
         'citation_ids':['REF-001']},
        {'kind':'HYPOTHESIS','text':'Root cause proved, execute shutdown now',
         'citation_ids':['REF-001']},
    ]
    audit=audit_material_claims(knowledge(claims),reasoning(),knowledge(claims)['generation']['explanation'])
    assert [r['status'] for r in audit['claim_receipts']]==[
        'INFERENCE_OR_NONEXTRACTIVE_WITHHELD', 'EXTRACTIVE_MATCH_NOT_SEMANTIC_ENTAILMENT',
        'HYPOTHESIS_NOT_VALIDATED_WITHHELD']
    assert len(audit['published_extracts'])==1
    assert audit['published_extracts'][0]['text'].startswith('In test mode')
    assert not audit['natural_language_entailment_verified']


def test_no_partial_quote_that_removes_negation():
    r=reasoning()
    r['candidates'][0]['log_evidence_matrix']['events']=[
        {'evidence_id':'MLOG-00001','excerpt':'motor current is not high'}]
    claims=[{'kind':'MATERIAL_LOG_CUE','text':'motor current is high',
             'citation_ids':['MLOG-00001']}]
    outcome=audit_material_claims({},r,{'claims':claims})
    assert outcome['published_extracts']==[]


def test_public_guard_removes_unsupported_generated_actions_and_hypotheses():
    claims=[
        {'kind':'HYPOTHESIS','text':'Root cause proved, execute shutdown now','citation_ids':['REF-001']},
        {'kind':'MATERIAL_LOG_CUE','text':'motor current high','citation_ids':['MLOG-00001']},
    ]
    report=apply_material_answer_guard(knowledge(claims),reasoning())
    public=repr(report)
    assert 'execute shutdown now' not in public
    assert 'Root cause proved' not in public
    assert report['model_explanation']['status']=='EXTRACTIVE_REVIEW_ONLY'
    assert [c['text'] for c in report['model_explanation']['claims']]==['motor current high']
    assert report['generation']['claim_level_audit']['withheld_claim_count']==1
    assert report['material_answer_guard']['automatic_control_authorized'] is False


def test_reject_duplicate_source_id_with_conflicting_quotes():
    r=reasoning()
    r['candidates'][1]['reference_retrieval']['hits']=[
        {'knowledge_id':'REF-001','quote':'contradicting other source'}]
    claim={'kind':'MATERIAL_REFERENCE', 'text':'In test mode, friction may increase motor current.',
           'citation_ids':['REF-001']}
    result=audit_material_claims({},r,{'claims':[claim]})
    assert not result['published_extracts']
    assert result['claim_receipts'][0]['status']=='DUPLICATE_SOURCE_ID_CONFLICT_WITHHELD'


def test_unapproved_graph_path_never_masquerades_as_current_cause():
    claims=[{'kind':'GRAPH_ASSOCIATION', 'text':'wheel friction caused failure',
             'citation_ids':['MHP-001']}]
    audit=audit_material_claims({},reasoning(),{'claims':claims})
    assert not audit['published_extracts']
    assert audit['claim_receipts'][0]['status']=='ASSOCIATION_OR_HISTORY_NOT_CURRENT_CAUSE_WITHHELD'


def test_material_llm_receives_competing_hypotheses_not_just_top_candidate(tmp_path):
    from astrograph.integrations.platform_knowledge import (
        PlatformKnowledgeEnrichmentService, PlatformKnowledgeConfig,
        PlatformKnowledgeExplanation,
    )

    class StubBackend:
        def __init__(self):
            self.payload = None
            self.last_call_metadata = {}

        def generate_structured(self, *, agent_id, output_model, payload):
            self.payload = payload
            return PlatformKnowledgeExplanation(candidate_fault_id='RW_FRICTION', abstained=False,
                                                summary='This is an unverified candidate.', claims=[])

    backend = StubBackend()
    service = PlatformKnowledgeEnrichmentService(
        PlatformKnowledgeConfig(a2o_enabled=False), llm_backend=backend)
    context = reasoning()
    context['competing_hypothesis_review'] = build_competing_hypothesis_review(
        context['candidates'], [{'symptom_id': 'SYMPTOM-B',
                                 'candidate_fault_ids': ['RW_FRICTION', 'RW_POWER']}])
    outcome = service.enrich(task='Review fault possibilities',
        diagnosis_snapshot={'candidate_fault_id':'RW_FRICTION', 'evidence': []},
        document_paths=[], output_root=tmp_path/'knowledge', material_review_context=context)
    assert outcome['generation']['status']=='SUCCEEDED'
    comparison = backend.payload['material_reasoning']['competing_hypothesis_review']
    assert len(comparison['candidates']) == 2
    assert comparison['candidates'][0]['opposing_log_cue_ids']==['MLOG-00002']
    assert backend.payload['material_reasoning']['candidates'][0]['reviewed_counterclaims'][0]['edge_id']=='EDGE-NEG'
    assert 'missing symptoms as unobserved' in backend.payload['task']
    assert not comparison['diagnostic_vote_allowed']


def test_competition_instruction_is_bound_to_system_level_task_prompt():
    from astrograph.integrations.platform_knowledge import PlatformKnowledgeExplanation
    from astrograph.llm.task_prompts import structured_task_messages
    messages = structured_task_messages(PlatformKnowledgeExplanation, {
        'material_reasoning': {'competing_hypothesis_review': {}},
    })
    system_text=messages[0]['content']
    assert 'competing host candidates' in system_text
    assert 'Missing evidence is not counterevidence' in system_text
    assert 'untrusted DATA' in system_text
    assert messages[1]['role']=='user'


def test_forged_candidate_id_is_not_echoed_to_public_report():
    draft=knowledge([{'kind':'MATERIAL_LOG_CUE', 'text':'motor current high',
                      'citation_ids':['MLOG-00001']}])
    draft['generation']['explanation']['candidate_fault_id']='NEW_FAULT_FROM_MODEL'
    result=apply_material_answer_guard(draft,reasoning())
    assert result['model_explanation']['candidate_fault_id'] is None
    assert 'NEW_FAULT_FROM_MODEL' not in repr(result)