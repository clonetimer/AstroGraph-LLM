"""R8 negative-direction and cross-document causal-comparison contracts.

Generic laboratory pump fixture; this suite contains no leaked SADA outcome.
"""
from dataclasses import dataclass

from astrograph.integrations.material_graph_builder import (
    GraphChunkExtraction, causal_quotation_defects, validate_fragment,
)
from astrograph.integrations.material_multihop import evaluate_causal_chains
from astrograph.integrations.material_reasoning import build_competing_hypothesis_review

SCOPE = 'GENERIC_PUMP_TEST_V2'


@dataclass
class Chunk:
    text: str
    source_sha256: str = 'a' * 64
    chunk_id: str = 'GENERIC-CHUNK'
    filename: str = 'synthetic-pump.md'
    source_locators: tuple[str, ...] = ('paragraph:1',)
    section_path: tuple[str, ...] = ('pump',)


def extract(text, source, target, src_type='FAILURE_MECHANISM', tgt_type='FAILURE_MECHANISM', *, rel='CAUSES'):
    payload = GraphChunkExtraction.model_validate({
        'entities': [
            {'local_id':'a','entity_type':src_type,'name':source},
            {'local_id':'b','entity_type':tgt_type,'name':target},
        ],
        'relations': [{'source_local_id':'a','relation_type':rel,
                       'target_local_id':'b','evidence_quote':text}],
    })
    return validate_fragment(payload, Chunk(text=text), source_group_id='JOB-'+'F'*24)


def reviewed(src_id, dst_id, src_name, dst_name, src_type, dst_type, *, sha, group, quote=None, edge_id=None):
    return {
        'proposal_id':edge_id or 'MGR-'+(sha[0]*24),
        'source_group_id':group,
        'source':{'entity_id':src_id, 'entity_type':src_type,'name':src_name,'aliases':[]},
        'target':{'entity_id':dst_id, 'entity_type':dst_type,'name':dst_name,'aliases':[]},
        'relation_type':'CAUSES','polarity':'AFFIRMATIVE',
        'claim_modality':'REFERENCE_ASSERTION_REQUIRES_REVIEW',
        'equipment_scope':SCOPE,'review_status':'INDEPENDENTLY_REVIEWED_REFERENCE_RELATION',
        'reviewer':'independent_engineer','truth_source_ref':'reviewed-doc-v1',
        'snapshot':3,
        'evidence_quote':quote or (src_name+' 导致 '+dst_name),
        'provenance':{'source_sha256':sha,'source_locators':['paragraph:1']},
    }


def chain_edges(multi_source):
    sha1, sha2 = 'a'*64, ('b'*64 if multi_source else 'a'*64)
    return [
        reviewed('f','m','PUMP_FAULT_1','shaft wear','FAULT_MODE','FAILURE_MECHANISM',
                 sha=sha1,group='JOB-'+'A'*24,edge_id='MGR-'+'1'*24),
        reviewed('m','s','shaft wear','bearing hot','FAILURE_MECHANISM','SYMPTOM',
                 sha=sha2,group='JOB-'+('B' if multi_source else 'A')*24,
                 edge_id='MGR-'+'2'*24),
    ]


def causal(edges):
    return evaluate_causal_chains(fault_id='PUMP_FAULT_1',fault_name='pump fault',
            equipment_scope=SCOPE,reviewed_edges=edges,snapshot=5,max_hops=3)


def test_explicit_reversed_english_passive_cannot_be_staged():
    source, target = 'seal wear', 'coolant contamination'
    quote = 'seal wear is caused by coolant contamination.'
    rows, rejected = extract(quote, source, target)
    assert rows == []
    assert [r['reason'] for r in rejected] == ['EXPLICIT_CAUSAL_DIRECTION_REVERSED']
    # Inverse ordering correctly expresses the same literal statement.
    rows, rejected = extract(quote,target,source)
    assert not rejected and len(rows) == 1


def test_explicit_reversed_chinese_and_active_english():
    ch = 'shaft wear 由 lubricant loss 导致。'
    assert extract(ch,'shaft wear','lubricant loss')[1][0]['reason'] == 'EXPLICIT_CAUSAL_DIRECTION_REVERSED'
    assert extract(ch,'lubricant loss','shaft wear')[0]
    en = 'lubricant loss causes shaft wear.'
    assert extract(en,'shaft wear','lubricant loss')[1][0]['reason'] == 'EXPLICIT_CAUSAL_DIRECTION_REVERSED'
    assert extract(en,'lubricant loss','shaft wear')[0]
    zh = 'lubricant loss 导致 shaft wear。'
    assert extract(zh,'shaft wear','lubricant loss')[1][0]['reason'] == 'EXPLICIT_CAUSAL_DIRECTION_REVERSED'


def test_reverse_quote_never_passes_approved_graph_reader_even_if_label_forged():
    edges=chain_edges(False)
    bad = {**edges[0], 'evidence_quote':'PUMP_FAULT_1 is caused by shaft wear'}
    assert 'EXPLICIT_CAUSAL_DIRECTION_REVERSED' in causal_quotation_defects(bad)
    result=causal([bad,edges[1]])
    assert result['chains'] == []
    assert any('EXPLICIT_CAUSAL_DIRECTION_REVERSED' in row['reasons'] for row in result['blocked'])


def test_multidocument_chain_carries_context_uncertainty_and_full_provenance():
    result=causal(chain_edges(True))
    assert result['status']=='APPROVED_REFERENCE_CAUSAL_CHAINS_UNVERIFIED_IN_CASE'
    assert len(result['chains']) == 1
    chain=result['chains'][0]
    assert chain['hop_count'] == 2 and chain['structural_claim_chain_valid']
    assert chain['cross_document_mechanism_splice']
    assert chain['joint_applicability_status']=='CROSS_DOCUMENT_CONTEXT_REQUIRES_INDEPENDENT_VALIDATION'
    assert chain['reference_document_sha256'] == ['a'*64,'b'*64]
    assert len(chain['reference_source_group_ids']) == 2
    assert not chain['joint_applicability_verified']
    assert not chain['causal_proof'] and not chain['current_root_cause_proven']
    assert not result['diagnostic_vote_allowed']


def test_single_document_chain_is_not_flight_verified_either():
    result=causal(chain_edges(False))
    chain=result['chains'][0]
    assert not chain['cross_document_mechanism_splice']
    assert chain['joint_applicability_status']=='SINGLE_DOCUMENT_CONTEXT_NOT_FLIGHT_VERIFIED'
    assert not chain['joint_applicability_verified']


def item(fid, symptom_chains, *, truncated=False):
    return {
        'fault_id':fid,'fault_name':fid,
        'log_evidence_matrix':{'supporting_ids':[],'contradicting_ids':[]},
        'reference_retrieval':{},'bidirectional_graph_review':{},
        'multi_hop': {'causal_chain_review':{
            'chains': symptom_chains,
            'search_truncated':truncated,
        }},
    }


def path(pid,symptom,*,cross=False):
    return {'path_id':pid,'terminal_symptom_id':symptom,
            'structural_claim_chain_valid':True,
            'cross_document_mechanism_splice':cross}


def test_hypothesis_comparison_distinguishes_shared_from_candidate_specific_without_voting():
    a=item('FAULT_A',[path('p1','shared'),path('p2','specific_a',cross=True),path('p3','specific_a')])
    b=item('FAULT_B',[path('p4','shared'),path('p5','specific_b')])
    result=build_competing_hypothesis_review([a,b],[])
    diff=result['causal_path_differential']
    assert diff['status']=='REFERENCE_PATHS_FOR_COMPARISON'
    assert diff['shared_reference_symptom_ids']==['shared']
    rowa,rowb=diff['candidates']
    assert [x['symptom_id'] for x in rowa['candidate_specific_reference_symptoms']]==['specific_a']
    assert rowa['candidate_specific_reference_symptoms'][0]['chain_ids']==['p2','p3']
    assert rowa['candidate_specific_reference_symptoms'][0]['cross_document_context_unverified']
    assert rowa['shared_reference_symptom_ids']==['shared']
    assert rowb['candidate_specific_reference_symptoms'][0]['symptom_id']=='specific_b'
    assert [r['fault_id'] for r in diff['candidates']]==['FAULT_A','FAULT_B']
    assert not diff['diagnostic_vote_allowed'] and not diff['current_root_cause_proven']
    assert diff['candidate_specific_only_within_supplied_frozen_set']
    assert diff['no_counterfactual_causation_inferred']


def test_truncated_or_missing_paths_do_not_masquerade_as_negative_evidence():
    a=item('FAULT_A',[],truncated=True)
    b=item('FAULT_B',[path('p','x')])
    diff=build_competing_hypothesis_review([a,b],[])['causal_path_differential']
    assert diff['status']=='PARTIAL_REFERENCE_COVERAGE'
    assert diff['truncated_candidate_fault_ids']==['FAULT_A']
    assert diff['candidates'][0]['candidate_specific_reference_symptoms']==[]
    assert diff['candidates'][0]['missing_reference_or_observation_is_not_counterevidence']
    assert not diff['candidates'][1]['current_fault_observation_verified']


def test_reused_entity_id_with_different_meaning_cannot_join_documents():
    edges=chain_edges(True)
    tampered = {**edges[1],
                'source':{**edges[1]['source'],'name':'unrelated bearing defect'}}
    result=causal([edges[0],tampered])
    assert result['chains']==[]
    assert sum('INCONSISTENT_ENTITY_IDENTITY_ACROSS_REFERENCES' in b['reasons']
               for b in result['blocked']) == 2


def test_unrelated_equipment_cannot_supply_missing_causal_hop():
    edges=chain_edges(False)
    edges[1]={**edges[1],'equipment_scope':'OTHER_MODEL_TEST'}
    result=causal(edges)
    assert not result['chains']
    assert any('MODEL_SCOPE_MISMATCH' in b['reasons'] for b in result['blocked'])


def test_causal_differential_reaches_llm_without_new_candidate_ids(tmp_path):
    from astrograph.integrations.platform_knowledge import (
        PlatformKnowledgeEnrichmentService, PlatformKnowledgeConfig,
        PlatformKnowledgeExplanation,
    )

    class HostSpyBackend:
        def __init__(self):
            self.payload=None
            self.last_call_metadata={}

        def generate_structured(self, *, agent_id, output_model, payload):
            self.payload=payload
            return PlatformKnowledgeExplanation(candidate_fault_id='FAULT_A',abstained=True,
                                                summary='Reference comparison only',claims=[])

    backend=HostSpyBackend()
    service=PlatformKnowledgeEnrichmentService(
        PlatformKnowledgeConfig(a2o_enabled=False),llm_backend=backend)
    a=item('FAULT_A',[path('p1','shared'),path('p2','symptom_a')])
    b=item('FAULT_B',[path('p3','shared')])
    context={
        'candidates':[a,b],
        'competing_hypothesis_review':build_competing_hypothesis_review([a,b],[]),
    }
    service.enrich(task='Compare two supplied hypotheses',
        diagnosis_snapshot={'candidate_fault_id':'FAULT_A','evidence':[]},
        document_paths=[],output_root=tmp_path/'knowledge',material_review_context=context)
    transmitted=backend.payload['material_reasoning']['competing_hypothesis_review']['causal_path_differential']
    assert [row['fault_id'] for row in transmitted['candidates']]==['FAULT_A','FAULT_B']
    assert transmitted['shared_reference_symptom_ids']==['shared']
    assert transmitted['candidates'][0]['candidate_specific_reference_symptoms'][0]['symptom_id']=='symptom_a'
    assert not transmitted['diagnostic_vote_allowed']
    assert 'not proof that other faults cannot cause it' in backend.payload['task'] or 'not proof that other faults cannot cause it' in str(backend.payload)
