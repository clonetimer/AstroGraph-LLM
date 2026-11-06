"""R7: document-anchored causal *claims*, not sensor-confirmed failure roots.

This generic pump case is deliberately not the user's already-disclosed SADA
failure: outcome labels must stay outside model/retrieval/test sources.
"""
from dataclasses import dataclass
from pathlib import Path
import pytest

from astrograph.integrations.material_graph_builder import (
    GraphChunkExtraction,validate_fragment,causal_quotation_defects,
)
from astrograph.integrations.material_multihop import evaluate_causal_chains
from astrograph.integrations.material_case_memory import MaterialCaseMemory
from astrograph.integrations.material_graph_release import verify_graph_release,MaterialGraphReleaseError

SCOPE='SAT_PUMP_TEST_V1'
FAULT='PUMP_FAULT_DEMO'
JOB='JOB-'+'E'*24

@dataclass
class Chunk:
    text: str = (
        'PUMP_FAULT_DEMO 导致 seal wear。\n'
        'seal wear 导致 lubricant contamination。\n'
        'lubricant contamination 导致 coolant degradation。\n'
        'coolant degradation 导致 bearing hot。\n'
        '如果 cooling valve closed，seal wear 可能导致 pressure fluctuation。\n'
        'seal wear 与 lubricant contamination 相关。\n'
    )
    chunk_id: str='PUMP-CHUNK'
    source_sha256: str='e'*64
    filename: str='synthetic-pump.md'
    source_locators: tuple=('line:1-6',)
    section_path: tuple=('pump',)


def proposal_data():
    entities=[('f','FAULT_MODE',FAULT),('a','FAILURE_MECHANISM','seal wear'),
              ('b','FAILURE_MECHANISM','lubricant contamination'),
              ('c','FAILURE_MECHANISM','coolant degradation'),
              ('d','SYMPTOM','bearing hot'),
              ('z','SYMPTOM','pressure fluctuation')]
    edges=[('f','a','CAUSES','PUMP_FAULT_DEMO 导致 seal wear。'),
           ('a','b','CAUSES','seal wear 导致 lubricant contamination。'),
           ('b','c','CAUSES','lubricant contamination 导致 coolant degradation。'),
           ('c','d','CAUSES','coolant degradation 导致 bearing hot。'),
           ('a','z','CAUSES','如果 cooling valve closed，seal wear 可能导致 pressure fluctuation。'),
           ('a','b','ASSOCIATED_WITH','seal wear 与 lubricant contamination 相关。')]
    return {'entities':[{'local_id':i,'entity_type':t,'name':n} for i,t,n in entities],
            'relations':[{'source_local_id':src,'target_local_id':dst,
                          'relation_type':rel,'evidence_quote':quote,
                          **({'condition_quote':'如果 cooling valve closed'} if dst=='z' else {})}
                         for src,dst,rel,quote in edges]}


def validated():
    return validate_fragment(GraphChunkExtraction.model_validate(proposal_data()),Chunk(),source_group_id=JOB)


def approved(tmp_path, include_conditional=True):
    rows,reject=validated()
    assert not reject and len(rows)==6
    db=MaterialCaseMemory(tmp_path/'graph.sqlite',reviewers=('independent_engineer',))
    rows=[r for r in rows if include_conditional or r['target']['name']!='pressure fluctuation']
    db.stage_graph_proposals(source_group_id=JOB,proposals=rows)
    assert db.approved_graph_for_reasoning(equipment_scope=SCOPE,snapshot=0,exclude_source_group_id='')==[]
    for idx,item in enumerate(rows):
        db.decide_graph(proposal_id=item['proposal_id'], reviewer='independent_engineer',
                        decision='CONFIRM',truth_source_ref='Independent document review rev-1',
                        equipment_scope=SCOPE,expected_snapshot=idx)
    return db,rows


def review(edges, *, snapshot=10,scope=SCOPE,hops=4):
    return evaluate_causal_chains(fault_id=FAULT, fault_name='pump fault',
        equipment_scope=scope, reviewed_edges=edges, snapshot=snapshot,max_hops=hops)


def test_extract_cause_requires_explicit_causal_cue_and_both_source_endpoints():
    rows,rejected=validated()
    assert len(rows)==6 and not rejected
    assert any(e['relation_type']=='CAUSES' and e['claim_modality']=='HYPOTHETICAL_OR_CONDITIONAL'
               for e in rows)
    bad=proposal_data()
    bad['relations'][1]['evidence_quote']='seal wear 与 lubricant contamination。'
    badchunk=Chunk(text=Chunk.text+'\nseal wear 与 lubricant contamination。')
    rows,reject=validate_fragment(GraphChunkExtraction.model_validate(bad),badchunk,source_group_id=JOB)
    assert 'CAUSAL_CUE_MISSING' in [x['reason'] for x in reject]
    assert len(rows)==5
    wrong=proposal_data()
    wrong['relations'][1]['evidence_quote']='seal wear 导致 invisible fault。'
    rows,reject=validate_fragment(GraphChunkExtraction.model_validate(wrong),Chunk(),source_group_id=JOB)
    assert 'QUOTE_NOT_FOUND_IN_SOURCE' in [x['reason'] for x in reject]


def test_four_hop_chain_requires_opt_in_and_is_not_actual_diagnosis(tmp_path):
    db,rows=approved(tmp_path)
    edges=db.approved_graph_for_reasoning(equipment_scope=SCOPE,snapshot=db.snapshot(),exclude_source_group_id='')
    three=review(edges,hops=3)
    assert three['chains']==[]
    four=review(edges)
    assert four['status']=='APPROVED_REFERENCE_CAUSAL_CHAINS_UNVERIFIED_IN_CASE'
    assert len(four['chains'])==1
    path=four['chains'][0]
    assert path['hop_count']==4
    assert [h['relation_type'] for h in path['hops']]==['CAUSES']*4
    assert all(h['source_sha256']=='e'*64 and h['reviewer']=='independent_engineer'
               and h['source_locators'] for h in path['hops'])
    assert not path['causal_proof'] and not path['sensor_observation_verified']
    assert not four['current_root_cause_proven'] and not four['diagnostic_vote_allowed']
    assert not four['automatic_control_authorized']
    assert any('UNVERIFIED_CAUSAL_CONDITION' in b['reasons'] for b in four['blocked'])


def test_no_causal_proof_from_association_only_or_reversed_direction(tmp_path):
    db,_=approved(tmp_path)
    edges=db.approved_graph_for_reasoning(equipment_scope=SCOPE,snapshot=db.snapshot(),exclude_source_group_id='')
    assoc=[e for e in edges if e['relation_type']=='ASSOCIATED_WITH']
    assert review(assoc)['chains']==[]
    backward=[]
    for e in edges:
        if e['relation_type']=='CAUSES':
            backward.append({**e,'source':e['target'],'target':e['source']})
    assert review(backward)['chains']==[]
    assert review(edges,scope='SAT_OTHER_V1')['chains']==[]
    assert review(edges,scope='UNKNOWN')['status']=='ABSTAIN_EQUIPMENT_SCOPE_REQUIRED'


def test_negated_or_conflicted_cause_cannot_bridge_path(tmp_path):
    db,_=approved(tmp_path)
    edges=db.approved_graph_for_reasoning(equipment_scope=SCOPE,snapshot=db.snapshot(),exclude_source_group_id='')
    target=next(e for e in edges if e['source']['name']=='seal wear' and e['target']['name']=='lubricant contamination')
    contradictory={**target,'proposal_id':'MGR-'+'A'*24,'polarity':'NEGATED',
                    'evidence_quote':'seal wear 不导致 lubricant contamination。'}
    result=review([*edges,contradictory])
    assert result['chains']==[]
    assert any('CONFLICTED_CAUSAL_CLAIM' in b['reasons'] for b in result['blocked'])
    result=review([{**e,'polarity':'NEGATED'} if e['proposal_id']==target['proposal_id'] else e for e in edges])
    assert not result['chains']


def test_snapshot_retraction_isolation_and_unapproved_relations(tmp_path):
    db,rows=approved(tmp_path)
    all_edges=db.approved_graph_for_reasoning(equipment_scope=SCOPE,snapshot=db.snapshot(),exclude_source_group_id='')
    assert review(all_edges)['chains']
    assert not review(all_edges,snapshot=2)['chains']
    target=next(e for e in rows if e['target']['name']=='lubricant contamination')
    before=db.snapshot()
    db.retract_graph(proposal_id=target['proposal_id'],reviewer='independent_engineer',
        reason='New engineering test contradicts document claim',truth_source_ref='independent-reanalysis')
    after=db.approved_graph_for_reasoning(equipment_scope=SCOPE,snapshot=db.snapshot(),exclude_source_group_id='')
    old=db.approved_graph_for_reasoning(equipment_scope=SCOPE,snapshot=before,exclude_source_group_id='')
    assert not review(after,snapshot=db.snapshot())['chains']
    assert review(old,snapshot=before)['chains']
    raw,_=validated()
    assert not review(raw)['chains']


def test_cause_integrity_is_enforced_at_staging_and_read(tmp_path):
    db,_=approved(tmp_path)
    approved_edges=db.approved_graph_for_reasoning(equipment_scope=SCOPE,snapshot=db.snapshot(),exclude_source_group_id='')
    bad={**next(e for e in approved_edges if e['relation_type']=='CAUSES'),
         'evidence_quote':'seal wear 与 lubricant contamination 相关。'}
    assert 'CAUSAL_CUE_MISSING' in causal_quotation_defects(bad)
    assert not review([bad])['chains']
    raw,_=validated()
    staged={**next(e for e in raw if e['relation_type']=='CAUSES'),
            'evidence_quote':'This is merely a correlation'}
    with pytest.raises(ValueError,match='invalid sourced'):
        db.stage_graph_proposals(source_group_id=JOB,proposals=[staged])


def test_invalid_condition_or_lying_modality_fails_closed(tmp_path):
    db,_=approved(tmp_path)
    edges=db.approved_graph_for_reasoning(equipment_scope=SCOPE,snapshot=db.snapshot(),exclude_source_group_id='')
    conditional=next(e for e in edges if e.get('condition_quote'))
    assert conditional['claim_modality']=='HYPOTHETICAL_OR_CONDITIONAL'
    assert not review([conditional])['chains']
    tampered={**conditional,'claim_modality':'REFERENCE_ASSERTION_REQUIRES_REVIEW'}
    assert 'CONDITIONAL_CLAIM_MISCLASSIFIED' in causal_quotation_defects(tampered)
    assert not review([tampered])['chains']
    unanchored={**conditional,'condition_quote':'if unknown system OK'}
    assert 'CONDITION_NOT_ANCHORED_IN_QUOTE' in causal_quotation_defects(unanchored)
    assert not review([unanchored])['chains']


def test_scoped_export_and_release_validate_causal_integrity(tmp_path):
    db,_=approved(tmp_path)
    release=db.export_graph_release(equipment_scope=SCOPE,snapshot=db.snapshot(),output_root=tmp_path/'release')
    _,graph=verify_graph_release(release['directory'])
    assert any(e['relation_type']=='CAUSES' and e.get('condition_quote') for e in graph['relations'])


def test_causal_search_budget_is_reported_as_partial_not_complete(tmp_path):
    db,_=approved(tmp_path)
    edges=db.approved_graph_for_reasoning(equipment_scope=SCOPE,snapshot=db.snapshot(),exclude_source_group_id='')
    result=evaluate_causal_chains(fault_id=FAULT,fault_name='pump fault',
        equipment_scope=SCOPE,reviewed_edges=edges,snapshot=db.snapshot(),
        max_hops=4,max_chains=1)
    assert result['search_truncated']
    assert result['status']=='PARTIAL_REFERENCE_CAUSAL_CHAINS_SEARCH_TRUNCATED'
    assert result['chains'] and not result['current_root_cause_proven']
