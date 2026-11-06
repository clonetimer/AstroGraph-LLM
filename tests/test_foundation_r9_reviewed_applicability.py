"""R9 generic pump contexts. No disclosed SADA root-cause terms or labels."""
from dataclasses import dataclass
import json
import math

import pytest

from astrograph.integrations.material_applicability import (
    SCHEMA, assess_contexts, normalize_reviewed_applicability,
)
from astrograph.integrations.material_case_memory import MaterialCaseMemory
from astrograph.integrations.material_graph_builder import GraphChunkExtraction, validate_fragment
from astrograph.integrations.material_graph_release import verify_graph_release, MaterialGraphReleaseError
from astrograph.integrations.material_multihop import evaluate_causal_chains

SCOPE = 'SAT_PUMP_TEST_V1'
GROUP = 'JOB-' + 'F' * 24


def ctx(*, modes=('TRACKING',), temp=(10, 50, 'DEGC'), thermal='HOT', basis='independent bench test 2026'):
    return {'schema_version': SCHEMA, 'equipment_model': SCOPE,
            'operating_modes': list(modes), 'environment': {'THERMAL_REGIME': thermal},
            'parameter_ranges': [{'name': 'SHAFT_TEMP', 'unit': temp[2],
                                  'min': temp[0], 'max': temp[1]}],
            'review_basis_ref': basis}


def edge(i, src, dest, kind_src, kind_dst, *, applicability=None, polarity='AFFIRMATIVE'):
    a = {'proposal_id': 'MGR-' + str(i) * 24,
         'source_group_id': 'JOB-' + 'A' * 24,
         'source': {'entity_id': src[0], 'name': src[1], 'entity_type': kind_src, 'aliases': []},
         'target': {'entity_id': dest[0], 'name': dest[1], 'entity_type': kind_dst, 'aliases': []},
         'relation_type': 'CAUSES', 'polarity': polarity,
         'claim_modality': 'REFERENCE_ASSERTION_REQUIRES_REVIEW',
         'evidence_quote': (src[1] + (' 不导致 ' if polarity == 'NEGATED' else ' 导致 ') + dest[1]),
         'equipment_scope': SCOPE, 'review_status': 'INDEPENDENTLY_REVIEWED_REFERENCE_RELATION',
         'reviewer': 'independent_engineer', 'truth_source_ref': 'bench records',
         'snapshot': 3,
         'provenance': {'source_sha256': str(i) * 64, 'source_locators': ['line:1']}}
    if applicability is not None:
        a['reviewed_applicability'] = applicability
    return a


def edges(a=None, b=None):
    return [edge(1, ('f', 'PUMP_FAULT_1'), ('m', 'shaft wear'), 'FAULT_MODE', 'FAILURE_MECHANISM', applicability=a),
            edge(2, ('m', 'shaft wear'), ('s', 'bearing hot'), 'FAILURE_MECHANISM', 'SYMPTOM', applicability=b)]


def review(records):
    return evaluate_causal_chains(fault_id='PUMP_FAULT_1', fault_name='pump fault',
                                  equipment_scope=SCOPE, reviewed_edges=records, snapshot=4)


def test_legacy_unannotated_edges_remain_retrievable_but_context_unknown():
    out = review(edges())
    assert len(out['chains']) == 1
    assert out['chains'][0]['reviewed_context_consistency']['status'] == 'INCOMPLETE_REVIEWED_CONTEXT'
    assert out['chains'][0]['joint_applicability_verified'] is False
    assert len(out['chains'][0]['reviewed_context_consistency']['unknown_edge_ids']) == 2


def test_compatible_ranges_normalize_celsius_and_kelvin_without_flight_proof():
    out = review(edges(ctx(temp=(10, 60, 'DEGC')), ctx(temp=(310, 320, 'K'))))
    assert len(out['chains']) == 1
    audit = out['chains'][0]['reviewed_context_consistency']
    assert audit['status'] == 'REVIEWED_CONSTRAINTS_MUTUALLY_COMPATIBLE_NOT_FLIGHT_VERIFIED'
    assert audit['intersection']['parameter_ranges']['SHAFT_TEMP']['min'] == 310
    assert audit['intersection']['parameter_ranges']['SHAFT_TEMP']['max'] == 320
    assert not audit['flight_applicability_verified']
    assert not out['diagnostic_vote_allowed']


def test_disjoint_operating_modes_prevent_path_splice():
    out = review(edges(ctx(modes=('TRACKING',)), ctx(modes=('STANDBY',))))
    assert not out['chains']
    assert any('OPERATING_MODE_DISJOINT' in x['reasons'] for x in out['blocked'])


def test_environment_and_numerical_disjoint_ranges_prevent_splice():
    thermal = review(edges(ctx(thermal='HOT'), ctx(thermal='COLD')))
    assert not thermal['chains']
    assert any('ENVIRONMENT_CONFLICT:THERMAL_REGIME' in x['reasons'] for x in thermal['blocked'])
    temp = review(edges(ctx(temp=(5, 10, 'DEGC')), ctx(temp=(285, 300, 'K'))))
    assert not temp['chains']
    assert any('PHYSICAL_RANGE_DISJOINT:SHAFT_TEMP' in x['reasons'] for x in temp['blocked'])


def test_nonfinite_and_unknown_unit_rejected_at_approval_boundary():
    for bad in (ctx(temp=(0, math.nan, 'DEGC')),
                ctx(temp=(0, 10, 'GIBBERISH')),
                ctx(temp=(20, 10, 'K'))):
        with pytest.raises(ValueError):
            normalize_reviewed_applicability(bad, equipment_scope=SCOPE)
        assert not review(edges(bad, ctx()))['chains']


def test_different_model_is_blocked_even_with_reviewed_relation():
    alt = ctx(); alt['equipment_model'] = 'OTHER_MODEL'
    out = review(edges(alt, ctx()))
    assert not out['chains']
    assert any('INVALID_REVIEWED_APPLICABILITY' in x['reasons'] for x in out['blocked'])


def test_disjoint_negation_does_not_cancel_distinct_condition_claims():
    positive, second = edges(ctx(modes=('TRACKING',)), ctx(modes=('TRACKING',)))
    negative = edge(3, ('f', 'PUMP_FAULT_1'), ('m', 'shaft wear'), 'FAULT_MODE',
                    'FAILURE_MECHANISM', applicability=ctx(modes=('STANDBY',)), polarity='NEGATED')
    out = review([positive, second, negative])
    assert len(out['chains']) == 1
    assert not out['current_root_cause_proven']
    negative['reviewed_applicability'] = ctx(modes=('TRACKING',))
    assert not review([positive, second, negative])['chains']


@dataclass
class Chunk:
    text: str = 'PUMP_FAULT_1 导致 shaft wear。'
    chunk_id: str = 'SYN-CHUNK'
    source_sha256: str = 'f' * 64
    filename: str = 'synthetic-pump.md'
    source_locators: tuple = ('line:1',)
    section_path: tuple = ('test',)


def test_independent_review_context_persists_in_snapshot_and_hashed_release_format(tmp_path):
    fragment = GraphChunkExtraction.model_validate({
        'entities': [{'local_id':'f','name':'PUMP_FAULT_1','entity_type':'FAULT_MODE'},
                     {'local_id':'m','name':'shaft wear','entity_type':'FAILURE_MECHANISM'}],
        'relations': [{'source_local_id':'f','target_local_id':'m', 'relation_type':'CAUSES',
                       'evidence_quote':'PUMP_FAULT_1 导致 shaft wear。'}]})
    proposals, rejected = validate_fragment(fragment, Chunk(), source_group_id=GROUP)
    assert len(proposals) == 1 and rejected == []
    db = MaterialCaseMemory(tmp_path/'case.sqlite', reviewers=('engineer',))
    forged = {**proposals[0], 'reviewed_applicability': ctx()}
    with pytest.raises(ValueError, match='invalid sourced'):
        db.stage_graph_proposals(source_group_id=GROUP, proposals=[forged])
    db.stage_graph_proposals(source_group_id=GROUP, proposals=proposals)
    with pytest.raises(ValueError, match='scope mismatch'):
        bad = ctx(); bad['equipment_model'] = 'OTHER_MODEL'
        db.decide_graph(proposal_id=proposals[0]['proposal_id'], reviewer='engineer',
                        decision='CONFIRM',truth_source_ref='bench report',equipment_scope=SCOPE,
                        expected_snapshot=0, reviewed_applicability=bad)
    db.decide_graph(proposal_id=proposals[0]['proposal_id'], reviewer='engineer',
                    decision='CONFIRM',truth_source_ref='bench report',equipment_scope=SCOPE,
                    expected_snapshot=0, reviewed_applicability=ctx())
    records = db.approved_graph_for_reasoning(equipment_scope=SCOPE, snapshot=1, exclude_source_group_id='')
    assert records[0]['reviewed_applicability']['parameter_ranges'][0]['unit'] == 'K'
    assert records[0]['reviewed_applicability']['review_basis_ref'] == 'independent bench test 2026'
    release = db.export_graph_release(equipment_scope=SCOPE, snapshot=1,output_root=tmp_path/'release')
    _, graph = verify_graph_release(release['directory'])
    assert graph['relations'][0]['reviewed_applicability']['operating_modes'] == ['TRACKING']
    graph_path = tmp_path/'release'/f'{SCOPE}-SNAPSHOT-00000001'/'graph.json'
    # Mutating exported context should invalidate the corresponding manifest checksum.
    raw = json.loads(graph_path.read_text())
    raw['relations'][0]['reviewed_applicability']['operating_modes'] = ['STANDBY']
    graph_path.write_text(json.dumps(raw))
    with pytest.raises(MaterialGraphReleaseError, match='checksum'):
        verify_graph_release(release['directory'])
