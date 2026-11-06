"""R10 synthetic pump timeline only; no case-specific SADA knowledge."""
from dataclasses import dataclass
from pathlib import Path
import json
import math

import pytest

from astrograph.integrations.material_applicability import SCHEMA as APP_SCHEMA
from astrograph.integrations.material_temporal import (
    SCHEMA, assess_temporal_causal_path, normalize_reviewed_temporal,
)
from astrograph.integrations.material_multihop import evaluate_causal_chains
from astrograph.integrations.material_case_memory import MaterialCaseMemory
from astrograph.integrations.material_graph_builder import GraphChunkExtraction, validate_fragment
from astrograph.integrations.material_graph_release import verify_graph_release, MaterialGraphReleaseError

SCOPE = 'SAT_PUMP_TEST_V1'


def app(mode, *, temp=(10, 30), thermal='HOT'):
    return {'schema_version': APP_SCHEMA, 'equipment_model': SCOPE,
            'operating_modes': [mode], 'environment': {'THERMAL_REGIME': thermal},
            'parameter_ranges': [{'name': 'SHAFT_TEMP', 'unit': 'DEGC', 'min': temp[0], 'max': temp[1]}],
            'review_basis_ref': 'independently verified test conditions'}


def temporal(stage, pre, post, *, process='PUMP_BENCH_SEQUENCE', lag=(2, 5), dwell=1):
    return {'schema_version': SCHEMA, 'equipment_model': SCOPE,
            'process_id': process, 'stage_index': stage,
            'pre_context': pre, 'post_context': post,
            'lag_seconds': {'min': lag[0], 'max': lag[1]},
            'minimum_source_dwell_seconds': dwell,
            'review_basis_ref': 'independent temporal experimental record'}


def edge(i, source, target, src_type, dst_type, *, context, timing=None):
    row = {'proposal_id': 'MGR-' + str(i) * 24,
           'source': {'entity_id': source[0], 'name': source[1], 'entity_type': src_type, 'aliases': []},
           'target': {'entity_id': target[0], 'name': target[1], 'entity_type': dst_type, 'aliases': []},
           'relation_type': 'CAUSES', 'polarity': 'AFFIRMATIVE',
           'claim_modality': 'REFERENCE_ASSERTION_REQUIRES_REVIEW',
           'evidence_quote': source[1] + ' 导致 ' + target[1],
           'equipment_scope': SCOPE, 'review_status': 'INDEPENDENTLY_REVIEWED_REFERENCE_RELATION',
           'reviewer': 'independent_engineer', 'truth_source_ref': 'original laboratory record',
           'snapshot': 3,
           'reviewed_applicability': context,
           'provenance': {'source_sha256': str(i) * 64, 'source_locators': ['line:1']}}
    if timing is not None:
        row['reviewed_temporal'] = timing
    return row


def chain(*, timelines=True):
    tracking = app('TRACKING', temp=(10, 30), thermal='HOT')
    standby = app('STANDBY', temp=(3, 9), thermal='COLD')
    a = edge(1, ('f', 'PUMP_FAULT_1'), ('m', 'bearing drag'), 'FAULT_MODE', 'FAILURE_MECHANISM',
             context=tracking,
             timing=temporal(0, tracking, standby, lag=(2, 5)) if timelines else None)
    b = edge(2, ('m', 'bearing drag'), ('s', 'bearing alarm'), 'FAILURE_MECHANISM', 'SYMPTOM',
             context=standby,
             timing=temporal(1, standby, standby, lag=(6, 11), dwell=3) if timelines else None)
    return [a, b]


def review(rows):
    return evaluate_causal_chains(fault_id='PUMP_FAULT_1', fault_name='PUMP_FAULT_1',
                                  equipment_scope=SCOPE, reviewed_edges=rows, snapshot=4)


def test_phase_aware_path_handles_reviewed_mode_and_temperature_transition():
    result = review(chain())
    assert len(result['chains']) == 1
    row = result['chains'][0]
    assert row['reviewed_context_consistency']['conflict_reasons']  # original R9 strict intersection
    audit = row['reviewed_temporal_consistency']
    assert audit['status'] == 'REVIEWED_SEQUENCE_COMPATIBLE_NOT_FLIGHT_VERIFIED'
    assert audit['reviewed_stage_count'] == 2
    assert audit['cumulative_lag_seconds'] == {'min': 8.0, 'max': 16.0}
    assert not row['joint_applicability_verified']
    assert not result['diagnostic_vote_allowed']
    assert not result['current_root_cause_proven']


def test_legacy_static_path_still_blocks_different_modes():
    result = review(chain(timelines=False))
    assert not result['chains']
    assert any('OPERATING_MODE_DISJOINT' in x['reasons'] for x in result['blocked'])


def test_partial_timeline_cannot_bypass_static_gate():
    rows = chain()
    del rows[1]['reviewed_temporal']
    out = review(rows)
    assert out['chains'] == []
    assert any('INCOMPLETE_REVIEWED_TEMPORAL_CHAIN' in x['reasons'] for x in out['blocked'])


def test_bad_stage_gap_wrong_order_and_process_rejected():
    for which, bad in [('gap', 4), ('order', 0), ('process', 'OTHER_PROCESS_01')]:
        rows = chain()
        rows[1]['reviewed_temporal']['stage_index' if which != 'process' else 'process_id'] = bad
        out = review(rows)
        assert out['chains'] == []
        assert any(('TEMPORAL_PROCESS_MISMATCH' if which == 'process' else 'TEMPORAL_STAGE_ORDER_OR_GAP')
                   in x['reasons'] for x in out['blocked'])


def test_incompatible_state_transition_rejected():
    rows = chain()
    rows[0]['reviewed_temporal']['post_context'] = app('TRACKING')
    out = review(rows)
    assert not out['chains']
    assert any('TEMPORAL_STATE_TRANSITION_INCOMPATIBLE' in x['reasons'] for x in out['blocked'])


def test_timing_cannot_widen_reviewed_edge_scope():
    rows = chain()
    rows[0]['reviewed_temporal']['pre_context'] = app('STANDBY')
    out = review(rows)
    assert not out['chains']
    assert any('INVALID_OR_SCOPE_WIDENING_REVIEWED_TIMELINE' in x['reasons'] for x in out['blocked'])
    rows = chain()
    rows[0]['reviewed_temporal']['pre_context'] = app('TRACKING', temp=(0, 70))
    assert not review(rows)['chains']


def test_bad_nonfinite_negative_boolean_or_inverted_lag_or_dwell_rejected():
    for lag, dwell in (((0, math.inf), 1), ((-1, 1), 0), ((3, 2), 0), ((0, 1), 3), ((True, 4), 0)):
        value = temporal(0, app('TRACKING'), app('STANDBY'), lag=lag, dwell=dwell)
        with pytest.raises(ValueError):
            normalize_reviewed_temporal(value, equipment_scope=SCOPE)


def test_reviewed_temporal_metadata_cannot_be_attached_to_unreviewed_edges():
    rows = chain()
    rows[0]['review_status'] = 'PENDING'
    assert not review(rows)['chains']
    rows = chain()
    rows[1]['equipment_scope'] = 'OTHER_MODEL'
    assert not review(rows)['chains']


@dataclass
class Chunk:
    text: str = 'PUMP_FAULT_1 导致 bearing drag。'
    chunk_id: str = 'R10-GENERIC-CHUNK'
    source_sha256: str = 'b' * 64
    filename: str = 'test-pump.md'
    source_locators: tuple = ('line:1',)
    section_path: tuple = ('synthetic',)


def test_temporal_approval_export_checksum_and_reject_llm_provided_timing(tmp_path):
    fragment = GraphChunkExtraction.model_validate({
        'entities': [{'local_id': 'f', 'name': 'PUMP_FAULT_1', 'entity_type': 'FAULT_MODE'},
                     {'local_id': 'm', 'name': 'bearing drag', 'entity_type': 'FAILURE_MECHANISM'}],
        'relations': [{'source_local_id': 'f', 'target_local_id': 'm', 'relation_type': 'CAUSES',
                       'evidence_quote': 'PUMP_FAULT_1 导致 bearing drag。'}]})
    proposals, rejected = validate_fragment(fragment, Chunk(), source_group_id='JOB-' + 'F' * 24)
    assert len(proposals) == 1 and not rejected
    db = MaterialCaseMemory(tmp_path/'source.db', reviewers=('independent_engineer',))
    wrong = {**proposals[0], 'reviewed_temporal': temporal(0, app('TRACKING'), app('STANDBY'))}
    with pytest.raises(ValueError, match='invalid sourced'):
        db.stage_graph_proposals(source_group_id='JOB-' + 'F' * 24, proposals=[wrong])
    db.stage_graph_proposals(source_group_id='JOB-' + 'F' * 24, proposals=proposals)
    with pytest.raises(ValueError):
        db.decide_graph(proposal_id=proposals[0]['proposal_id'], reviewer='independent_engineer',
                        decision='CONFIRM', truth_source_ref='independent inspection',
                        equipment_scope=SCOPE, expected_snapshot=0,
                        reviewed_temporal=temporal(0, app('TRACKING'), app('STANDBY')))
    db.decide_graph(proposal_id=proposals[0]['proposal_id'], reviewer='independent_engineer',
                    decision='CONFIRM', truth_source_ref='independent inspection',
                    equipment_scope=SCOPE, expected_snapshot=0, reviewed_applicability=app('TRACKING'),
                    reviewed_temporal=temporal(0, app('TRACKING'), app('STANDBY')))
    approved = db.approved_graph_for_reasoning(equipment_scope=SCOPE, snapshot=1, exclude_source_group_id='')
    assert approved[0]['reviewed_temporal']['stage_index'] == 0
    release = db.export_graph_release(equipment_scope=SCOPE, snapshot=1,
                                      output_root=tmp_path/'release')
    _, graph = verify_graph_release(release['directory'])
    assert graph['relations'][0]['reviewed_temporal']['post_context']['operating_modes'] == ['STANDBY']
    graph_file = Path(release['directory']) / 'graph.json'
    payload = json.loads(graph_file.read_text(encoding='utf-8'))
    payload['relations'][0]['reviewed_temporal']['lag_seconds']['min'] = 500
    graph_file.write_text(json.dumps(payload))
    with pytest.raises(MaterialGraphReleaseError, match='checksum'):
        verify_graph_release(release['directory'])
