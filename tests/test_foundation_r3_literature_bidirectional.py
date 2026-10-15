"""R3 document and GraphRAG adversarial boundaries; no LLM/remote endpoint."""
from __future__ import annotations

import hashlib
from pathlib import Path

from astrograph.integrations.material_hierarchical_rag import HierarchicalMaterialIndex
from astrograph.integrations.material_multihop import evaluate_bidirectional_hypotheses
from astrograph.integrations.platform_knowledge import _LocalChunk
from astrograph.integrations.result_log_review import SourceLine
from astrograph.retrieval.structured_document import SectionFirstDocumentIngestor

SCOPE = 'SAT_X_ADCS_V1'
SHA = 'a' * 64


def edge(eid, source_id, source_name, source_type, target_id, target_name,
         target_type, relation, *, scope=SCOPE, polarity='AFFIRMATIVE',
         reviewer='expert', modality='REFERENCE_ASSERTION_REQUIRES_REVIEW', snapshot=5):
    return {
        'proposal_id': eid, 'equipment_scope': scope,
        'snapshot': snapshot,
        'review_status': 'INDEPENDENTLY_REVIEWED_REFERENCE_RELATION',
        'source': {'entity_id': source_id, 'name': source_name, 'entity_type': source_type},
        'target': {'entity_id': target_id, 'name': target_name, 'entity_type': target_type},
        'relation_type': relation, 'polarity': polarity, 'claim_modality': modality,
        'provenance': {'source_sha256': SHA, 'source_locators': ['page:2']},
        'evidence_quote': f'{source_name} {relation} {target_name}',
        'reviewer': reviewer, 'truth_source_ref': 'human-approval',
    }


def chunk(cid, text, source):
    return _LocalChunk(
        chunk_id=cid, source_id=source, filename=source + '.md',
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        text=text, section_path=('Diagnosis',), source_locators=('line:3',),
        review_status='HUMAN_REVIEWED', content_safety_status='PASS',
        modalities=('TEXT',),
    )


def test_graph_expansion_requires_review_scope_and_provenance():
    good = edge('approved', 'f1', 'FAULT_WHEEL', 'FAULT_MODE',
                'm1', 'wheel bearing abrasion', 'FAILURE_MECHANISM', 'ASSOCIATED_WITH')
    corpus = [chunk('needle', 'wheel bearing abrasion produces friction.', 'manual'),
              chunk('other', 'sun sensor reset was routine.', 'other')]
    index = HierarchicalMaterialIndex(corpus)
    params = dict(base_query='indeterminate', fault_id='FAULT_WHEEL',
                  fault_name='FAULT_WHEEL', equipment_scope=SCOPE, graph_snapshot=5)
    for poison in ({**good, 'reviewer': ''}, {**good, 'equipment_scope':'SAT_Y_ADCS_V1'},
                   {**good, 'provenance': {'source_sha256':'f'*64, 'source_locators':[]}},
                   {**good, 'snapshot':6}):
        result = index.retrieve(**params, approved_edges=[poison])
        assert 'APPROVED_GRAPH_HINTS' not in result['methods']
        assert result['hits'] == []
    result = index.retrieve(**params, approved_edges=[good])
    assert result['hits'][0]['knowledge_id'] == 'needle'
    assert 'APPROVED_GRAPH_HINTS' in result['methods']
    earlier = index.retrieve(**params, approved_edges=[{**good, 'snapshot': 1}])
    assert earlier['hits'][0]['knowledge_id'] == 'needle'
    negative = {**good, 'polarity': 'NEGATED', 'proposal_id':'negative'}
    contradicted = index.retrieve(**params, approved_edges=[good, negative])
    assert 'APPROVED_GRAPH_HINTS' not in contradicted['methods']


def test_contrastive_literature_points_never_count_as_negative_votes():
    index = HierarchicalMaterialIndex([
        chunk('wheel', 'Reaction wheel bearing abrasion raises motor current.', 'wheel-doc'),
        chunk('battery', 'Battery cell degradation reduces SOC.', 'battery-doc'),
    ])
    result = index.retrieve_contrastive(
        base_query='abnormal operating condition', fault_id='WHEEL',
        fault_name='bearing abrasion', competitor_fault_id='EPS',
        competitor_fault_name='cell degradation', equipment_scope=SCOPE)
    assert result['status'] == 'REVIEW_POINTERS_AVAILABLE'
    assert result['primary_only_chunk_ids'] or result['competitor_only_chunk_ids']
    assert result['evidence_absence_is_not_refutation']
    assert not result['current_root_cause_proven']
    assert not result['automatic_control_authorized']


def test_bidirectional_graph_retrieves_ambiguous_symptom_without_reverse_causality():
    graph = [
        edge('a', 'f1', 'FAULT_WHEEL', 'FAULT_MODE',
             'm1', 'wheel bearing wear', 'FAILURE_MECHANISM', 'ASSOCIATED_WITH'),
        edge('b', 'm1', 'wheel bearing wear', 'FAILURE_MECHANISM',
             's1', 'motor current rises', 'SYMPTOM', 'MANIFESTS_AS'),
        edge('c', 'f2', 'FAULT_POWER', 'FAULT_MODE',
             's1', 'motor current rises', 'SYMPTOM', 'INDICATED_BY'),
        edge('d', 'f2', 'FAULT_POWER', 'FAULT_MODE',
             's2', 'bus voltage drops', 'SYMPTOM', 'INDICATED_BY',
             scope='SAT_Y_ADCS_V1'),
    ]
    lines = [SourceLine('log_reports', 'log.txt', 'line:1',
                        'motor current rises warning', 'b'*64)]
    result = evaluate_bidirectional_hypotheses(
        candidates=[{'fault_id':'WHEEL','fault_name':'FAULT_WHEEL'},
                    {'fault_id':'POWER','fault_name':'FAULT_POWER'}],
        equipment_scope=SCOPE, log_lines=lines, reviewed_edges=graph,
        snapshot=5)
    assert result['status'] == 'LEXICAL_REVERSE_POINTERS_REQUIRE_OBSERVATION_VALIDATION'
    assert len(result['shared_symptoms']) == 1
    assert set(result['shared_symptoms'][0]['candidate_fault_ids']) == {'WHEEL','POWER'}
    assert all(x['reverse_direction_is_causal'] is False for x in result['reverse_observation_index'])
    assert result['blocked_edge_count'] == 1
    assert all(not s['sensor_observation_verified'] for c in result['candidates']
               for s in c['expected_symptoms'])
    assert not result['diagnostic_vote_allowed']
    graph[1] = {**graph[1], 'polarity': 'NEGATED'}
    negated = evaluate_bidirectional_hypotheses(
        candidates=[{'fault_id':'WHEEL','fault_name':'FAULT_WHEEL'}],
        equipment_scope=SCOPE, log_lines=lines, reviewed_edges=graph,
        snapshot=5)
    assert not negated['candidates'][0]['expected_symptoms']


def test_markdown_image_alt_requires_review_and_cannot_escape_asset_directory(tmp_path):
    root = tmp_path / 'assets'
    root.mkdir()
    (root / 'safe.png').write_bytes(b'local bytes')
    outside = tmp_path / 'secret.png'
    outside.write_bytes(b'secret bytes')
    parser = SectionFirstDocumentIngestor()
    elements, warnings = parser.parse_markdown(
        '# Sensor\n\n![unverified diagnostic verdict](safe.png)\n\n'
        '![do not read](../secret.png)', asset_root=root)
    assert len(elements) == 2
    assert elements[0].asset_sha256 == hashlib.sha256(b'local bytes').hexdigest()
    assert not elements[0].representation_reviewed
    assert elements[1].asset_sha256 is None
    assert not elements[1].representation_reviewed
    assert 'IMAGE_ASSET_OUTSIDE_ROOT_BLOCKED:5' in warnings
    assert sum('IMAGE_REPRESENTATION_REVIEW_REQUIRED' in w for w in warnings) == 2


def test_negative_reviewed_claims_retrieved_for_audit_not_diagnostic_voting():
    positive = edge('pos', 'f1', 'FAULT_WHEEL', 'FAULT_MODE',
                    'm1', 'bearing wear', 'FAILURE_MECHANISM', 'ASSOCIATED_WITH')
    negative = {**positive, 'proposal_id':'neg', 'polarity':'NEGATED'}
    wrong_scope = {**negative, 'proposal_id':'cross', 'equipment_scope':'SAT_Y_ADCS_V1'}
    unreviewed = {**negative, 'proposal_id':'draft', 'reviewer':''}
    result = HierarchicalMaterialIndex.retrieve_reviewed_counterevidence(
        fault_id='WHEEL', fault_name='FAULT_WHEEL',
        approved_edges=[positive, negative, wrong_scope, unreviewed],
        equipment_scope=SCOPE, graph_snapshot=5)
    assert result['status'] == 'REVIEWED_COUNTERCLAIMS_FOUND'
    assert {e['edge_id'] for e in result['items']} == {'pos','neg'}
    assert {e['status'] for e in result['items']} == {'REVIEWED_CONFLICTING_TRIPLE'}
    assert all(e['source_sha256'] == SHA for e in result['items'])
    assert not result['automatic_diagnostic_vote_allowed']
    other = HierarchicalMaterialIndex.retrieve_reviewed_counterevidence(
        fault_id='WHEEL', fault_name='FAULT_WHEEL',
        approved_edges=[negative], equipment_scope='SAT_Z', graph_snapshot=5)
    assert not other['items']


def test_older_approved_edges_remain_in_pinned_current_graph_snapshot():
    earlier = edge('old', 'f1','FAULT_WHEEL','FAULT_MODE',
                   's1','motor current rises','SYMPTOM','INDICATED_BY',snapshot=1)
    from astrograph.retrieval.approved_community_search import build_approved_community_reports
    old_index = build_approved_community_reports(
        reviewed_edges=[earlier], equipment_scope=SCOPE, snapshot=5)
    assert old_index['indexed_edge_count'] == 1
    future = build_approved_community_reports(
        reviewed_edges=[{**earlier, 'snapshot':6}], equipment_scope=SCOPE, snapshot=5)
    assert not future['reports']
    # Revisions >=1 remain relevant in snapshot=5 when returned by the
    # snapshot-pinned repository (which handles graph retractions).
    checked = evaluate_bidirectional_hypotheses(
        candidates=[{'fault_id':'WHEEL','fault_name':'FAULT_WHEEL'}],
        equipment_scope=SCOPE, log_lines=[], reviewed_edges=[earlier],snapshot=5)
    assert checked['candidates'][0]['expected_symptoms']


def test_conditional_negative_cannot_veto_unconditional_reviewed_relation():
    good = edge('good', 'f1', 'FAULT_WHEEL', 'FAULT_MODE',
                's1', 'wheel current spike', 'SYMPTOM', 'INDICATED_BY', snapshot=1)
    conditional = {**good, 'proposal_id': 'cond', 'polarity': 'NEGATED',
                   'claim_modality': 'HYPOTHETICAL_OR_CONDITIONAL', 'snapshot': 2}
    from astrograph.retrieval.approved_community_search import build_approved_community_reports
    community = build_approved_community_reports(
        reviewed_edges=[good,conditional],equipment_scope=SCOPE,snapshot=5)
    assert community['indexed_edge_count'] == 1
    index = HierarchicalMaterialIndex([chunk('needle', 'wheel current spike', 'manual')])
    search = index.retrieve(base_query='other',fault_id='FAULT_WHEEL',
                            fault_name='FAULT_WHEEL',approved_edges=[good,conditional],
                            equipment_scope=SCOPE,graph_snapshot=5)
    assert 'APPROVED_GRAPH_HINTS' in search['methods']
    result = evaluate_bidirectional_hypotheses(
        candidates=[{'fault_id':'WHEEL','fault_name':'FAULT_WHEEL'}],
        equipment_scope=SCOPE,log_lines=[], reviewed_edges=[good,conditional],snapshot=5)
    assert result['candidates'][0]['expected_symptoms']
    counters = index.retrieve_reviewed_counterevidence(
        fault_id='WHEEL',fault_name='FAULT_WHEEL',approved_edges=[good,conditional],
        equipment_scope=SCOPE,graph_snapshot=5)
    assert counters['items'][0]['status'] == 'CONDITIONAL_NEGATION_REQUIRES_REVIEW'