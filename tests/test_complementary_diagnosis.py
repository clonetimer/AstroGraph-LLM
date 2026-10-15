from copy import deepcopy
from pathlib import Path

import pytest

from astrograph.oag.cli import demo_input
from astrograph.oag.complementary import complementary_diagnosis
from astrograph.oag.runtime import OAG_PIPELINE_ID, build_csv_oag_runtime
from astrograph.runtime import RuntimeRequest

ROOT = Path(__file__).resolve().parents[1]


def context():
    data = demo_input(ROOT)
    runtime = build_csv_oag_runtime(ROOT / 'benchmarks/knowledge_graph/online_extended')
    result = runtime.execute(RuntimeRequest(request_id='complement-test', pipeline_id=OAG_PIPELINE_ID,
                                            payload=data.model_dump(mode='json')))
    assert result.status.value == 'SUCCEEDED'
    return data.bundle, result.output['summary']


def uncertain_context():
    bundle, summary = context()
    summary = deepcopy(summary)
    summary['suspected_fault_id'] = None
    summary['status'] = 'BLOCKED'
    summary['blocking_conflicts'] = ['THREE_TASK_DETECTOR_NOT_TRIGGERED']
    summary['open_set_actions'] = []
    summary['three_task_results'] = {'detection': {'detected': False}, 'classification': []}
    return bundle, summary


def test_detector_disagreement_retains_hypothesis_not_confirmed_class():
    bundle, summary = uncertain_context()
    result = complementary_diagnosis(bundle, summary)
    assert result['status'] == 'CANDIDATE_SET'
    assert result['candidate_fault_ids']
    assert result['selected_fault_id'] is None
    assert result['detector_mechanism_disagreement']
    assert result['additional_observations']
    assert result['coverage_guarantee'] is None
    assert all(not p['temporal_propagation_observed'] for p in result['mechanism_paths'])


def test_hard_conflict_blocks_candidate_set():
    bundle, summary = uncertain_context()
    summary['blocking_conflicts'].append('IDENTICAL_PAIRED_TELEMETRY_NO_DEVIATION')
    result = complementary_diagnosis(bundle, summary)
    assert result['status'] == 'BLOCKED'
    assert not result['candidate_fault_ids']


def test_identical_reference_suppresses_model_only_candidates():
    bundle, summary = uncertain_context()
    summary['three_task_results'] = {'detection': {'detected': True}, 'classification': [
        {'fault_id': 'F', 'score': 1., 'rank': 1}]}
    result = complementary_diagnosis(bundle, summary, paired_reference_identical=True)
    assert result['status'] == 'NORMAL_REFERENCE_MATCH'
    assert result['candidate_fault_ids'] == []


def test_tied_boundary_overflow_is_not_silently_truncated():
    bundle, summary = uncertain_context()
    summary['three_task_results'] = {'detection': {'detected': True}, 'classification': [
        {'fault_id': f'F{i}', 'score': .25, 'rank': 1} for i in range(4)]}
    result = complementary_diagnosis(bundle, summary, maximum_candidates=3)
    assert result['status'] == 'CANDIDATE_OVERFLOW'
    assert not result['candidate_fault_ids']
    assert result['proposed_count'] >= 4


def test_no_measurement_and_observed_mismatch_are_different():
    bundle, summary = uncertain_context()
    bundle = bundle.model_copy(deep=True)
    from astrograph.evidence.executable import EvidenceFactStatus
    first, second = bundle.evidence_facts[:2]
    first.status = EvidenceFactStatus.MISSING_DATA
    second.status = EvidenceFactStatus.NOT_SATISFIED
    result = complementary_diagnosis(bundle, summary)
    matrix = {r['definition_id']: r for r in result['evidence_state_matrix']}
    assert matrix[first.evidence_definition_id]['state'] == 'MISSING_DATA'
    assert matrix[second.evidence_definition_id]['state'] == 'NOT_SATISFIED'


def test_future_mechanism_features_are_rejected():
    bundle, summary = context()
    bundle = bundle.model_copy(deep=True)
    bundle.feature_results[0].window_end_s = bundle.episode.end_s + 1
    with pytest.raises(ValueError, match='future'):
        complementary_diagnosis(bundle, summary)


def test_unmasked_trace_can_retain_hierarchy_excluded_alternative():
    bundle, summary = uncertain_context()
    summary['three_task_results'] = {
        'detection': {'detected': True},
        'classification': [{'fault_id': 'AFTER_MASK', 'score': 1., 'rank': 1}],
        'head_trace': {'raw_class_probabilities': {'a': .49, 'b': .51},
                       'class_to_runtime_fault': {'a': 'AFTER_MASK', 'b': 'BEFORE_MASK'}}}
    result = complementary_diagnosis(bundle, summary, maximum_candidates=10)
    assert 'BEFORE_MASK' in result['candidate_fault_ids']
    assert result['data_candidate_source'] == 'PRE_HIERARCHY_CLASS_DISTRIBUTION'


def test_runtime_exports_bounded_decision_and_assessment():
    data = demo_input(ROOT).model_copy(update={'complementary_diagnosis_enabled': True, 'review_evidence': True})
    runtime = build_csv_oag_runtime(ROOT / 'benchmarks/knowledge_graph/online_extended')
    result = runtime.execute(RuntimeRequest(request_id='complement-e2e', pipeline_id=OAG_PIPELINE_ID,
                                            payload=data.model_dump(mode='json')))
    assert result.status.value == 'SUCCEEDED'
    decision = result.output['summary']['diagnostic_decision']
    assert decision['status'] == 'SUPPORTED_SINGLETON'
    assert result.output['assessment']['bounded_candidate_fault_ids'] == decision['candidate_fault_ids']


@pytest.mark.parametrize('bound', [0, 1, 11])
def test_invalid_candidate_bounds_rejected(bound):
    bundle, summary = context()
    with pytest.raises(ValueError):
        complementary_diagnosis(bundle, summary, maximum_candidates=bound)
