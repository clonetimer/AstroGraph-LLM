from pathlib import Path
import runpy


def test_hypothesis_hit_is_not_confirmed_accuracy_and_failures_stay_in_denominator():
    measure = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                'scripts/evaluate_complementary_diagnosis.py'))['measure']
    methods = ['DATA_CLASS', 'DATA_RAW_TOP2', 'FUSION_CLASS', 'COMPLEMENTARY']
    rows = [
        {'target': 'F', 'predictions': {m: {'kind': 'SET', 'ids': ['F', 'G']} for m in methods}},
        {'target': 'F', 'predictions': {m: {'kind': 'UNAVAILABLE', 'ids': []} for m in methods}},
        {'target': None, 'predictions': {m: {'kind': 'SET', 'ids': ['F']} for m in methods}},
    ]
    result = measure(rows)['COMPLEMENTARY']
    assert result['n'] == 3
    assert result['fault_candidate_hit_rate'] == .5
    assert result['fault_candidate_mean_size'] == 2
    assert result['fault_candidate_precision'] == .5
    assert result['strict_class_correct'] == 0
    assert result['unavailable'] == 1
    assert result['normal_or_pre_event_candidate_alerts'] == 1


def test_adoption_rejects_an_added_error_even_when_correct_count_improves():
    from copy import deepcopy
    compare = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                'scripts/compare_complementary_campaigns.py'))['compare']
    rows = [{'case_id': f'C{i}', 'endpoint_s': 40, 'mission': 'M', 'initial_condition_family': i,
             'target': 'F', 'predictions': {'COMPLEMENTARY': {'kind': 'SET', 'ids': ['F', 'G']}}}
            for i in range(3)]
    left = {'rows': rows}
    right = deepcopy(left)
    for row in right['rows']:
        row['predictions']['COMPLEMENTARY'] = {'kind': 'CLASS', 'ids': ['F']}
    assert compare(left, right)['adoption_gate'] == 'PASS'
    right['rows'][-1]['predictions']['COMPLEMENTARY']['ids'] = ['G']
    result = compare(left, right)
    assert result['totals']['added_correct_specific'] == 2
    assert result['totals']['added_wrong_specific'] == 1
    assert result['adoption_gate'] == 'NOT_PASSED'
