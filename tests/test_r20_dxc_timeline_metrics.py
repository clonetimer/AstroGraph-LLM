from astrograph.evaluation.dxc_timeline_metrics import evaluate_case, summarize


def test_multiple_faults_are_one_candidate_and_early_alarm_is_not_true_positive():
    truth = {'true_fault_ids': ['a:Failed', 'b:Offset'], 'fault_events': [{'injection_time_s': 40}, {'injection_time_s': 60}]}
    prediction = {'predicted_fault_ids': ['a:Failed', 'c:Drift'], 'timeline': [
        {'time_s': 35, 'channel_strengths': {'x': 90}, 'predicted_fault_ids': ['a:Failed']},
        {'time_s': 65, 'channel_strengths': {'x': 90}, 'predicted_fault_ids': ['a:Failed', 'c:Drift']}]}
    metrics = evaluate_case(truth, prediction)
    assert metrics['DXC09_first_positive_category'] == 'FP'
    assert metrics['fault_detected'] is True
    assert metrics['fault_isolated'] is False
    assert metrics['DXC09_candidate_classification_errors'] == 2
    assert metrics['DXC09_candidate_classification_error_rate'] == 2 / 3
    aggregate = summarize([{'metrics': metrics, 'runtime': {'inference_wall_s': .1}}])
    assert aggregate['DXC09_patched_single_candidate_classification']['classification_errors'] == 2


def test_empty_nominal_candidate_has_no_classification_rate_observation():
    metrics = evaluate_case({'true_fault_ids': [], 'fault_events': []}, {'predicted_fault_ids': [], 'timeline': []})
    assert metrics['DXC09_first_positive_category'] == 'TN'
    assert metrics['DXC09_candidate_classification_errors'] == 0
    assert metrics['DXC09_candidate_classification_error_rate'] is None
