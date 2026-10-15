"""Set accuracy and timed detection; namespaces distinguish official DXC rules."""
from statistics import mean
from .fault_set_metrics import evaluate_fault_set

def evaluate_case(truth,prediction):
    faults=truth['true_fault_ids'];events=truth['fault_events'];timeline=prediction['timeline']
    injection=min((e['injection_time_s'] for e in events),default=None)
    complete=max((e['injection_time_s'] for e in events),default=None)
    alarms=[r['time_s'] for r in timeline if r['channel_strengths']]
    detected=next((t for t in alarms if injection is not None and t>=injection),None)
    final_set=prediction['predicted_fault_ids'];isolation=None
    if faults and set(final_set)==set(faults):
        stable=timeline[-1]['time_s']
        for r in reversed(timeline):
            if set(r['predicted_fault_ids'])!=set(faults):break
            stable=r['time_s']
        isolation=max(stable,complete)
    metrics=evaluate_fault_set(faults,final_set,injection_time=injection,first_detection_time=detected,final_isolation_time=isolation)
    metrics['fault_detected']=detected is not None
    metrics['false_alarm_before_injection']=sum(injection is not None and t<injection for t in alarms)
    first=alarms[0] if alarms else None
    # DAResult.java's first-positive rule treats an early positive as FP even
    # in a faulty scenario. Keep it separate from end-state binary detection.
    category='FP' if first is not None and (injection is None or first<injection) else 'TP' if first is not None else 'TN' if injection is None else 'FN'
    metrics['DXC09_first_positive_category']=category
    # Patched official Evaluator.jar updateClassification: each candidate's
    # weight multiplies the cardinality of symmetric difference, normalized
    # by union for the per-scenario rate. Our frozen output has one full-set
    # candidate of weight 1; this is not a top-1 fault approximation.
    truth_set, predicted_set = set(faults), set(final_set)
    union = truth_set | predicted_set
    errors = len(truth_set ^ predicted_set)
    metrics['DXC09_candidate_classification_errors'] = errors
    metrics['DXC09_candidate_classification_error_rate'] = errors / len(union) if union else None
    metrics['metric_definition']='final injected-ever set; detection after earliest injection; isolation stable exact set after final injection'
    return metrics

def summarize(cases):
    if not cases:return {'scored_count':0,'status':'NO_ADMITTED_TRUTH'}
    counts={k:sum(c['metrics']['DXC09_first_positive_category']==k for c in cases) for k in ('TP','FP','TN','FN')}
    def ratio(a,b):return a/b if b else 0.0
    return {'scored_count':len(cases),'exact_fault_set_match':mean(c['metrics']['exact_fault_set_match'] for c in cases),
      'macro_set_precision':mean(c['metrics']['set_precision'] for c in cases),'macro_set_recall':mean(c['metrics']['set_recall'] for c in cases),
      'macro_set_f1':mean(c['metrics']['set_f1'] for c in cases),'false_positive_fault_count':sum(c['metrics']['false_positive_fault_count'] for c in cases),
      'false_negative_fault_count':sum(c['metrics']['false_negative_fault_count'] for c in cases),
      'fault_detected_count':sum(c['metrics']['fault_detected'] for c in cases),'fault_isolated_count':sum(c['metrics']['fault_isolated'] for c in cases),
      'DXC09_original_first_positive_style':{'counts':counts,'false_positive_rate':ratio(counts['FP'],counts['FP']+counts['TN']),
       'false_negative_rate':ratio(counts['FN'],counts['FN']+counts['TP']),'detection_accuracy':(counts['TP']+counts['TN'])/len(cases),
       'definition_source':'NASA DXC09_Framework-1.1.zip DAResult.java; early faulty alarm is FP, not TP',
       'diagnostic_utility':None,'diagnostic_utility_status':'NOT_ENABLED_IN_OFFICIAL_INDUSTRIAL_CONFIG'},
      'DXC09_patched_single_candidate_classification':{
       'classification_errors':sum(c['metrics']['DXC09_candidate_classification_errors'] for c in cases),
       'classification_error_rate':mean(c['metrics']['DXC09_candidate_classification_error_rate'] for c in cases if c['metrics']['DXC09_candidate_classification_error_rate'] is not None) if any(c['metrics']['DXC09_candidate_classification_error_rate'] is not None for c in cases) else None,
       'definition_source':'NASA DXC09_Evaluator.zip Evaluator.jar DAResult bytecode: weighted symmetric difference / union; empty union omitted',
       'candidate_policy':'one complete predicted fault set, weight 1; supplied truth and valid XML component-mode hypotheses',
       'scope':'classification formula verified independently; not a claim of running the complete Java competition evaluator'},
      'mean_inference_wall_s':mean(c['runtime']['inference_wall_s'] for c in cases)}
