"""Set-valued offline fault comparison, separate from DXC competition utility."""
from __future__ import annotations
import math

def evaluate_fault_set(truth, prediction, *, injection_time=None, first_detection_time=None, final_isolation_time=None):
    if not isinstance(truth,list) or not isinstance(prediction,list):raise ValueError('SET_VALUED_FAULT_IDS_REQUIRED')
    if len(set(truth))!=len(truth) or len(set(prediction))!=len(prediction):raise ValueError('DUPLICATE_FAULT_IDS')
    a=set(truth);b=set(prediction);tp=len(a&b);fp=len(b-a);fn=len(a-b)
    def delay(t):
        if t is None or injection_time is None:return None
        if not math.isfinite(t) or not math.isfinite(injection_time) or t<injection_time:raise ValueError('DETECTION_TIME_INVALID')
        return t-injection_time
    return {'exact_fault_set_match':a==b,'set_precision':tp/len(b) if b else (1.0 if not a else 0.0),
      'set_recall':tp/len(a) if a else (1.0 if not b else 0.0),'set_f1':2*tp/(len(a)+len(b)) if a or b else 1.0,
      'false_positive_fault_count':fp,'false_negative_fault_count':fn,'fault_detected':bool(a and b),
      'fault_isolated':bool(a and a==b),'first_detection_delay':delay(first_detection_time),'final_isolation_delay':delay(final_isolation_time),
      'metric_definition':'case-level injected-ever set comparison; not original DXC time utility'}

def admit_truth(truth):
    if truth.get('status')!='PARSED_INJECTION_TRUTH':raise ValueError('SCORING_BLOCKED_TRUTH_UNAVAILABLE')
    if not isinstance(truth.get('true_fault_ids'),list):raise ValueError('SET_VALUED_FAULT_IDS_REQUIRED')
