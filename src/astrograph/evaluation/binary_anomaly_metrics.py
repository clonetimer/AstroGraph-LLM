"""Segment classification metrics; scores are not probabilities."""
from __future__ import annotations
import math

def binary_metrics(truth, scores, threshold):
    if len(truth)!=len(scores) or not truth:raise ValueError('METRIC_INPUT_INVALID')
    if any(y not in (0,1) for y in truth) or any(not math.isfinite(s) for s in scores):raise ValueError('METRIC_INPUT_INVALID')
    tp=sum(y==1 and s>=threshold for y,s in zip(truth,scores)); fp=sum(y==0 and s>=threshold for y,s in zip(truth,scores))
    fn=sum(truth)-tp;tn=len(truth)-sum(truth)-fp
    def ratio(a,b):return a/b if b else None
    precision=ratio(tp,tp+fp);recall=ratio(tp,tp+fn)
    result={'evaluated_count':len(truth),'TP':tp,'FP':fp,'TN':tn,'FN':fn,'precision':precision,'recall':recall,
      'F0.5':ratio(1.25*tp,1.25*tp+.25*fn+fp),'F1':ratio(2*tp,2*tp+fn+fp),
      'balanced_accuracy':((recall+tn/(tn+fp))/2 if recall is not None and tn+fp else None),
      'false_positive_segment_count':fp,'false_negative_segment_count':fn}
    positives=sum(truth);negatives=len(truth)-positives
    if positives and negatives:
        groups={}
        for y,s in zip(truth,scores):
            pair=groups.setdefault(s,[0,0]);pair[y]+=1
        lower_neg=0;auc=0
        for s,(n,p) in sorted(groups.items()):auc+=p*(lower_neg+.5*n);lower_neg+=n
        ct=cf=0;ap=0
        for s,(n,p) in sorted(groups.items(),reverse=True):ct+=p;cf+=n;ap+=(p/positives)*(ct/(ct+cf))
        result.update(AUROC=auc/(positives*negatives),AUPRC_average_precision=ap)
    else:result.update(AUROC=None,AUPRC_average_precision=None)
    return result
