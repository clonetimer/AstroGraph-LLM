"""ESA corrected eventwise score with explicit rare/gap challenges.

Integer nanosecond coordinates; annotated intervals closed, alarms half-open
except a final closed alarm. EW precision excludes unselected annotated events
from event FP, as the reference does; rare/gap alarm counts are additional.
"""
from __future__ import annotations
from collections import defaultdict

def merge(intervals):
    out=[]
    for a,b in sorted(intervals):
        if b<a:raise ValueError('EVENT_INTERVAL_REVERSED')
        if out and a<=out[-1][1]:out[-1][1]=max(b,out[-1][1])
        else:out.append([a,b])
    return out

def overlap(alarm,event):
    a,b=event;s,e=alarm['start_ns'],alarm['end_ns']
    return s<=b and (e>a or (e==a and alarm.get('end_closed',False)))

def duration(intervals):return sum(b-a for a,b in merge(intervals))

def intersection_duration(left,right):
    a=merge(left);b=merge(right);i=j=0;total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return total

def evaluate_events(truth,alarms,*,full_range,confirmation_duration_s=600):
    lower,upper=full_range
    if upper<=lower:raise ValueError('EVALUATION_TIME_RANGE_INVALID')
    grouped=defaultdict(list);categories={}
    for row in truth:
        category=row['category'];aid=row['event_id'];a=row['start_ns'];b=row['end_ns']
        if category not in {'ANOMALY','RARE_NOMINAL','COMMUNICATION_GAP'}:raise ValueError('ESA_EVENT_SEMANTICS_UNADMITTED')
        if not lower<=a<=b<=upper:raise ValueError('TRUTH_OUTSIDE_EVALUATION_RANGE')
        if aid in categories and categories[aid]!=category:raise ValueError('EVENT_CATEGORY_DRIFT')
        categories[aid]=category;grouped[aid].append((a,b))
    for alarm in alarms:
        if not lower<=alarm['start_ns']<=alarm['end_ns']<=upper:raise ValueError('ALARM_OUTSIDE_EVALUATION_RANGE')
    all_gt=merge([interval for intervals in grouped.values() for interval in intervals])
    selected={aid:merge(intervals) for aid,intervals in grouped.items() if categories[aid]=='ANOMALY'}
    matched=[False]*len(alarms);tp=fn=redundant=0;delays={};challenges={'RARE_NOMINAL':0,'COMMUNICATION_GAP':0}
    for aid,intervals in grouped.items():
        intervals=merge(intervals);hits=[i for i,p in enumerate(alarms) if any(overlap(p,e) for e in intervals)]
        if categories[aid]!='ANOMALY':
            challenges[categories[aid]]+=bool(hits);continue
        if hits:
            tp+=1
            for i in hits:matched[i]=True
        else:fn+=1
        for e in intervals:redundant+=max(0,sum(overlap(p,e) for p in alarms)-1)
        first=min((max(p['start_ns'],a) for p in alarms for a,b in intervals if overlap(p,(a,b))),default=None)
        confirmed=min((max(p['start_ns']+int(confirmation_duration_s*1e9),a) for p in alarms for a,b in intervals
          if overlap(p,(a,b)) and p['end_ns']-p['start_ns']>=confirmation_duration_s*1e9 and max(p['start_ns']+int(confirmation_duration_s*1e9),a)<=min(p['end_ns'],b)),default=None)
        start=min(a for a,b in intervals);delays[aid]={'first_detection_delay_s':(first-start)/1e9 if first is not None else None,
          'confirmed_detection_delay_s':(confirmed-start)/1e9 if confirmed is not None else None}
    fp=sum(not m and not any(overlap(p,e) for e in all_gt) for p,m in zip(alarms,matched))
    predicted=[(p['start_ns'],p['end_ns']) for p in alarms]
    nominal_ns=upper-lower-duration(all_gt);false_ns=duration(predicted)-intersection_duration(predicted,all_gt)
    if nominal_ns<=0:raise ValueError('NOMINAL_EVALUATION_DURATION_INSUFFICIENT')
    tnr=1-false_ns/nominal_ns;raw_precision=tp/(tp+fp) if tp+fp else 0.;precision=raw_precision*tnr;recall=tp/(tp+fn) if tp+fn else 0.
    f=.0 if .25*precision+recall==0 else 1.25*precision*recall/(.25*precision+recall)
    return {'TP_events':tp,'FP_events':fp,'FN_events':fn,'redundant_detections':redundant,'raw_event_precision':raw_precision,
      'event_precision':precision,'event_recall':recall,'corrected_eventwise_F0.5':f,'duration_true_negative_rate':tnr,
      'false_alarm_event_count':fp,'false_alarm_duration_s':false_ns/1e9,'nominal_duration_s':nominal_ns/1e9,
      'rare_nominal_false_positive_count':challenges['RARE_NOMINAL'],'communication_gap_false_positive_count':challenges['COMMUNICATION_GAP'],
      'event_detection_delays':delays,'confirmed_detection_policy':f'continuous alarm for {confirmation_duration_s}s; custom auxiliary metric, not ESA EW scorer',
      'definition':'ESA ESAScores EW correction; Category=Anomaly; rare/gap excluded from event FP per official reference and separately counted'}
