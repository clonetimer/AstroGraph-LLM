"""Independent EPS engineering baseline, never an oracle or formal diagnosis.

Uses only sensor/command records and the independently supplied XML catalog.
Its limited hardware model is deliberately exposed and measured, not qualified.
"""
from __future__ import annotations
from collections import defaultdict,deque
from statistics import median
from astrograph.integrations.multidimensional_diagnostics import MultiMetricWindowDetector

def validate_online_records(records):
    for r in records:
        allowed={'kind','time_tick','values'} if r.get('kind')=='sensors' else {'kind','time_tick','command_id','value'} if r.get('kind')=='command' else set()
        if not allowed or set(r)!=allowed:raise ValueError('DXC_TRUTH_OR_UNADMITTED_FIELD_IN_MODEL_INPUT')

def predict(records,catalog,config):
    from .data_exposure import guard_prediction_generation
    guard_prediction_generation()
    validate_online_records(records)
    components={c['component_id']:c for c in catalog['components']};adjacent=defaultdict(set)
    for e in catalog['connections']:adjacent[e['c1']].add(e['c2']);adjacent[e['c2']].add(e['c1'])
    baseline=defaultdict(list);windows=defaultdict(lambda:deque(maxlen=config['window_samples']));detectors={};timeline=[];last=-float('inf');started=False
    unverified=set();unsupported=set()
    for r in records:
        if r['kind']!='sensors':continue
        t=r['time_tick']/1000.
        for channel,value in r['values'].items():
            c=components.get(channel)
            if not c or c.get('kind')!='sensor':unsupported.add(channel);continue
            if c.get('unit_status')!='VERIFIED_FROM_XML':unverified.add(channel);continue
            if isinstance(value,bool):continue
            sample={'t':t,'values':{channel:value}};windows[channel].append(sample)
            if t<config['baseline_duration_s']:baseline[channel].append(sample)
        if t<config['baseline_duration_s']:continue
        if not started:
            for channel,samples in baseline.items():
                if len(samples)>=config['minimum_reference_samples']:
                    detectors[channel]=MultiMetricWindowDetector.from_samples(samples,[channel],baseline_fraction=1.,minimum_channels=1)
            started=True
        if t-last<config['decision_stride_s']:continue
        last=t;faults=set();strengths={}
        for channel,detector in detectors.items():
            window=list(windows[channel])
            if len(window)<4:continue
            anomalous,score,_=detector.detect(window)
            if not anomalous or score<config['score_threshold']:continue
            strengths[channel]=score;c=components[channel];values=[s['values'][channel] for s in window];modes=c.get('fault_modes',[])
            dimensions=detector._window_dimension_scores(channel,window)
            if max(values)==min(values) and 'Stuck' in modes:mode='Stuck'
            elif dimensions.get('TREND',0)>=dimensions.get('LEVEL_SHIFT',0) and 'Drift' in modes:mode='Drift'
            elif 'Offset' in modes:mode='Offset'
            else:mode=None
            if mode:faults.add(channel+':'+mode)
            # Units establish a current channel; spelling never establishes units.
            center=detector.references[channel].center
            if c.get('unit') in ('Amps','A') and center>1e-6 and abs(median(values))<abs(center)*config['current_drop_fraction']:
                for component in adjacent[channel]:
                    if 'FailedOff' in components.get(component,{}).get('fault_modes',[]):faults.add(component+':FailedOff')
        timeline.append({'time_s':t,'predicted_fault_ids':sorted(faults),'channel_strengths':strengths})
    if not timeline:raise ValueError('DXC_REFERENCE_OR_DECISION_WINDOW_INSUFFICIENT')
    return {'candidate_sets':[timeline[-1]['predicted_fault_ids']],'predicted_fault_ids':timeline[-1]['predicted_fault_ids'],'timeline':timeline,
      'unverified_unit_channels':sorted(unverified),'unsupported_channels':sorted(unsupported),'reference_channels':sorted(detectors),
      'semantics':'engineering hypothesis set; not formal fault evidence; incomplete hardware/intermittent model','qualification':'NOT_FLIGHT_QUALIFIED'}
