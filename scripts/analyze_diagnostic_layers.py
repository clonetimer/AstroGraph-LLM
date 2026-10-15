"""Evaluator-only, paired layer comparison; never imported by inference."""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
NORMAL='NORMAL_REFERENCE_MATCH'
ABSTAIN='ABSTAIN'
FAILED='DATA_UNAVAILABLE'
PRIMARY={None,'ADCS_RW_JAM','ADCS_RW_TORQUE_AUTHORITY_LOSS','ADCS_RW_SPEED_SENSOR_BIAS'}


def layer_decisions(row):
    s=row.get('summary')
    if row['status']!='SUCCEEDED' or not s:
        return {k:FAILED for k in ['DATA_ONLY','DATA_PLUS_REFERENCE_IDENTITY','KG_ONLY','FUSION_NO_TIE','FUSION']}
    t=s['three_task_results'];detected=t['detection']['detected']
    data=t['classification'][0]['fault_id'] if detected else NORMAL
    identical=row.get('paired_reference_identical',False)
    eligible=[x['fault_id'] for x in s['candidates'] if x['eligible_for_explanation']]
    graph=eligible[0] if len(eligible)==1 else ABSTAIN
    fused=s['suspected_fault_id'] or ABSTAIN
    no_tie=ABSTAIN if s.get('tie_resolution') else fused
    return {'DATA_ONLY':data,'DATA_PLUS_REFERENCE_IDENTITY':NORMAL if identical else data,
            'KG_ONLY':NORMAL if identical else graph,'FUSION_NO_TIE':NORMAL if identical else no_tie,
            'FUSION':NORMAL if identical else fused}


def analyze(directory, dataset):
    evaluation=json.loads((directory/'evaluation.json').read_text())
    terminal=[];temporal=[]
    for record in evaluation['records']:
        case=directory/record['case_id'];windows=json.loads((case/'windows.json').read_text())
        metadata=json.loads((dataset/record['source_path']).with_name('sample_metadata.json').read_text())
        target=record['evaluation_target'];expected=target or NORMAL
        for row in windows:
            # Old post42 recordings lack the public per-window pairing receipt.
            # Terminal input already contains its hash-bound native receipt.
            if 'paired_reference_identical' not in row and row['endpoint_s']==300 and (case/'input.json').exists():
                binding=json.loads((case/'input.json').read_text())['three_task_result']['evidence_projection']
                row={**row,'paired_reference_identical':binding['source_window_sha256']==binding['nominal_window_sha256']}
            decisions=layer_decisions(row)
            active=metadata.get('event_start_s') is not None and row['endpoint_s']>=metadata['event_start_s']
            entry={'case_id':record['case_id'],'trajectory_family':metadata['pair_seed'],'endpoint_s':row['endpoint_s'],
                   'expected':expected if active else NORMAL,'primary_scope':target in PRIMARY,
                   'fault_active':active,'window_start_s':(row.get('summary') or {}).get('window',{}).get('start_s'),
                   'decisions':decisions}
            temporal.append(entry)
            if row['endpoint_s']==300:terminal.append(entry)
    def measure(rows):
        result={}
        for layer in ['DATA_ONLY','DATA_PLUS_REFERENCE_IDENTITY','KG_ONLY','FUSION_NO_TIE','FUSION']:
            pairs=[(x['expected'],x['decisions'][layer]) for x in rows]
            result[layer]={'n':len(pairs),'correct':sum(a==b for a,b in pairs),
                          'abstained':sum(b==ABSTAIN for _,b in pairs),'unavailable':sum(b==FAILED for _,b in pairs),
                          'confusion':[{'expected':a,'predicted':b,'count':n} for (a,b),n in sorted(Counter(pairs).items())]}
        return result
    primary=[x for x in terminal if x['primary_scope']];negative=[x for x in terminal if not x['primary_scope']]
    pairs=Counter()
    for x in primary:
        a=x['decisions']['DATA_PLUS_REFERENCE_IDENTITY']==x['expected'];b=x['decisions']['FUSION']==x['expected']
        pairs[('both_correct' if a and b else 'corrected_by_fusion' if b else 'lost_by_fusion' if a else 'both_wrong')]+=1
    negatives={layer:sum(x['decisions'][layer] in PRIMARY-{None} for x in negative)
               for layer in ['DATA_ONLY','DATA_PLUS_REFERENCE_IDENTITY','KG_ONLY','FUSION_NO_TIE','FUSION']}
    return {'source':str(directory),'primary_terminal':measure(primary),'paired_reference_control_comparison':dict(pairs),
            'negative_controls':{'n':len(negative),'false_in_scope_diagnoses':negatives,'detail':negative},
            'primary_pre_event_windows':measure([x for x in temporal if x['primary_scope'] and not x['fault_active']]),
            'primary_active_windows':measure([x for x in temporal if x['primary_scope'] and x['fault_active']]),
            'terminal_rows':terminal,'all_window_rows':temporal,
            'metric_boundary':'Abstention and unavailable count as incorrect in exact accuracy. NORMAL_REFERENCE_MATCH uses identical paired telemetry, not missing candidate. Windows and paired fault scenarios are dependent. KG has extra matched-reference input; compare DATA_PLUS_REFERENCE_IDENTITY and report KG_ONLY. No LLM invoked by these replayed diagnosis records.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.evaluation,a.dataset);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['all_window_rows','terminal_rows','primary_active_windows','primary_pre_event_windows']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
