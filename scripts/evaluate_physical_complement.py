"""Truth-free physical challenger applied to verified cached native windows."""
from pathlib import Path
import argparse,hashlib,json
import pandas as pd
from astrograph.oag.physical_diagnosis import diagnose_physical_complement
if __package__:
    from .evaluate_diagnostic_enhancement import load_run,measure
else:
    from evaluate_diagnostic_enhancement import load_run,measure


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((a.baseline/'plan.json').read_text());paths=sorted(a.dataset.glob('**/telemetry.csv'))
    assert [hashlib.sha256(f.read_bytes()).hexdigest() for f in paths]==plan['telemetry_sha256'],'cached inference input differs'
    manifest=json.loads((a.dataset/'dataset_manifest.json').read_text());missions={s['scenario_id']:s['mission_id'] for s in manifest['operating_scenarios']}
    completed={}
    records=json.loads((a.baseline/'evaluation.json').read_text())['records']
    for c in records:
        path=a.dataset/c['source_path'];parts=Path(c['source_path']).parts
        mission=parts[1] if parts[0]=='nominal_baseline' else missions[parts[1]]
        source=pd.read_csv(path,low_memory=False);nominal=pd.read_csv(a.dataset/'nominal_baseline'/mission/path.parent.name/'telemetry.csv',low_memory=False)
        windows=[]
        for w in json.loads((a.baseline/c['case_id']/'windows.json').read_text()):
            result=diagnose_physical_complement(source[source.time_s<=w['endpoint_s']+1e-9],nominal,w.get('summary') if w['status']=='SUCCEEDED' else None)
            key=(c['source_path'],w['endpoint_s']);completed[key]=result
            windows.append({'endpoint_s':w['endpoint_s'],'result':result})
        (a.output/(c['case_id']+'.json')).write_text(json.dumps(windows,indent=2)+'\n')
    # Ground truth joins only after every diagnosis has finished.
    base=load_run(a.baseline,a.dataset);new={k:{**r,'prediction':completed[k]['prediction'],'degraded':None} for k,r in base.items()}
    report={'baseline':{'strict':measure(base),'effect':measure(base,effect=True)},'physical':{'strict':measure(new),'effect':measure(new,effect=True)},'rows':[]}
    for k,r in new.items():
        report['rows'].append({'source':k[0],'endpoint_s':k[1],'target':r['target'],'family':r['family'],'event_start_s':r['event_start_s'],'before':base[k]['prediction'],'after':r['prediction'],'route':completed[k]['decision_route']})
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in d['effect'].items() if k!='by_family'} for name,d in report.items() if name!='rows'},indent=2))

if __name__=='__main__':main()
