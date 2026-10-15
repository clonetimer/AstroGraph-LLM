"""Offline parallel fitting after mode acceptance; not a product DAG engine.

Example: PYTHONPATH=src python scripts/run_mode_benchmark.py --data ../data \
  --output ../runs --workers 3 --acceptance-junit ../mode-acceptance.xml
Normal-mode replays and the final offline truth join are separate commands.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--acceptance-junit',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.workers<=max(1,os.cpu_count() or 1):raise ValueError('invalid worker count')
    tree=ET.parse(args.acceptance_junit)
    suites=list(tree.iter('testsuite'))
    if not suites or any(int(s.get(k,'0')) for s in suites for k in ('errors','failures')):
        raise ValueError('successful mode acceptance is required before fitting')
    cases=list(tree.iter('testcase'))
    if not any('test_modes' in c.get('classname','') for c in cases):
        raise ValueError('acceptance receipt must contain actual work-mode tests')
    if any(c.find('failure') is not None or c.find('error') is not None or c.find('skipped') is not None for c in cases if 'test_modes' in c.get('classname','')):
        raise ValueError('mode acceptance must not be skipped')
    root=Path(__file__).resolve().parents[1];data=args.data.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False);(output/'logs').mkdir()
    protocol=json.loads((data/'protocol.json').read_text());lock=threading.Lock()
    env={**os.environ,'PYTHONPATH':str(root/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    def event(row):
        with lock,(output/'events.jsonl').open('a') as f:f.write(json.dumps({'at_unix':time.time(),**row})+'\n')
    def fit(family,seed):
        key=family+'__'+str(seed)
        command=[sys.executable,str(root/'scripts/train_mode_benchmark.py'),'--data',str(data),'--output',str(output/key),'--family',family,'--seed',str(seed),'--threads','1']
        event({'stage':'START','key':key,'command':command})
        with (output/'logs'/f'{key}.txt').open('w') as log:
            completed=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        row={'stage':'END','key':key,'returncode':completed.returncode};event(row);return row
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(fit,f,s) for f in protocol['families'] for s in protocol['seeds']]
        results=[f.result() for f in as_completed(jobs)]
    (output/'STATUS.json').write_text(json.dumps(results,indent=2)+'\n')
    return 0 if all(r['returncode']==0 for r in results) else 1

if __name__=='__main__':raise SystemExit(main())
