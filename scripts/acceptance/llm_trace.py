#!/usr/bin/env python3
"""Verify a saved task's actual prompt/response hashes offline (never generate)."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from astrograph.llm.trace_verification import verify_task

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new output file; old receipts are immutable')
    try:
        result=verify_task(a.task_dir)
    except Exception as exc:
        result={'status':'FAILED','error_type':type(exc).__name__,'reason':str(exc),'generation_requests_sent':0}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'output':str(a.output)},ensure_ascii=False))
    return 0 if result['status']=='VERIFIED' else 2
if __name__=='__main__':raise SystemExit(main())
