"""Offline portable package checks. No credentials, user documents, or network calls."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys


def verify_files(root: Path) -> dict:
    root=Path(root).resolve()
    failures=[]; checked=0
    try:
        manifest_path=root/'MANIFEST.json'
        if manifest_path.stat().st_size > 4*1024*1024:
            raise ValueError('oversized manifest')
        manifest=json.loads(manifest_path.read_text('utf-8'))
        entries=manifest['files']
        if not isinstance(entries,list) or not 1 <= len(entries) <= 20000:
            raise ValueError('invalid file list')
        seen=set()
        for entry in entries:
            name=entry['path'];size=entry['bytes'];digest=entry['sha256']
            parts=name.split('/') if isinstance(name,str) else []
            if (not parts or name.casefold() in seen or PurePosixPath(name).is_absolute()
                    or any(part in {'','.','..'} for part in parts)
                    or re.search(r'[\\:\x00-\x1f]',name) or type(size) is not int or size < 0
                    or not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest)):
                raise ValueError('invalid manifest entry')
            seen.add(name.casefold());path=root/name
            if not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in [path,*path.parents] if p!=root.parent):
                failures.append({'path':name,'code':'LINK_OR_ESCAPE'});continue
            if not path.is_file():
                failures.append({'path':name,'code':'MISSING'});continue
            if path.stat().st_size!=size:
                failures.append({'path':name,'code':'SIZE_MISMATCH'});continue
            with path.open('rb') as handle:
                actual=hashlib.file_digest(handle,'sha256').hexdigest()
            if actual!=digest: failures.append({'path':name,'code':'HASH_MISMATCH'})
            checked+=1
    except (OSError,ValueError,TypeError,KeyError):
        failures.append({'path':'MANIFEST.json','code':'INVALID_OR_MISSING_MANIFEST'})
    return {'status':'FAIL' if failures else 'PASS','checked_files':checked,'failures':failures}


def run_checks(root: Path, *, report_path: Path) -> int:
    root=root.resolve()
    evidence={'schema_version':'astrograph-installation-check.v1',
              'checked_at':datetime.now(timezone.utc).isoformat(),
              'python':platform.python_version(),'platform':sys.platform,
              'windows_version':platform.version() if os.name=='nt' else None,
              'bits':64 if sys.maxsize>2**32 else 32,
              'real_model_called':False,'internet_required':False}
    evidence['integrity']=verify_files(root)
    # Native Windows runtime verification is separate from source-only checks.
    runtime_ok=(os.name=='nt' and sys.maxsize>2**32 and sys.version_info[:2]==(3,14))
    evidence['runtime']={'status':'PASS' if runtime_ok else 'FAIL','code':'WINDOWS_X64_PY314' if runtime_ok else 'UNEXPECTED_RUNTIME'}
    if evidence['integrity']['status']=='PASS' and runtime_ok:
        try:
            proc=subprocess.run([sys.executable,'-X','utf8','-B','-u',str(root/'app/workbench.py'),'--self-test'],
                                cwd=root,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
            # This subprocess uses only an internal stub; it never receives user settings or secrets.
            evidence['self_test']={'status':'PASS' if proc.returncode==0 else 'FAIL',
                                   'exit_code':proc.returncode,'output':proc.stdout[-12000:]}
        except subprocess.TimeoutExpired:
            evidence['self_test']={'status':'FAIL','code':'SELF_TEST_TIMEOUT'}
        except OSError:
            evidence['self_test']={'status':'FAIL','code':'SELF_TEST_START_FAILED'}
    else:
        evidence['self_test']={'status':'NOT_RUN','code':'PACKAGE_OR_RUNTIME_CHECK_FAILED'}
    evidence['status']='PASS' if all(evidence[k]['status']=='PASS' for k in ['integrity','runtime','self_test']) else 'FAIL'
    try:
        report_path.parent.mkdir(parents=True,exist_ok=True)
        temporary=report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n','utf-8')
        temporary.replace(report_path)
    except OSError:
        print('Could not write diagnostics. Extract the ZIP into a writable folder.')
        print(json.dumps(evidence,ensure_ascii=True,indent=2));return 1
    print('AstroGraph package check: '+evidence['status'])
    print('Checked files: '+str(evidence['integrity']['checked_files']))
    for failure in evidence['integrity']['failures'][:20]:print(failure['code']+': '+failure['path'])
    print('Diagnostic report: '+str(report_path))
    return 0 if evidence['status']=='PASS' else 1


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    return run_checks(args.root,report_path=args.report or args.root/'data/diagnostics/installation-check.json')

if __name__=='__main__':raise SystemExit(main())
