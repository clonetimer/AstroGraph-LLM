"""Read-only dependency and optional capability probe; never installs or fabricates modules."""
from __future__ import annotations
import argparse
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
from packaging.requirements import Requirement

ROOT=Path(__file__).resolve().parents[1]

def inspect() -> dict:
    packages={};pending=['langchain','langgraph'];visited=set()
    while pending:
        name=pending.pop(0)
        canonical=name.lower().replace('_','-')
        if canonical in visited:continue
        visited.add(canonical)
        try:dist=metadata.distribution(name)
        except metadata.PackageNotFoundError:
            packages[canonical]={'installed':False};continue
        row={'installed':True,'version':dist.version,'unsatisfied':[]}
        for text in dist.requires or []:
            req=Requirement(text)
            if req.marker and not req.marker.evaluate({'extra':''}):continue
            try:version=metadata.version(req.name)
            except metadata.PackageNotFoundError:version=None
            if version is None or (req.specifier and not req.specifier.contains(version,prereleases=True)):
                row['unsatisfied'].append({'requirement':str(req),'installed_version':version})
            pending.append(req.name)
        packages[canonical]=row
    try:
        from langgraph.graph import StateGraph
        langgraph={'status':'AVAILABLE','import_verified':True}
    except ImportError as exc:
        langgraph={'status':'BLOCKED_DEPENDENCY','import_verified':False,'missing_module':exc.name}
    dor=shutil.which('dor');dor_status={'executable_found':bool(dor),'active_discovery_cli_registered':False}
    if dor:
        try:
            result=subprocess.run([dor,'--help'],capture_output=True,text=True,timeout=20)
            dor_status.update(help_exit_code=result.returncode,
                active_discovery_cli_registered='active-discovery-advisory' in result.stdout)
        except subprocess.TimeoutExpired:dor_status['status']='HELP_TIMEOUT'
    node=shutil.which('node');ppl={'node_available':bool(node),'profile_runtime_installed':False}
    if node:
        check=subprocess.run([node,'--input-type=module','-e',"import('@ppl/profile-runtime').then(()=>console.log('ok'))"],
                             cwd=ROOT/'integrations/ppl',capture_output=True,text=True,timeout=15)
        ppl['profile_runtime_installed']=check.returncode==0
    return {'schema_version':'astrograph-implementation-environment.v1','python_version':sys.version,
            'native_dag':'AVAILABLE','langgraph':langgraph,'packages':packages,'dor':dor_status,'ppl':ppl,
            'scope':'Dependency metadata and import checks only; not model accuracy or security attestation.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);parser.add_argument('--require-langgraph',action='store_true')
    args=parser.parse_args();result=inspect();text=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding='utf-8')
    print(text)
    return 2 if args.require_langgraph and result['langgraph']['status']!='AVAILABLE' else 0
if __name__=='__main__':raise SystemExit(main())
