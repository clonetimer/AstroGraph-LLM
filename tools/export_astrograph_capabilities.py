#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from astrograph.agentization.capabilities import CapabilityEnvironment, build_registry
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project-root",type=Path,default=Path.cwd()); ap.add_argument("--output",type=Path,required=True); ns=ap.parse_args()
    root=ns.project_root.resolve(); env=CapabilityEnvironment(root,(root,),root/"research/agentization/runtime_artifacts",root/"configs/agentization/workflows",(root,))
    ns.output.parent.mkdir(parents=True,exist_ok=True); ns.output.write_text(json.dumps(build_registry(env).manifest(),indent=2,sort_keys=True)+"\n",encoding="utf-8")
if __name__=="__main__": main()
