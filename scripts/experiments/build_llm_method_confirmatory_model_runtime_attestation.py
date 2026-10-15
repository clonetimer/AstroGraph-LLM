#!/usr/bin/env python3
"""Recompute a content-bound model/tokenizer/server runtime attestation from a governed spec."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from astrograph.evaluation.confirmatory_runtime_identity import build_model_runtime_attestation, load_model_runtime_spec

def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--spec",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    att=build_model_runtime_attestation(load_model_runtime_spec(a.spec)); path=a.output.resolve(); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(att,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":att["status"],"runtime_identity_sha256":att["runtime_identity_sha256"],"path":str(path)},ensure_ascii=False,indent=2)); return 0 if att["status"]=="VALID" else 2
if __name__=="__main__": raise SystemExit(main())
