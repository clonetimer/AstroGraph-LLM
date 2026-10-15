#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from astrograph.agentization.lineage_resolution import Step6AssessmentInput, assess_step6_route

def main() -> int:
    p=argparse.ArgumentParser(description="Assess the evidence-feasible AstroGraph Step-6 route without reading outcomes.")
    p.add_argument("input", type=Path); p.add_argument("--output", type=Path, required=True)
    a=p.parse_args(); payload=json.loads(a.input.read_text(encoding="utf-8"))
    result=assess_step6_route(Step6AssessmentInput.model_validate(payload))
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result.model_dump(mode="json"),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return 0
if __name__=="__main__": raise SystemExit(main())
