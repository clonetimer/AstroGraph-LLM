#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from astrograph.agentization.stable_release import StabilityInput, assess_stability

def main() -> int:
    p=argparse.ArgumentParser(description="Assess AstroGraph software/research release stability.")
    p.add_argument("input", type=Path); p.add_argument("--output", type=Path, required=True)
    a=p.parse_args(); result=assess_stability(StabilityInput.model_validate_json(a.input.read_text(encoding="utf-8")))
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result.model_dump(mode="json"),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return 0
if __name__=="__main__": raise SystemExit(main())
