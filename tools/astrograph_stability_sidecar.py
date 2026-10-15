#!/usr/bin/env python3
"""Minimal NDJSON sidecar for Step-6 and stability capabilities."""
from __future__ import annotations
import json, sys
from astrograph.agentization.stability_capabilities import invoke_stability_capability

def main() -> int:
    for line in sys.stdin:
        try:
            req=json.loads(line); rid=req.get("id")
            result=invoke_stability_capability(req["capability"], req.get("payload",{}))
            out={"id":rid,"ok":True,"result":result}
        except Exception as exc:
            out={"id":req.get("id") if isinstance(req,dict) else None,"ok":False,"error":{"type":type(exc).__name__,"message":str(exc)}}
        print(json.dumps(out,sort_keys=True),flush=True)
    return 0
if __name__=="__main__": raise SystemExit(main())
