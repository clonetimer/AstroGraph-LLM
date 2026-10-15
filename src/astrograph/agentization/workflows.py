from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .protocol import CapabilityProtocolError, sha256_json

def load_workflows(config_root: Path) -> dict[str,dict[str,Any]]:
    out={}
    if not config_root.exists(): return out
    for p in sorted(config_root.glob("*.json")):
        raw=json.loads(p.read_text(encoding="utf-8"))
        wid=str(raw.get("workflow_id",p.stem))
        raw["content_sha256"]=sha256_json({k:v for k,v in raw.items() if k!="content_sha256"})
        out[wid]=raw
    return out

def validate_workflow(raw: dict[str,Any]) -> dict[str,Any]:
    nodes=raw.get("nodes"); edges=raw.get("edges")
    if not isinstance(nodes,list) or not isinstance(edges,list):
        raise CapabilityProtocolError("INVALID_WORKFLOW","nodes and edges must be arrays")
    ids=[str(n.get("id","")) for n in nodes if isinstance(n,dict)]
    if any(not x for x in ids) or len(ids)!=len(set(ids)):
        raise CapabilityProtocolError("INVALID_WORKFLOW_NODES","node IDs must be non-empty and unique")
    graph={x:[] for x in ids}; indeg={x:0 for x in ids}
    for e in edges:
        if not isinstance(e,dict): raise CapabilityProtocolError("INVALID_WORKFLOW_EDGE","edge must be an object")
        a,b=str(e.get("from","")),str(e.get("to",""))
        if a not in graph or b not in graph: raise CapabilityProtocolError("UNKNOWN_WORKFLOW_NODE",f"edge references unknown node: {a}->{b}")
        graph[a].append(b); indeg[b]+=1
    queue=sorted([n for n,d in indeg.items() if d==0]); seen=[]
    while queue:
        n=queue.pop(0); seen.append(n)
        for m in graph[n]:
            indeg[m]-=1
            if indeg[m]==0: queue.append(m); queue.sort()
    if len(seen)!=len(ids): raise CapabilityProtocolError("WORKFLOW_CYCLE","workflow must be acyclic")
    return {"valid":True,"node_count":len(ids),"edge_count":len(edges),"topological_order":seen,"content_sha256":sha256_json(raw)}
