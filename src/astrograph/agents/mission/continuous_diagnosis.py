"""Append-only case timelines and deterministic next-best-observation proposals."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from astrograph.evidence.executable import stable_id
from .storage import sha


class _Strict(BaseModel):
    model_config=ConfigDict(extra="forbid", frozen=True)


class CaseTimelineEvent(_Strict):
    schema_version: Literal["astrograph-case-timeline-event.v1"]="astrograph-case-timeline-event.v1"
    event_id: str
    case_id: str
    task_id: str
    observed_at: str
    diagnosis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    detected: bool
    selected_fault_id: str | None=None
    candidate_fault_ids: tuple[str,...]=()
    input_data_sha256: str | None=Field(default=None,pattern=r"^[0-9a-f]{64}$")
    model_refs: tuple[str,...]=()
    knowledge_version: str | None=None
    metadata: dict[str,Any]=Field(default_factory=dict)


class CaseTimelineStore:
    """Content-addressed append-only case history for cross-window diagnosis."""
    def __init__(self, root: str|Path):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)

    def _path(self,case_id:str)->Path:
        safe="".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in case_id)
        return self.root/f"{safe}.jsonl"

    def events(self,case_id:str)->list[CaseTimelineEvent]:
        p=self._path(case_id)
        if not p.is_file(): return []
        return [CaseTimelineEvent.model_validate(json.loads(x)) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

    def append(self,event:CaseTimelineEvent)->Path:
        p=self._path(event.case_id);existing={x.event_id:x for x in self.events(event.case_id)}
        if event.event_id in existing:
            if existing[event.event_id]!=event: raise ValueError("timeline event ID reused with different content")
            return p
        with p.open("a",encoding="utf-8") as f: f.write(json.dumps(event.model_dump(mode="json"),ensure_ascii=False)+"\n")
        return p

    def trajectory(self,case_id:str)->dict[str,Any]:
        events=self.events(case_id)
        revisions=0
        for left,right in zip(events,events[1:]):
            if left.selected_fault_id!=right.selected_fault_id: revisions+=1
        return {"case_id":case_id,"event_count":len(events),"diagnosis_revision_count":revisions,
                "latest":events[-1].model_dump(mode="json") if events else None,
                "events":[x.model_dump(mode="json") for x in events]}


def make_timeline_event(*,case_id:str,task_id:str,observed_at:str,diagnosis:dict[str,Any],
                        input_data_sha256:str|None=None,model_refs:list[str]|None=None,
                        knowledge_version:str|None=None,metadata:dict[str,Any]|None=None)->CaseTimelineEvent:
    body=diagnosis.get("diagnosis",diagnosis)
    selected=body.get("selected_fault_id") or body.get("conclusion_fault_id")
    candidates=tuple(dict.fromkeys([x for x in (body.get("candidate_fault_ids") or []) if x]))
    detected=bool(body.get("detected", selected or candidates))
    digest=sha(diagnosis)
    event_id=stable_id("TL",{"case_id":case_id,"task_id":task_id,"observed_at":observed_at,"diagnosis_sha256":digest})
    return CaseTimelineEvent(event_id=event_id,case_id=case_id,task_id=task_id,observed_at=observed_at,
        diagnosis_sha256=digest,detected=detected,selected_fault_id=selected,candidate_fault_ids=candidates,
        input_data_sha256=input_data_sha256,model_refs=tuple(model_refs or []),knowledge_version=knowledge_version,
        metadata=metadata or {})


def recommend_next_observations(*,candidate_fault_ids:list[str],observed_channels:list[str],
                                fault_catalog_path:str|Path,top_k:int=5)->dict[str,Any]:
    candidates=list(dict.fromkeys(x for x in candidate_fault_ids if x))
    if not candidates:
        return {"status":"NOT_APPLICABLE","recommendations":[],"reason":"NO_FAULT_CANDIDATES"}
    catalog=json.loads(Path(fault_catalog_path).read_text(encoding="utf-8"))
    faults={str(x.get("fault_id")):x for x in catalog.get("fault_modes",[]) if x.get("fault_id") in candidates}
    observed=set(observed_channels);by_channel:dict[str,dict[str,dict[str,Any]]]={}
    role_weight={"PRIMARY":3.0,"SUPPORTING":2.0,"CONTEXT":1.0}
    for fault_id,fault in faults.items():
        for row in fault.get("telemetry_channels",[]):
            ch=str(row.get("channel","")).strip()
            if not ch or ch in observed: continue
            by_channel.setdefault(ch,{})[fault_id]={"role":str(row.get("role","SUPPORTING")).upper(),
                "expected_effect":str(row.get("expected_effect","")),"required":bool(row.get("required",False))}
    scored=[]
    for ch,signatures in by_channel.items():
        signature_tokens=[]
        for fid in candidates:
            sig=signatures.get(fid)
            signature_tokens.append("ABSENT" if sig is None else f"{sig['role']}|{sig['expected_effect']}")
        diversity=len(set(signature_tokens))
        required_count=sum(int(x.get("required",False)) for x in signatures.values())
        max_role=max((role_weight.get(x.get("role",""),0.0) for x in signatures.values()),default=0.0)
        score=float((diversity-1)*3 + required_count*2 + max_role)
        if len(candidates)>1 and diversity<=1: continue
        scored.append({"channel":ch,"priority_score":score,"candidate_signatures":{
            fid:signatures.get(fid,{"role":"ABSENT","expected_effect":"","required":False}) for fid in candidates},
            "reason":"DISCRIMINATES_COMPETING_FAULTS" if len(candidates)>1 else "CONFIRMS_SINGLE_HYPOTHESIS"})
    scored.sort(key=lambda x:(-x["priority_score"],x["channel"]))
    return {"schema_version":"astrograph-next-best-observation.v1",
        "status":"AVAILABLE" if scored else "INSUFFICIENT_KNOWLEDGE",
        "candidate_fault_ids":candidates,"observed_channels":sorted(observed),"recommendations":scored[:top_k],
        "control_action_authorized":False,
        "claim_boundary":"Knowledge-based observation recommendation only; it requests evidence and never commands spacecraft control."}


__all__=["CaseTimelineEvent","CaseTimelineStore","make_timeline_event","recommend_next_observations"]
