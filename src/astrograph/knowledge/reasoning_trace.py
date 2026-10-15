
"""Content-addressed, label-free trace for faithful multistage reasoning."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from enum import Enum
import hashlib,json,re
from typing import Any,Mapping,Sequence
_SHA=re.compile(r"^[0-9a-f]{64}$")
_BANNED={"ground_truth","true_label","fault_label","y_true","target","correctness","accuracy"}
class ReasoningStage(str,Enum):
    TEMPORAL_LOCALIZATION="TEMPORAL_LOCALIZATION"; CANDIDATE_RETRIEVAL="CANDIDATE_RETRIEVAL"; EVIDENCE_EXECUTION="EVIDENCE_EXECUTION"; CAUSAL_CONSISTENCY="CAUSAL_CONSISTENCY"; FUSION="FUSION"; EXPLANATION="EXPLANATION"
_ORDER={s:i for i,s in enumerate(ReasoningStage)}
def _reject(v:Any)->None:
    if isinstance(v,Mapping):
        for k,c in v.items():
            if str(k).strip().lower() in _BANNED: raise ValueError(f"truth/outcome key forbidden in runtime trace: {k}")
            _reject(c)
    elif isinstance(v,(list,tuple)):
        for c in v:_reject(c)
@dataclass(frozen=True)
class StageRecord:
    stage:ReasoningStage; input_artifact_sha256s:tuple[str,...]; output_artifact_sha256:str; executed_evidence_ids:tuple[str,...]=(); metadata:Mapping[str,Any]=None
    def __post_init__(self):
        for d in self.input_artifact_sha256s+(self.output_artifact_sha256,):
            if not _SHA.fullmatch(d): raise ValueError("stage artifact hashes must be lowercase SHA-256")
        if len(set(self.executed_evidence_ids))!=len(self.executed_evidence_ids): raise ValueError("executed evidence IDs must be unique")
        _reject(self.metadata or {})
@dataclass(frozen=True)
class MultistageReasoningTrace:
    schema:str; trajectory_id:str; records:tuple[StageRecord,...]; trace_sha256:str|None=None
    def __post_init__(self):
        if self.schema!="astrograph-multistage-reasoning-trace.v1": raise ValueError("unsupported trace schema")
        if not self.trajectory_id or not self.records: raise ValueError("trajectory and records required")
        orders=[_ORDER[r.stage] for r in self.records]
        if orders!=sorted(orders) or len(orders)!=len(set(orders)): raise ValueError("reasoning stages must be unique and ordered")
        executed=set()
        for r in self.records:
            if r.stage is ReasoningStage.EVIDENCE_EXECUTION: executed|=set(r.executed_evidence_ids)
            elif r.stage is ReasoningStage.EXPLANATION:
                cited=set((r.metadata or {}).get("cited_evidence_ids",()))
                if not cited.issubset(executed): raise ValueError("explanation cites evidence not executed in this trace")
        expected=self.compute_sha256()
        if self.trace_sha256 is not None and self.trace_sha256!=expected: raise ValueError("trace_sha256 mismatch")
    def payload(self)->dict[str,Any]:
        rows=[]
        for r in self.records:
            d=asdict(r); d["stage"]=r.stage.value; rows.append(d)
        return {"schema":self.schema,"trajectory_id":self.trajectory_id,"records":rows}
    def compute_sha256(self)->str:
        raw=json.dumps(self.payload(),sort_keys=True,separators=(",",":"),ensure_ascii=False).encode(); return hashlib.sha256(raw).hexdigest()
