
"""Independent-group summaries for provenance-blocked coverage-response studies."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import math
from statistics import mean
from typing import Sequence
@dataclass(frozen=True)
class CoveragePoint:
    policy_id:str; replicate_id:int; group_id:int; target_fraction:float; achieved_capability_fraction:float; metric_value:float
    def __post_init__(self):
        if not 0<=self.target_fraction<=1 or not 0<=self.achieved_capability_fraction<=1: raise ValueError("fractions must be in [0,1]")
        if not math.isfinite(self.metric_value): raise ValueError("metric must be finite")
@dataclass(frozen=True)
class CoverageResponseSummary:
    policy_id:str; replicate_id:int; group_count:int; levels:tuple[float,...]; group_macro_metric:tuple[float,...]; area_under_response:float; net_gain_full_minus_zero:float; monotonicity_violations:int

def summarize_coverage_response(points:Sequence[CoveragePoint])->CoverageResponseSummary:
    if not points: raise ValueError("points required")
    policies={p.policy_id for p in points}; reps={p.replicate_id for p in points}
    if len(policies)!=1 or len(reps)!=1: raise ValueError("one policy and one intervention replicate per summary")
    seen=set()
    for p in points:
        key=(p.group_id,p.achieved_capability_fraction)
        if key in seen: raise ValueError("duplicate group/coverage record; windows are not independent replicates")
        seen.add(key)
    levels=sorted({p.achieved_capability_fraction for p in points})
    groups={p.group_id for p in points}
    for level in levels:
        if {p.group_id for p in points if p.achieved_capability_fraction==level}!=groups: raise ValueError("every level must contain the same independent groups")
    ys=[mean(p.metric_value for p in points if p.achieved_capability_fraction==x) for x in levels]
    auc=sum((levels[i]-levels[i-1])*(ys[i]+ys[i-1])/2 for i in range(1,len(levels)))
    violations=sum(1 for a,b in zip(ys,ys[1:]) if b<a)
    return CoverageResponseSummary(next(iter(policies)),next(iter(reps)),len(groups),tuple(levels),tuple(ys),auc,ys[-1]-ys[0],violations)
